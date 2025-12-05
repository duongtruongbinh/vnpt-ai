"""Logic solver node implementing a Manual Code Execution workflow."""

import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_experimental.utilities import PythonREPL

from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.llm import get_large_model
from src.utils.logging import print_log

_python_repl = PythonREPL()

CODE_AGENT_PROMPT = """Nhiệm vụ của bạn là giải các câu hỏi trắc nghiệm bằng cách viết mã Python thực thi được.

QUY TẮC BẮT BUỘC:
1. Viết script Python giải quyết vấn đề, tự động import thư viện cần thiết.
2. Code phải tự động tính toán ra kết quả, KHÔNG được hardcode đáp án.
3. Cuối đoạn code, phải có logic so sánh kết quả tính được với các lựa chọn (A, B, C, D).
4. In kết quả cuối cùng theo định dạng CHÍNH XÁC sau:
   print("Đáp án: X") 
   (Trong đó X là ký tự A, B, C hoặc D).

VÍ DỤ MẪU:
Câu hỏi: 15% của 200 là bao nhiêu? A. 20, B. 30...
Output mong đợi:
```python
value = 200 * 0.15
print(f"Calculated: {value}")

options = {"A": 20, "B": 30, "C": 40, "D": 50}
for key, val in options.items():
    if value == val:
        print(f"Đáp án: {key}")
        break
```
        
Chỉ trả về block code Python, không giải thích thêm."""


def extract_python_code(text: str) -> str | None:
    """Find and extract Python code from block ``` python ...   ```"""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_answer(text: str, max_choices: int = 26) -> str | None:
    """Find 'Đáp án: X' in the text response"""
    valid_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:max_choices]
    match = re.search(r"Đáp án:\s*([A-Z])", text, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if answer in valid_labels:
            return answer
    return None


def _indent_code(code: str) -> str:
    """Format code to make it easier to read in the terminal"""
    return "\n".join(f"        {line}" for line in code.splitlines())


def logic_solver_node(state: GraphState) -> dict:
    """Solve math/logic questions using Python code execution."""
    llm = get_large_model()
    all_choices = get_choices_from_state(state)
    choices_text = format_choices(all_choices)

    question_content = f"Câu hỏi: {state['question']}\n{choices_text}"

    messages: list[BaseMessage] = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=question_content)
    ]

    max_steps = 5
    for step in range(max_steps):
        response = llm.invoke(messages)
        content = response.content
        messages.append(response)

        code_block = extract_python_code(content)

        if code_block:
            print_log(f"        [Logic] Step {step+1}: Found Python code. Executing...")
            print_log(f"        [Logic] Code:\n{_indent_code(code_block)}")

            try:
                if "print" not in code_block:
                    lines = code_block.splitlines()
                    if lines:
                        last_line = lines[-1]
                        if "=" in last_line:
                            var_name = last_line.split("=")[0].strip()
                        else:
                            var_name = last_line.strip()
                        code_block += f"\nprint({var_name})"

                output = _python_repl.run(code_block)
                output = output.strip() if output else "No output."
                print_log(f"        [Logic] Code output: {output}")

                code_ans = extract_answer(output, max_choices=len(all_choices) or 4)
                if code_ans:
                    print_log(f"        [Logic] Final Answer: {code_ans}")
                    return {"answer": code_ans}

                feedback_msg = f"Kết quả chạy code: {output}.\n"
                feedback_msg += "Lưu ý: Bạn vẫn chưa đưa ra đáp án cuối cùng, duyệt lại code và các đáp án để chỉnh sửa phù hợp."

                messages.append(HumanMessage(content=feedback_msg))

            except Exception as e:
                error_msg = f"Error running code: {str(e)}"
                print_log(f"        [Error] {error_msg}")
                messages.append(HumanMessage(content=f"{error_msg}. Hãy kiểm tra logic và sửa lại code."))

            continue

        if step < max_steps - 1:
            print_log("        [Warning] No code or answer found. Reminding model...")
            messages.append(HumanMessage(content="Lưu ý: Bạn vẫn chưa đưa ra đáp án cuối cùng, duyệt lại code và các đáp án để chỉnh sửa phù hợp."))

    print_log("        [Warning] Max steps reached. Defaulting to A.")
    return {"answer": "A"}
