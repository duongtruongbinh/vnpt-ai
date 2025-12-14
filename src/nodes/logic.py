"""Logic solver node implementing a Manual Code Execution workflow."""

import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_experimental.utilities import PythonREPL

from src.data_processing.answer import extract_answer
from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.llm import get_large_model
from src.utils.logging import print_log
from src.utils.prompts import load_prompt

_python_repl = PythonREPL()


def extract_python_code(text: str) -> str | None:
    """Find and extract Python code from block ``` python ...   ```"""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _validate_code_syntax(code: str) -> tuple[bool, str]:
    """Check if code has valid Python syntax. Returns (is_valid, error_message)."""
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def _is_placeholder_code(code: str) -> bool:
    """Check if code contains placeholders or is incomplete."""
    if not code or len(code.strip()) < 10:
        return True
    if "..." in code:
        return True
    # Check for {key}-style placeholders (but not f-string or dict literals)
    if re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", code):
        # Exclude common dict/set patterns and f-strings
        if not re.search(r'["\'][^"\']*\{[a-zA-Z_]', code):
            return True
    return False


def _indent_code(code: str) -> str:
    """Format code to make it easier to read in the terminal."""
    return "\n".join(f"        {line}" for line in code.splitlines())


def _fallback_text_reasoning(llm, question: str, choices_text: str, max_choices: int) -> dict:
    """Fallback to CoT reasoning when code execution fails."""
    print_log("        [Logic] Falling back to CoT reasoning...")

    fallback_system = (
        "Nhiệm vụ của bạn là trả lời câu hỏi "
        "được đưa ra bằng khả năng phân tích và suy luận logic. "
        "Hãy phân tích vấn đề và suy luận đề từng bước một. " 
        "Cuối cùng, hãy trả lời theo đúng định dạng: 'Đáp án: X' "
        "trong đó X là ký tự đại diện cho lựa chọn đúng (A, B, C, D, ...)."
    )

    fallback_user = (
        f"Câu hỏi: {question}\n"
        f"{choices_text}"
    )

    fallback_messages: list[BaseMessage] = [
        SystemMessage(content=fallback_system),
        HumanMessage(content=fallback_user)
    ]

    fallback_response = llm.invoke(fallback_messages)
    fallback_content = fallback_response.content
    print_log(f"        [Logic] Fallback response received.")

    return {"text": fallback_content}


def _request_final_answer(llm, question: str, choices_text: str, computed_results: str) -> str:
    """Request a strict final answer from the model."""
    system_prompt = (
        "Bạn là trợ lý AI. Dựa vào kết quả tính toán được cung cấp, "
        "hãy đưa ra đáp án cuối cùng. CHỈ trả lời đúng một dòng: Đáp án: X "
        "(trong đó X là A, B, C hoặc D)."
    )
    user_prompt = (
        f"Câu hỏi: {question}\n"
        f"{choices_text}\n"
        f"Kết quả tính toán: {computed_results}\n\n"
        "Trả lời đúng một dòng: Đáp án: X"
    )
    
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content


def logic_solver_node(state: GraphState) -> dict:
    """Solve math/logic questions using Python code execution."""
    llm = get_large_model()
    all_choices = get_choices_from_state(state)
    max_choices = len(all_choices) or 4
    choices_text = format_choices(all_choices)

    system_prompt = load_prompt("logic_solver.j2", "system")
    user_prompt = load_prompt("logic_solver.j2", "user", question=state["question"], choices=choices_text)

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    step_texts: list[str] = []
    computed_outputs: list[str] = []

    max_steps = 5
    for step in range(max_steps):
        response = llm.invoke(messages)
        content = response.content
        step_texts.append(content)
        messages.append(response)

        code_block = extract_python_code(content)

        if code_block:
            if _is_placeholder_code(code_block):
                print_log(f"        [Logic] Step {step+1}: Placeholder code detected. Requesting complete code...")
                regen_msg = (
                    "Code không hợp lệ (chứa placeholder hoặc không đầy đủ). "
                    "Hãy cung cấp code Python hoàn chỉnh, có thể chạy được, không chứa '...' hay placeholder. "
                    "In ra các giá trị tính toán được. "
                    "Cuối cùng, kết thúc bằng một dòng duy nhất: Đáp án: X (X là A, B, C hoặc D)."
                )
                messages.append(HumanMessage(content=regen_msg))
                continue
            
            print_log(f"        [Logic] Step {step+1}: Found Python code. Executing...")
            
            # Validate syntax before execution
            is_valid, syntax_error = _validate_code_syntax(code_block)
            if not is_valid:
                print_log(f"        [Error] Syntax error detected: {syntax_error}")
                error_msg = f"SyntaxError: {syntax_error}. "
                error_msg += "Lưu ý: KHÔNG sử dụng các từ khóa Python như 'lambda', 'class', 'def' làm tên biến. "
                error_msg += "Hãy đổi tên biến và thử lại."
                messages.append(HumanMessage(content=error_msg))
                continue
            
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
                computed_outputs.append(output)

                # Do NOT extract answer from code output directly
                # Instead, feed output back to model and ask for final answer line
                feedback_msg = (
                    f"Kết quả thực thi code: {output}\n\n"
                    "Dựa vào kết quả trên, hãy so sánh với các đáp án và đưa ra câu trả lời cuối cùng. "
                    "Kết thúc bằng đúng một dòng: Đáp án: X (X là A, B, C hoặc D)."
                )
                messages.append(HumanMessage(content=feedback_msg))

            except Exception as e:
                error_msg = f"Error running code: {str(e)}"
                print_log(f"        [Error] {error_msg}")
                messages.append(HumanMessage(content=f"{error_msg}. Hãy kiểm tra logic và sửa lại code."))

            continue

        # Check if current step contains an explicit answer
        step_answer = extract_answer(content, max_choices=max_choices)
        if step_answer:
            print_log(f"        [Logic] Step {step+1}: Found explicit answer: {step_answer}")
            combined_raw = "\n---STEP---\n".join(step_texts)
            return {"answer": step_answer, "raw_response": combined_raw, "route": "math"}

        if step < max_steps - 1:
            print_log("        [Warning] No code or answer found. Reminding model...")
            messages.append(HumanMessage(content="Lưu ý: Bạn vẫn chưa đưa ra đáp án cuối cùng. Hãy kết thúc bằng: Đáp án: X"))

    # Max steps reached - build combined_raw and try to extract answer
    print_log("        [Warning] Max steps reached. Attempting answer extraction from combined text...")
    
    # Build combined_raw from all steps
    combined_raw = "\n---STEP---\n".join(step_texts) if step_texts else ""
    
    # Try fallback text reasoning with error handling
    try:
        fallback_result = _fallback_text_reasoning(llm, state["question"], choices_text, max_choices)
        fallback_text = fallback_result["text"]
        if fallback_text:
            combined_raw += "\n---FALLBACK---\n" + fallback_text
    except Exception as e:
        print_log(f"        [Error] Fallback reasoning failed: {e}")
        fallback_text = ""
    
    # Extract answer from the entire combined text (takes LAST explicit answer)
    final_answer = extract_answer(combined_raw, max_choices=max_choices)
    
    if final_answer:
        print_log(f"        [Logic] Extracted final answer from combined text: {final_answer}")
        return {"answer": final_answer, "raw_response": combined_raw, "route": "math"}
    
    # Still no answer - do one final strict LLM call with error handling
    print_log("        [Logic] No explicit answer found. Requesting strict final answer...")
    computed_str = "; ".join(computed_outputs) if computed_outputs else "Không có kết quả tính toán"
    try:
        strict_response = _request_final_answer(llm, state["question"], choices_text, computed_str)
        combined_raw += "\n---FINAL---\n" + strict_response
        
        final_answer = extract_answer(strict_response, max_choices=max_choices)
        if final_answer:
            print_log(f"        [Logic] Final strict answer: {final_answer}")
            return {"answer": final_answer, "raw_response": combined_raw, "route": "math"}
    except Exception as e:
        print_log(f"        [Error] Final answer request failed: {e}")
    
    # Absolute fallback - default to A
    print_log("        [Warning] All extraction attempts failed. Defaulting to A.")
    return {"answer": "A", "raw_response": combined_raw, "route": "math"}
