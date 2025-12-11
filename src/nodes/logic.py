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


def _indent_code(code: str) -> str:
    """Format code to make it easier to read in the terminal."""
    return "\n".join(f"        {line}" for line in code.splitlines())


def logic_solver_node(state: GraphState) -> dict:
    """Solve math/logic questions using Python code execution."""
    llm = get_large_model()
    all_choices = get_choices_from_state(state)
    choices_text = format_choices(all_choices)

    system_prompt = load_prompt("logic_solver.j2", "system")
    user_prompt = load_prompt("logic_solver.j2", "user", question=state["question"], choices=choices_text)

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    raw_responses: list[str] = []

    max_steps = 5
    for step in range(max_steps):
        response = llm.invoke(messages)
        content = response.content
        raw_responses.append(content)
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
                    combined_raw = "\n---STEP---\n".join(raw_responses)
                    return {"answer": code_ans, "raw_response": combined_raw}

                feedback_msg = f"Code output: {output}.\n"
                feedback_msg += "Lưu ý: Bạn vẫn chưa trả lời đáp án cuối cùng, duyệt lại code và các đáp án để chỉnh sửa phù hợp."
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
    combined_raw = "\n---STEP---\n".join(raw_responses) if raw_responses else ""
    return {"answer": "A", "raw_response": combined_raw}
