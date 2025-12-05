"""Direct Answer node for reading comprehension or general questions without RAG."""

from langchain_core.prompts import ChatPromptTemplate

from src.nodes.rag import extract_answer
from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.llm import get_large_model
from src.utils.logging import print_log

DIRECT_SYSTEM_PROMPT = """Bạn là chuyên gia đọc hiểu và phân tích.
Nhiệm vụ: Trả lời câu hỏi dựa trên thông tin được cung cấp trong đề bài (nếu có) hoặc kiến thức chung.

Lưu ý:
1. Nếu đề bài có đoạn văn, CHỈ dựa vào đoạn văn đó để suy luận.
2. Suy luận ngắn gọn, logic.
3. Kết thúc bằng: "Đáp án: X" (X là một trong các lựa chọn A, B, C, D, ...)."""

DIRECT_USER_PROMPT = """Câu hỏi: {question}
{choices}"""


def direct_answer_node(state: GraphState) -> dict:
    """Answer questions directly using Large Model (Skip Retrieval)."""
    print_log("        [Direct] Processing Reading Comprehension/General Question...")

    all_choices = get_choices_from_state(state)
    choices_text = format_choices(all_choices)
    
    llm = get_large_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", DIRECT_SYSTEM_PROMPT),
        ("human", DIRECT_USER_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "choices": choices_text,
    })

    content = response.content.strip()
    print_log(f"        [Direct] Reasoning: {content}...")

    answer = extract_answer(content, max_choices=len(all_choices) or 4)
    print_log(f"        [Direct] Final Answer: {answer}")
    return {"answer": answer}
