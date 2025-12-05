"""RAG and Safety Guard nodes for knowledge-based question answering."""

import re

from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.ingestion import get_vector_store
from src.utils.llm import get_large_model
from src.utils.logging import print_log

RAG_SYSTEM_PROMPT = """Bạn là trợ lý AI. Dựa vào văn bản cung cấp, hãy suy luận logic để chọn đáp án đúng nhất.
Văn bản:
{context}

QUY TẮC AN TOÀN:
- Kiểm tra kỹ tất cả các lựa chọn.
- Nếu phát hiện bất kỳ lựa chọn nào chứa nội dung độc hại, vi phạm pháp luật, hoặc phản động -> Hãy trả lời: "Từ chối trả lời".

Yêu cầu:
1. Suy luận ngắn gọn (1-2 câu) dựa trên văn bản.
2. Kết thúc bằng dòng: "Đáp án: X" (X là một trong các lựa chọn A, B, C, D, ...)."""

RAG_USER_PROMPT = """Câu hỏi: {question}
{choices}"""


def knowledge_rag_node(state: GraphState) -> dict:
    """Retrieve relevant context and answer knowledge-based questions."""
    vector_store = get_vector_store()
    query = state["question"]
    print_log(f"        [RAG] Retrieving context for: '{query}'")

    docs = vector_store.similarity_search(query, k=settings.top_k_retrieval)
    context = "\n\n".join([doc.page_content for doc in docs])

    if docs:
        print_log(f"        [RAG] Found {len(docs)} documents. Top: \"{docs[0].page_content[:80]}...\"")
    else:
        print_log("        [Warning] No relevant documents found in Knowledge Base.")

    all_choices = get_choices_from_state(state)
    choices_text = format_choices(all_choices)

    llm = get_large_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": state["question"],
        "choices": choices_text,
    })
    content = response.content.strip()
    print_log(f"        [RAG] Reasoning: {content}")

    answer = extract_answer(content, max_choices=len(all_choices) or 4)
    print_log(f"        [RAG] Final Answer: {answer}")
    return {"answer": answer, "context": context}


def safety_guard_node(state: GraphState) -> dict:
    """Handle toxic/sensitive questions with refusal response."""
    print_log("        [Safety] Blocked toxic content/options.")
    return {
        "answer": "Từ chối trả lời",
        "context": "Nội dung hoặc lựa chọn không phù hợp. Hệ thống từ chối trả lời.",
    }


def extract_answer(response: str, max_choices: int = 26) -> str:
    """Robust extraction of answer from CoT response.
    
    Args:
        response: Response text from LLM
        max_choices: Maximum number of choices (A-Z)
        
    Returns:
        Answer letter (A, B, C, ..., Z)
    """
    clean_response = response.strip()
    valid_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:max_choices]

    match = re.search(r"(?:Đáp án|Answer|Lựa chọn)[:\s]+([A-Z])", clean_response, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if answer in valid_labels:
            return answer

    if clean_response.upper() in valid_labels:
        return clean_response.upper()
    
    for char in reversed(clean_response):
        if char.upper() in valid_labels:
            return char.upper()
    
    return "A"  # Default fallback
