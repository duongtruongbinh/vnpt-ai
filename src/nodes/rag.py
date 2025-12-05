"""RAG and Safety Guard nodes for knowledge-based question answering."""

import re

from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.graph import GraphState
from src.utils.ingestion import get_vector_store
from src.utils.llm import get_large_model

RAG_SYSTEM_PROMPT = """Bạn là trợ lý AI. Dựa vào văn bản cung cấp, hãy suy luận logic để chọn đáp án đúng nhất.
Văn bản:
{context}

QUY TẮC AN TOÀN:
- Kiểm tra kỹ 4 lựa chọn A, B, C, D.
- Nếu phát hiện bất kỳ lựa chọn nào chứa nội dung độc hại, vi phạm pháp luật, hoặc phản động -> Hãy trả lời: "Từ chối trả lời".

Yêu cầu:
1. Suy luận ngắn gọn (1-2 câu) dựa trên văn bản.
2. Kết thúc bằng dòng: "Đáp án: X" (X là A, B, C, hoặc D)."""

RAG_USER_PROMPT = """Câu hỏi: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}"""

def knowledge_rag_node(state: GraphState) -> dict:
    """Retrieve relevant context and answer knowledge-based questions."""
    
    vector_store = get_vector_store()

    query = state["question"]
    print(f"        [RAG] Retrieving context for: '{query}'")
    
    docs = vector_store.similarity_search(query, k=settings.top_k_retrieval)
    context = "\n\n".join([doc.page_content for doc in docs])

    if docs:
        print(f"        [RAG] Found {len(docs)} documents. Top match: \"{docs[0].page_content[:100]}...\"")
    else:
        print("        [RAG] Warning: No relevant documents found in Knowledge Base.")

    llm = get_large_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": state["question"],
        "option_a": state["option_a"],
        "option_b": state["option_b"],
        "option_c": state["option_c"],
        "option_d": state["option_d"],
    })
    content = response.content.strip()
    print(f"        [RAG] Reasoning: {content[:200]}...")
    
    answer = extract_answer(content)
    print(f"        [RAG] Final Answer: {answer}")
    return {"answer": answer, "context": context}

def safety_guard_node(state: GraphState) -> dict:
    """Handle toxic/sensitive questions with refusal response."""
    print("        [Safety] Blocked toxic content/options.")
    return {
        "answer": "Từ chối trả lời", 
        "context": "Nội dung hoặc lựa chọn không phù hợp. Hệ thống từ chối trả lời.",
    }

def extract_answer(response: str) -> str:
    """Robust extraction of answer from CoT response."""
    clean_response = response.strip()
    match = re.search(r"(?:Đáp án|Answer|Lựa chọn)[:\s]+([ABCD])", clean_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if clean_response.upper() in ["A", "B", "C", "D"]:
        return clean_response.upper()
    for char in reversed(clean_response):
        if char.upper() in "ABCD":
            return char.upper()
    return "A"