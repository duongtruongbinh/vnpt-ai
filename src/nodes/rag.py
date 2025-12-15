"""RAG node for knowledge-based question answering with Retrieve & Rerank."""

import re

from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.data_processing.answer import extract_answer
from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.ingestion import get_vector_store
from src.utils.llm import get_large_model, get_small_model
from src.utils.logging import print_log
from src.utils.prompts import load_prompt


def _rerank_documents(query: str, docs: list, top_k: int = 3) -> list:
    """Rerank retrieved documents using the small LLM.
    
    Args:
        query: The user question
        docs: List of retrieved documents
        top_k: Number of top documents to return after reranking
    
    Returns:
        List of reranked documents (top_k most relevant)
    """
    if len(docs) <= top_k:
        return docs
    
    llm = get_small_model()
    
    # Build document list for reranking prompt
    doc_list = ""
    for i, doc in enumerate(docs):
        content_preview = doc.page_content[:350].replace("\n", " ")
        doc_list += f"[{i}] {content_preview}...\n\n"
    
    rerank_system = (
        "Bạn là chuyên gia đánh giá độ liên quan của văn bản. "
        "Nhiệm vụ: Chọn ra các đoạn văn bản LIÊN QUAN NHẤT với câu hỏi.\n"
        "Chỉ trả về danh sách các số ID (ví dụ: 0, 3, 5), không giải thích."
    )
    
    rerank_user = (
        f"Câu hỏi: {query}\n\n"
        f"Các đoạn văn bản:\n{doc_list}\n"
        f"Hãy chọn {top_k} đoạn văn bản LIÊN QUAN NHẤT với câu hỏi. "
        f"Trả về danh sách ID (số từ 0 đến {len(docs)-1}), cách nhau bởi dấu phẩy."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", rerank_system),
        ("human", rerank_user),
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({})
        content = response.content.strip()
        print_log(f"        [RAG] Reranker response: {content}")
        
        # Parse selected IDs from response
        selected_ids = []
        numbers = re.findall(r'\d+', content)
        for num_str in numbers:
            idx = int(num_str)
            if 0 <= idx < len(docs) and idx not in selected_ids:
                selected_ids.append(idx)
                if len(selected_ids) >= top_k:
                    break
        
        if selected_ids:
            reranked = [docs[i] for i in selected_ids]
            print_log(f"        [RAG] Reranked: selected {len(reranked)} docs from {len(docs)}")
            return reranked
        
        print_log("        [RAG] Rerank parsing failed, using first top_k docs")
        return docs[:top_k]
        
    except Exception as e:
        print_log(f"        [RAG] Reranking failed: {e}. Using keyword boosting fallback.")
        return docs[:top_k]



def knowledge_rag_node(state: GraphState) -> dict:
    """Retrieve relevant context, rerank, and answer knowledge-based questions."""
    vector_store = get_vector_store()
    query = state["question"]
    print_log(f"        [RAG] Retrieving context for: '{query}'")

    docs = vector_store.similarity_search(query, k=settings.top_k_retrieval)
    print_log(f"        [RAG] Retrieved {len(docs)} documents")

    if not docs:
        print_log("        [Warning] No relevant documents found in Knowledge Base.")
        context = ""
    else:
        reranked_docs = _rerank_documents(query, docs, top_k=settings.top_k_rerank)
        
        context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
        
        if reranked_docs:
            print_log(f"        [RAG] Using {len(reranked_docs)} reranked docs. Top: \"{reranked_docs[0].page_content[:80]}...\"")

    all_choices = get_choices_from_state(state)
    choices_text = format_choices(all_choices)

    llm = get_large_model()
    
    system_prompt = load_prompt("rag.j2", "system", context=context)
    user_prompt = load_prompt("rag.j2", "user", question=state["question"], choices=choices_text)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    chain = prompt | llm
    response = chain.invoke({})
    content = response.content.strip()
    print_log(f"        [RAG] Reasoning: {content}")

    answer = extract_answer(content, max_choices=len(all_choices) or 4)
    print_log(f"        [RAG] Final Answer: {answer}")
    return {"answer": answer, "context": context, "raw_response": content}
