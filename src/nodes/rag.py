"""RAG node for knowledge-based question answering."""

from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.ingestion import get_vector_store
from src.utils.llm import get_large_model
from src.utils.logging import print_log
from src.data_processing.answer import extract_answer


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
    return {"answer": answer, "context": context, "raw_response": content}

