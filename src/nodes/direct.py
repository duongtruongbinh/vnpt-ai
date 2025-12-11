"""Direct Answer node for reading comprehension or general questions without RAG."""

from langchain_core.prompts import ChatPromptTemplate

from src.data_processing.answer import extract_answer
from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.llm import get_large_model
from src.utils.logging import print_log
from src.utils.prompts import load_prompt


def direct_answer_node(state: GraphState) -> dict:
    """Answer questions directly using Large Model (Skip Retrieval)."""
    print_log("        [Direct] Processing Reading Comprehension/General Question...")

    all_choices = get_choices_from_state(state)
    choices_text = format_choices(all_choices)
    
    llm = get_large_model()
    
    system_prompt = load_prompt("direct_answer.j2", "system")
    user_prompt = load_prompt("direct_answer.j2", "user", question=state["question"], choices=choices_text)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    chain = prompt | llm
    response = chain.invoke({})

    content = response.content.strip()
    print_log(f"        [Direct] Reasoning: {content}...")

    answer = extract_answer(content, max_choices=len(all_choices) or 4)
    print_log(f"        [Direct] Final Answer: {answer}")
    return {"answer": answer, "raw_response": content}
