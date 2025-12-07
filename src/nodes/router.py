"""Router node for classifying questions and directing to appropriate handlers."""

import string
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

from src.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.llm import get_small_model
from src.utils.logging import print_log


def _find_refusal_option(state: GraphState) -> str | None:
    """Find refusal option in choices and return corresponding letter."""
    all_choices = get_choices_from_state(state)
    option_labels = list(string.ascii_uppercase[:len(all_choices)])
    
    refusal_patterns = [
        "tôi không thể", "không thể trả lời", "không thể cung cấp", "không thể chia sẻ",
        "từ chối trả lời", "từ chối cung cấp",
        "nằm ngoài phạm vi", "không thuộc phạm vi",
        "tôi là ai", "tôi là mô hình ngôn ngữ", 
        "hành vi vi phạm", "trái pháp luật", "không hỗ trợ",
    ]
    
    for i, choice in enumerate(all_choices):
        txt = choice.lower().strip()
        if any(p in txt for p in refusal_patterns):
            return option_labels[i]
            
    return None


def _classify_with_llm(state: GraphState) -> str:
    """Classify question using LLM."""
    choices_text = format_choices(get_choices_from_state(state))
    llm = get_small_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", ROUTER_USER_PROMPT),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "choices": choices_text,
    })
    return response.content.strip().lower()


def router_node(state: GraphState) -> dict:
    """Analyze question and determine routing path. Returns answer immediately for toxic content."""
    question = state["question"].lower()
    
    # Fast-track: Direct answer for reading comprehension
    direct_keywords = ["đoạn thông tin", "đoạn văn", "bài đọc", "căn cứ vào đoạn", "theo đoạn"]
    if any(k in question for k in direct_keywords) and len(question.split()) > 50:
        print_log("        [Router] Fast-track: Direct Answer (Found Context block)")
        return {"route": "direct"}
    
    # Fast-track: Math/Logic for LaTeX or math keywords
    math_signals = [
        "$", "\\frac", "^",
        "tính giá trị", "biểu thức", "phương trình", "hàm số", "đạo hàm",
        "xác suất", "lãi suất", "vận tốc", "gia tốc", "điện trở",
        "bao nhiêu gam", "mol", "nguyên tử khối",
    ]
    if any(s in question for s in math_signals):
        print_log("        [Router] Fast-track: Math (Keywords/LaTeX detected)")
        return {"route": "math"}
    
    # Slow-track: Use LLM for classification
    print_log("        [Router] Slow-track: Using LLM to classify...")
    try:
        route = _classify_with_llm(state)
        print_log(f"        [Router] LLM Decision: {route}")
        
        if "direct" in route:
            route_type = "direct"
        elif "math" in route or "logic" in route:
            route_type = "math"
        elif "toxic" in route or "danger" in route or "harmful" in route:
            # Check for refusal option and return answer immediately
            refusal_answer = _find_refusal_option(state)
            if refusal_answer:
                print_log(f"        [Router] Toxic detected, found refusal option: {refusal_answer}")
                return {"route": "toxic", "answer": refusal_answer}
            print_log("        [Router] Toxic detected, no refusal option found, defaulting to A")
            return {"route": "toxic", "answer": "A"}
        else:
            route_type = "rag"
        
        return {"route": route_type}
    except Exception as e:
        print_log(f"        [Router] Error: {e}. Fallback to RAG.")
        return {"route": "rag"}


def route_question(state: GraphState) -> Literal["knowledge_rag", "logic_solver", "direct_answer", "__end__"]:
    """Conditional edge function to route to appropriate node based on state route."""
    route = state.get("route", "rag")
    answer = state.get("answer")
    
    # Toxic questions are always resolved in router_node, go directly to END
    if route == "toxic":
        print_log(f"        [Router] Toxic resolved with answer: {answer}")
        return "__end__"
    
    if route == "direct":
        return "direct_answer"
    if route == "math":
        return "logic_solver"
    return "knowledge_rag"
