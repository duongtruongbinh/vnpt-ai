"""Router node for classifying questions and directing to appropriate handlers."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

from src.state import GraphState, format_choices, get_choices_from_state
from src.utils.llm import get_small_model
from src.utils.logging import print_log

ROUTER_SYSTEM_PROMPT = """Nhiệm vụ: Phân loại câu hỏi vào 1 trong 4 nhóm chính xác tuyệt đối.

QUAN TRỌNG: Bạn phải kiểm tra kỹ nội dung của CÂU HỎI và tất cả các LỰA CHỌN.

1. "toxic":
   - Nếu BẤT KỲ lựa chọn nào chứa nội dung vi phạm pháp luật, kích động bạo lực, phản động, hoặc trái thuần phong mỹ tục.
   - Nội dung cờ bạc, ma túy, mại dâm, chế tạo vũ khí, tự tử.
   - Ngôn từ thù ghét, xúc phạm danh nhân/lãnh đạo.
   -> Ưu tiên xếp vào nhóm này ngay lập tức nếu phát hiện.

2. "direct": 
   - Câu hỏi chứa đoạn văn bản, đoạn thông tin dài.
   - Yêu cầu đọc hiểu, trích xuất thông tin từ đoạn văn đó.

3. "math":
   - Các bài toán đố, tính toán số học, hình học, xác suất.
   - Các câu hỏi cần lập luận, logic, tìm quy luật.
   
4. "rag": 
   - Kiến thức Lịch sử, Địa lý, Văn hóa, Xã hội, Văn học, Luật pháp ...
   - Những câu hỏi cần tra cứu kiến thức (không có trong đề).

Chỉ trả về đúng 1 từ: toxic, math, direct, hoặc rag."""

ROUTER_USER_PROMPT = """Câu hỏi: {question}
{choices}

Nhóm:"""


def router_node(state: GraphState) -> dict:
    """Analyze question and determine routing path."""
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

    route = response.content.strip().lower()

    if "math" in route or "logic" in route:
        route_type = "math"
    elif "toxic" in route or "danger" in route or "harmful" in route:
        route_type = "toxic"
    else:
        route_type = "knowledge"

    return {"route": route_type}


def route_question(state: GraphState) -> Literal["knowledge_rag", "logic_solver", "safety_guard", "direct_answer"]:
    """Conditional edge function to route to appropriate node."""
    question = state["question"].lower()

    # Fast-track: Direct answer for reading comprehension
    direct_keywords = ["đoạn thông tin", "đoạn văn", "bài đọc", "căn cứ vào đoạn", "theo đoạn"]
    if any(k in question for k in direct_keywords) and len(question.split()) > 50:
        print_log("        [Router] Fast-track: Direct Answer (Found Context block)")
        return "direct_answer"

    # Fast-track: Math/Logic for LaTeX or math keywords
    math_signals = [
        "$", "\\frac", "^",
        "tính giá trị", "biểu thức", "phương trình", "hàm số", "đạo hàm",
        "xác suất", "lãi suất", "vận tốc", "gia tốc", "điện trở",
        "bao nhiêu gam", "mol", "nguyên tử khối",
    ]
    if any(s in question for s in math_signals):
        print_log("        [Router] Fast-track: Math (Keywords/LaTeX detected)")
        return "logic_solver"

    # Slow-track: Use LLM for classification
    print_log("        [Router] Slow-track: Using LLM to classify...")
    try:
        choices_text = format_choices(get_choices_from_state(state))
        llm = get_small_model()
        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_USER_PROMPT),
        ])

        chain = prompt | llm
        response = chain.invoke({"question": state["question"], "choices": choices_text})
        route = response.content.strip().lower()
        print_log(f"        [Router] LLM Decision: {route}")

        if "direct" in route:
            return "direct_answer"
        if "math" in route or "logic" in route:
            return "logic_solver"
        if "toxic" in route or "danger" in route:
            return "safety_guard"
        return "knowledge_rag"

    except Exception as e:
        print_log(f"        [Router] Error: {e}. Fallback to RAG.")
        return "knowledge_rag"
