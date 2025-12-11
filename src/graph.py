"""LangGraph definition for the RAG pipeline."""

from langgraph.graph import END, StateGraph

from src.state import GraphState
from src.nodes.logic import logic_solver_node
from src.nodes.rag import knowledge_rag_node
from src.nodes.router import route_question, router_node
from src.nodes.direct import direct_answer_node


def build_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    
    workflow = StateGraph(GraphState)

    workflow.add_node("router", router_node)
    workflow.add_node("knowledge_rag", knowledge_rag_node)
    workflow.add_node("logic_solver", logic_solver_node)
    workflow.add_node("direct_answer", direct_answer_node)
    
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            "knowledge_rag": "knowledge_rag",
            "logic_solver": "logic_solver",
            "direct_answer": "direct_answer",
            "__end__": END,
        },
    )

    workflow.add_edge("knowledge_rag", END)
    workflow.add_edge("logic_solver", END)
    workflow.add_edge("direct_answer", END)
    return workflow.compile()

graph = None

def get_graph():
    """Get or create the compiled graph singleton."""
    global graph
    if graph is None:
        graph = build_graph()
    return graph