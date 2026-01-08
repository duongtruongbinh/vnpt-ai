"""State schema definitions for the RAG pipeline graph."""

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """State schema for the RAG pipeline graph."""

    question_id: str
    question: str
    all_choices: list[str]
    route: str
    context: str
    answer: str
    raw_response: str

