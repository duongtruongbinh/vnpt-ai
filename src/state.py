"""State schema definitions for the RAG pipeline graph."""

import string
from typing import TypedDict


class GraphState(TypedDict, total=False):
    """State schema for the RAG pipeline graph."""

    question_id: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    all_choices: list[str]  
    route: str
    context: str
    answer: str
    raw_response: str  # Full LLM output before answer extraction
    code_executed: str
    code_output: str


def get_choices_from_state(state: GraphState) -> list[str]:
    """Extract all choices from state, handling both new and legacy formats."""
    all_choices = state.get("all_choices", [])
    if all_choices:
        return all_choices
    # Fallback for legacy format
    choices = [
        state.get("option_a", ""),
        state.get("option_b", ""),
        state.get("option_c", ""),
        state.get("option_d", ""),
    ]
    return [c for c in choices if c]


def format_choices(choices: list[str]) -> str:
    """Format choices as labeled lines (A. ..., B. ..., etc.)."""
    lines = []
    for i, choice in enumerate(choices):
        if i < len(string.ascii_uppercase):
            lines.append(f"{string.ascii_uppercase[i]}. {choice}")
    return "\n".join(lines)
