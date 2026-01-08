import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_processing.models import QuestionInput
    from src.state import GraphState


def question_to_state(q: "QuestionInput") -> "GraphState":
    """Convert QuestionInput to GraphState for pipeline processing."""
    state: "GraphState" = {
        "question_id": q.qid,
        "question": q.question,
        "all_choices": q.choices,
    }
    return state


def format_choices(choices: list[str]) -> str:
    """Format choices for LLM prompts (A. ..., B. ..., etc.)."""
    return "\n".join(f"{label}. {text}" for label, text in zip(string.ascii_uppercase, choices))


def format_choices_display(choices: list[str]) -> str:
    """Format choices for console display (2 columns)."""
    labels = string.ascii_uppercase
    lines = []
    for i in range(0, len(choices), 2):
        parts = []
        for j in range(2):
            idx = i + j
            if idx < len(choices):
                label = labels[idx] if idx < len(labels) else str(idx)
                parts.append(f"{label}. {choices[idx]:<30}")
        if parts:
            lines.append("   " + " ".join(parts))
    return "\n".join(lines)