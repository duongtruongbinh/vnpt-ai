"""Data processing utilities for the RAG pipeline."""

from src.data_processing.answer import (
    extract_answer,
    extract_and_normalize,
    normalize_answer,
    validate_answer,
)
from src.data_processing.formatting import (
    choices_to_options,
    format_choices_display,
    question_to_state,
)
from src.data_processing.loaders import (
    load_test_data_from_csv,
    load_test_data_from_json,
)
from src.data_processing.models import InferenceLogEntry, PredictionOutput, QuestionInput

__all__ = [
    # Models
    "QuestionInput",
    "PredictionOutput",
    "InferenceLogEntry",
    # Loaders
    "load_test_data_from_json",
    "load_test_data_from_csv",
    # Formatting
    "choices_to_options",
    "question_to_state",
    "format_choices_display",
    # Answer processing
    "extract_answer",
    "validate_answer",
    "normalize_answer",
    "extract_and_normalize",
]
