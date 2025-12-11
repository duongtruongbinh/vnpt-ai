"""Data loading utilities for test questions."""

import csv
import json
from pathlib import Path

from src.data_processing.models import QuestionInput

# Standard column mappings for choice columns
_CHOICE_COLUMN_MAPPINGS = {
    "choice_a": 0, "choice_b": 1, "choice_c": 2, "choice_d": 3,
    "option_a": 0, "option_b": 1, "option_c": 2, "option_d": 3,
    "a": 0, "b": 1, "c": 2, "d": 3,
}


def load_test_data_from_json(file_path: Path) -> list[QuestionInput]:
    """Load test questions from JSON file.

    Expected format: List of dicts with qid, question, choices, answer (optional)

    Args:
        file_path: Path to JSON file

    Returns:
        List of QuestionInput objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    if file_path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON files are supported: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of questions: {file_path}")

    questions = []
    for item in data:
        if "choices" not in item or not isinstance(item["choices"], list):
            raise ValueError(f"Question {item.get('qid', 'unknown')} must have 'choices' as a list")

        questions.append(QuestionInput(
            qid=item["qid"],
            question=item["question"],
            choices=item["choices"],
            answer=item.get("answer"),
        ))

    return questions


def _normalize_row_keys(row: dict[str, str]) -> dict[str, str]:
    """Normalize row keys to lowercase and strip whitespace."""
    return {k.lower().strip(): v for k, v in row.items()}


def _extract_choices_from_row(row: dict[str, str]) -> list[str]:
    """Extract choices from a normalized CSV row.

    Tries multiple strategies:
    1. Individual choice columns (choice_a/option_a/a, etc.)
    2. JSON array in 'choices' column
    3. Comma/semicolon separated string in 'choices' column

    Args:
        row: Normalized row dict with lowercase keys

    Returns:
        List of choice strings (may contain empty strings)
    """
    # Strategy 1: Individual columns (choice_a, option_a, a, etc.)
    choices = ["", "", "", ""]
    found_individual = False

    for col_name, idx in _CHOICE_COLUMN_MAPPINGS.items():
        if col_name in row and row[col_name]:
            choices[idx] = row[col_name].strip()
            found_individual = True

    if found_individual:
        return [c for c in choices if c]

    # Strategy 2 & 3: Parse 'choices' column
    choices_raw = row.get("choices", "")
    if not choices_raw:
        return []

    # Try JSON parse first
    try:
        parsed = json.loads(choices_raw)
        if isinstance(parsed, list):
            return [str(c).strip() for c in parsed if str(c).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: split by comma or semicolon
    return [c.strip() for c in choices_raw.replace(";", ",").split(",") if c.strip()]


def load_test_data_from_csv(file_path: Path) -> list[QuestionInput]:
    """Load test questions from CSV file.

    Supports multiple CSV formats:
    - Columns: qid, question, choice_a, choice_b, choice_c, choice_d
    - Columns: qid, question, option_a, option_b, option_c, option_d
    - Columns: qid, question, A, B, C, D
    - Columns: qid, question, choices (JSON array or comma-separated)

    Args:
        file_path: Path to CSV file

    Returns:
        List of QuestionInput objects

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    questions = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm_row = _normalize_row_keys(row)

            qid = norm_row.get("qid", "").strip()
            question = norm_row.get("question", "").strip()

            if not qid or not question:
                continue

            choices = _extract_choices_from_row(norm_row)
            if not choices:
                choices = ["", "", "", ""]

            questions.append(QuestionInput(
                qid=qid,
                question=question,
                choices=choices,
                answer=norm_row.get("answer", "").strip() or None,
            ))

    return questions
