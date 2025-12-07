"""Answer extraction and validation utilities.

Consolidates answer-related logic:
- Extraction from LLM responses (CoT format)
- Validation against valid choices
- Normalization with fallback defaults
"""

import re
import string

from src.utils.logging import print_log


def extract_answer(response: str, max_choices: int = 26) -> str:
    """Extract answer letter from LLM response (supports CoT format).
    
    Handles multiple formats:
    - "Đáp án: A", "Answer: B", "Lựa chọn: C"
    - Single letter response
    - Letter anywhere in response (fallback)
    
    Args:
        response: Response text from LLM
        max_choices: Maximum number of valid choices (A-Z)
        
    Returns:
        Answer letter (A, B, C, ...) or "A" as default fallback
    """
    clean_response = response.strip()
    valid_labels = string.ascii_uppercase[:max_choices]

    # Pattern 1: Explicit answer format (highest priority)
    match = re.search(
        r"(?:Đáp án|Answer|Lựa chọn)[:\s]+([A-Z])",
        clean_response,
        re.IGNORECASE
    )
    if match:
        answer = match.group(1).upper()
        if answer in valid_labels:
            return answer

    # Pattern 2: Single letter response
    if clean_response.upper() in valid_labels:
        return clean_response.upper()
    
    # Pattern 3: Last valid letter in response (fallback)
    for char in reversed(clean_response):
        if char.upper() in valid_labels:
            return char.upper()
    
    return "A"


def validate_answer(answer: str, num_choices: int) -> tuple[bool, str]:
    """Validate if answer is within valid range and normalize it.
    
    Args:
        answer: Raw answer string from model
        num_choices: Number of choices available (A, B, C, D, ...)
        
    Returns:
        Tuple of (is_valid, normalized_answer)
    """
    valid_answers = string.ascii_uppercase[:num_choices]
    
    # Check if answer is a valid option label
    if answer.upper() in valid_answers:
        return True, answer.upper()
    
    return False, answer


def normalize_answer(
    answer: str,
    num_choices: int,
    question_id: str | None = None,
    default: str = "A",
) -> str:
    """Normalize and validate answer with fallback to default.
    
    Combines extraction, validation, and normalization:
    - Validates answer is within valid range (A, B, C, D, ...)
    - Normalizes refusal responses  
    - Falls back to default for invalid answers
    
    Args:
        answer: Raw answer string from model
        num_choices: Number of choices available
        question_id: Optional question ID for logging warnings
        default: Default answer if validation fails
        
    Returns:
        Normalized answer string
    """
    is_valid, normalized = validate_answer(answer, num_choices)
    
    if not is_valid:
        if question_id:
            print_log(
                f"        [Warning] Invalid answer '{answer}' for {question_id}, "
                f"defaulting to {default}"
            )
        return default
    
    return normalized


def extract_and_normalize(
    response: str,
    num_choices: int,
    question_id: str | None = None,
    default: str = "A",
) -> str:
    """Extract answer from response and normalize it (convenience function).
    
    Combines extract_answer() and normalize_answer() into a single call.
    
    Args:
        response: Raw LLM response text
        num_choices: Number of valid choices
        question_id: Optional question ID for logging
        default: Default answer if extraction/validation fails
        
    Returns:
        Normalized answer string
    """
    extracted = extract_answer(response, max_choices=num_choices)
    return normalize_answer(extracted, num_choices, question_id, default)
