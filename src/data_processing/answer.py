"""Answer extraction and validation utilities.

Consolidates answer-related logic:
- Extraction from LLM responses (CoT format)
- Validation against valid choices
- Normalization with fallback defaults
"""

import re
import string

from src.utils.logging import print_log


def extract_answer(response: str, max_choices: int = 4) -> str | None:
    """Extract answer letter from LLM response using strict explicit answer lines.
    
    Only accepts answers from explicit final-answer lines with colon:
    - "Đáp án: A", "Answer: B" (preferred)
    - "Lựa chọn: C" (secondary)
    
    Returns the LAST valid explicit answer line found (later lines override earlier).
    
    Args:
        response: Response text from LLM
        max_choices: Maximum number of valid choices
        
    Returns:
        Answer letter (A, B, C, D) or None if no explicit answer found
    """
    if not response:
        return None
    
    valid_labels = string.ascii_uppercase[:max_choices]
    
    # Pattern for primary labels: "Đáp án:" or "Answer:" (highest priority)
    primary_pattern = r"^[ \t]*\**(?:Đáp\s*án|Answer)[ \t]*[:：][ \t]*\**([A-Z])\b"
    
    # Pattern for secondary label: "Lựa chọn:" (lower priority)
    secondary_pattern = r"^[ \t]*\**Lựa\s*chọn[ \t]*[:：][ \t]*\**([A-Z])\b"
    
    # Find all matches for both patterns
    primary_matches = re.findall(primary_pattern, response, flags=re.IGNORECASE | re.MULTILINE)
    secondary_matches = re.findall(secondary_pattern, response, flags=re.IGNORECASE | re.MULTILINE)
    
    if primary_matches:
        answer = primary_matches[-1].upper()
        if answer in valid_labels:
            return answer
    
    if secondary_matches:
        answer = secondary_matches[-1].upper()
        if answer in valid_labels:
            return answer
    
    # Single letter response (entire response is just a letter)
    clean_response = response.strip()
    if len(clean_response) == 1 and clean_response.upper() in valid_labels:
        return clean_response.upper()
    
    return None


def validate_answer(answer: str, num_choices: int) -> tuple[bool, str]:
    """Validate if answer is within valid range and normalize it.
    
    Args:
        answer: Raw answer string from model
        num_choices: Number of choices available (A, B, C, D, ...)
        
    Returns:
        Tuple of (is_valid, normalized_answer)
    """
    valid_answers = string.ascii_uppercase[:num_choices]
    if answer and answer.upper() in valid_answers:
        return True, answer.upper()
    
    return False, answer or ""


def normalize_answer(
    answer: str | None,
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
        answer: Raw answer string from model (can be None)
        num_choices: Number of choices available
        question_id: Optional question ID for logging warnings
        default: Default answer if validation fails
        
    Returns:
        Normalized answer string
    """
    if answer is None:
        if question_id:
            print_log(
                f"        [Warning] No answer extracted for {question_id}, "
                f"defaulting to {default}"
            )
        return default
    
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
