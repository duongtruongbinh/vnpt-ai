"""Common utility functions used across the project."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text: clean whitespace, unicode, and formatting.
    
    Applies:
    - Unicode NFKC normalization
    - Zero-width character removal
    - Whitespace normalization
    - Multiple newline compression
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Unicode NFKC normalization (composing characters)
    text = unicodedata.normalize("NFKC", text)
    
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    
    # Normalize whitespace & newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"http[s]?://\S+", "", text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    
    return text.strip()


def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics from text.
    
    Example: 'văn hóa' -> 'van hoa'
    
    Args:
        text: Text with Vietnamese diacritics
        
    Returns:
        Text with diacritics removed, lowercased
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _extract_qid_number(qid: str) -> tuple[str, int]:
    """Extract prefix and numeric part from qid for natural sorting.
    
    Args:
        qid: Question ID like "test_0001" or "val_123"
        
    Returns:
        Tuple of (prefix, number) for sorting
    """
    match = re.match(r"^([a-zA-Z_]+)(\d+)$", qid)
    if match:
        return (match.group(1), int(match.group(2)))
    return (qid, 0)


def sort_qids(qids: list[str]) -> list[str]:
    """Sort question IDs naturally (test_0001 < test_0002 < test_0010).
    
    Args:
        qids: List of question IDs
        
    Returns:
        Sorted list of question IDs
    """
    return sorted(qids, key=_extract_qid_number)
