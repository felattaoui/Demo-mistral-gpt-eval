"""
Normalization functions by format type.

Transforms OCR values to match the extracted format.
This allows comparing semantically equivalent values:
- "15/01/2024" vs "2024-01-15" → same date
- "1,234.56 €" vs "1234.56" → same amount
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


def normalize_date(ocr_value: str) -> Optional[str]:
    """
    Normalize an OCR date to ISO format (YYYY-MM-DD).

    Supports multiple input formats:
    - DD/MM/YYYY (European)
    - MM/DD/YYYY (American)
    - DD-MM-YYYY
    - YYYY/MM/DD
    - YYYY-MM-DD (already ISO)

    Args:
        ocr_value: OCR text containing a date

    Returns:
        Date in ISO format (YYYY-MM-DD) or None if no date found
    """
    if not ocr_value:
        return None

    ocr_str = str(ocr_value).strip()

    # Common date patterns with their strptime format
    # Note: European formats (DD/MM/YYYY) are tried first as they are more common in France
    patterns: List[Tuple[str, str]] = [
        # Formats with /
        (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"),  # DD/MM/YYYY (European)
        (r"(\d{4})/(\d{2})/(\d{2})", "%Y/%m/%d"),  # YYYY/MM/DD

        # Formats with -
        (r"(\d{2})-(\d{2})-(\d{4})", "%d-%m-%Y"),  # DD-MM-YYYY
        (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),  # YYYY-MM-DD (ISO)

        # Formats with .
        (r"(\d{2})\.(\d{2})\.(\d{4})", "%d.%m.%Y"),  # DD.MM.YYYY (German)
    ]

    for pattern, date_format in patterns:
        match = re.search(pattern, ocr_str)
        if match:
            try:
                dt = datetime.strptime(match.group(), date_format)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                # Invalid format (e.g., 31/02/2024), try next pattern
                continue

    return None  # No recognized format


def normalize_amount(ocr_value: Any) -> Optional[str]:
    """
    Normalize an OCR amount to standardized numeric format.

    Handles formats:
    - "1,234.56 €" → "1234.56" (US)
    - "1.234,56 €" → "1234.56" (European)
    - "1 234,56" → "1234.56" (French with space)
    - "$1,234.56" → "1234.56"

    Args:
        ocr_value: OCR text or number containing an amount

    Returns:
        Amount formatted with 2 decimals or None if invalid
    """
    if ocr_value is None:
        return None

    # If already a number
    if isinstance(ocr_value, (int, float)):
        return f"{float(ocr_value):.2f}"

    ocr_str = str(ocr_value).strip()
    if not ocr_str:
        return None

    # Remove currency symbols and spaces
    cleaned = re.sub(r'[$€£¥\s]', '', ocr_str)

    # Remove letters (e.g., "EUR", "USD")
    cleaned = re.sub(r'[A-Za-z]', '', cleaned)

    if not cleaned:
        return None

    # Handle European format (1.234,56) vs US (1,234.56)
    if ',' in cleaned and '.' in cleaned:
        # Both separators present
        if cleaned.rfind(',') > cleaned.rfind('.'):
            # European format: . is thousands separator, , is decimal
            # 1.234,56 → 1234.56
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            # US format: , is thousands separator, . is decimal
            # 1,234.56 → 1234.56
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        # Comma only
        # Check if it's a thousands separator or decimal
        parts = cleaned.split(',')
        if len(parts) == 2 and len(parts[1]) == 2:
            # Probably European decimal (e.g., 1234,56)
            cleaned = cleaned.replace(',', '.')
        elif len(parts) == 2 and len(parts[1]) == 3:
            # Probably thousands separator (e.g., 1,234)
            cleaned = cleaned.replace(',', '')
        else:
            # Default, treat as decimal
            cleaned = cleaned.replace(',', '.')
    # If only dots, it's either decimal or thousands
    # Keep as is since dot is the standard separator

    try:
        value = float(cleaned)
        return f"{value:.2f}"
    except ValueError:
        return None


def normalize_phone(ocr_value: str) -> Optional[str]:
    """
    Normalize a phone number (digits only).

    Examples:
    - "06 12 34 56 78" → "0612345678"
    - "+33 6 12 34 56 78" → "33612345678"
    - "(01) 234-5678" → "012345678"

    Args:
        ocr_value: OCR text containing a phone number

    Returns:
        Number with only digits or None if empty
    """
    if not ocr_value:
        return None

    # Keep only digits
    digits = re.sub(r'\D', '', str(ocr_value))

    return digits if digits else None


# Registry of normalizers by format type
NORMALIZERS: Dict[str, Callable[[Any], Optional[str]]] = {
    "date": normalize_date,
    "amount": normalize_amount,
    "phone": normalize_phone,
}


def normalize_value(value: Any, format_type: str) -> Optional[str]:
    """
    Normalize a value according to its format type.

    Args:
        value: Value to normalize
        format_type: Format type ("date", "amount", "phone")

    Returns:
        Normalized value or None if format not supported or value invalid
    """
    normalizer = NORMALIZERS.get(format_type)
    if normalizer:
        return normalizer(value)
    return None


def find_and_normalize_in_text(
    text: str,
    format_type: str,
    target_value: str
) -> Optional[str]:
    """
    Search for a value in OCR text and normalize it.

    Useful when you want to find the OCR value corresponding to an extraction.
    For example, if "2024-01-15" was extracted, search all dates in OCR
    and return the one that matches after normalization.

    Args:
        text: Complete OCR text
        format_type: Format type to search for
        target_value: Extracted value to match

    Returns:
        Normalized OCR value that matches or None
    """
    if format_type == "date":
        # Search for all potential dates in text
        date_patterns = [
            r"\d{2}/\d{2}/\d{4}",
            r"\d{4}/\d{2}/\d{2}",
            r"\d{2}-\d{2}-\d{4}",
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}\.\d{2}\.\d{4}",
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                normalized = normalize_date(match.group())
                if normalized == target_value:
                    return normalized

    elif format_type == "amount":
        # Search for all potential numeric values
        amount_patterns = [
            r"[\d\s,.]+(?:\s*[€$£¥])?",
            r"[€$£¥]\s*[\d\s,.]+",
        ]
        for pattern in amount_patterns:
            for match in re.finditer(pattern, text):
                normalized = normalize_amount(match.group())
                if normalized == target_value:
                    return normalized

    return None
