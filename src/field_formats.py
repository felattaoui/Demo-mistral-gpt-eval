"""
Field format configuration for normalization.

This module defines:
- Supported format types (date, amount, phone)
- Explicit per-field configuration
- Auto-detection fallback based on field name
"""

from __future__ import annotations

from typing import Any, Optional


# Supported formats
FORMAT_DATE = "date"
FORMAT_AMOUNT = "amount"
FORMAT_PHONE = "phone"

# Explicit configuration (takes priority)
# Mapping: field_name -> format_type
FIELD_FORMATS = {
    "document_date": FORMAT_DATE,
    "total_amount": FORMAT_AMOUNT,
    "subtotal": FORMAT_AMOUNT,
    "tax_amount": FORMAT_AMOUNT,
    # Add other fields as needed
}


def auto_detect_format(field_name: str, value: Any = None) -> Optional[str]:
    """
    Auto-detect format based on field name.

    Used as fallback if the field is not in FIELD_FORMATS.

    Args:
        field_name: Field name
        value: Extracted value (optional, for future value-based detection)

    Returns:
        Detected format type or None if no known format
    """
    field_lower = field_name.lower()

    # Detection by field name
    if "date" in field_lower:
        return FORMAT_DATE

    if any(kw in field_lower for kw in ["amount", "total", "price", "cost", "subtotal", "tax"]):
        return FORMAT_AMOUNT

    if any(kw in field_lower for kw in ["phone", "tel", "fax", "mobile"]):
        return FORMAT_PHONE

    return None  # No normalization


def get_field_format(field_name: str, value: Any = None) -> Optional[str]:
    """
    Get the format of a field (explicit config or auto-detection).

    Args:
        field_name: Field name
        value: Extracted value (optional)

    Returns:
        Format type or None if no known format
    """
    # Explicit config takes priority
    if field_name in FIELD_FORMATS:
        return FIELD_FORMATS[field_name]

    # Fallback to auto-detection
    return auto_detect_format(field_name, value)


def register_field_format(field_name: str, format_type: str) -> None:
    """
    Dynamically register a format for a field.

    Args:
        field_name: Field name
        format_type: Format type (FORMAT_DATE, FORMAT_AMOUNT, FORMAT_PHONE)
    """
    FIELD_FORMATS[field_name] = format_type
