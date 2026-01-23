"""
Confidence calculation using anchoring with normalization.

New architecture:
1. Extraction (GPT) → JSON
2. Normalization (if format known) → Transform OCR to match extracted format
3. Fuzzy Match (on normalized values) → Per-field score
4. LLM-as-Judge (optional) → Groundedness, Relevance, Fluency

Output: Per-field scores without weighted meta-score.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from field_formats import get_field_format, FORMAT_DATE, FORMAT_AMOUNT
from normalizers import normalize_value, normalize_amount, normalize_date


class ConfidenceCalculator:
    """
    Calculate extraction confidence by comparing OCR output with extracted values.

    Features:
    - Normalizes values before comparison (dates, amounts)
    - Per-field fuzzy matching
    - HITL flagging for low confidence fields
    - Separate format validation
    """

    def __init__(self, fuzzy_threshold: float = 0.7, hitl_threshold: float = 0.7):
        """
        Args:
            fuzzy_threshold: Minimum similarity ratio to consider a fuzzy match
            hitl_threshold: Fields below this threshold are flagged for human review
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.hitl_threshold = hitl_threshold

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison (lowercase, remove extra spaces, etc.)"""
        if text is None:
            return ""
        text = str(text).lower().strip()
        # Remove common OCR artifacts
        text = re.sub(r'\^?\{\}?\[?\]?', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not text1 and not text2:
            return 1.0  # Both empty = match
        if not text1 or not text2:
            return 0.0  # One empty = no match

        t1 = self.normalize_text(text1)
        t2 = self.normalize_text(text2)

        if t1 == t2:
            return 1.0

        # Check if one contains the other
        if t1 in t2 or t2 in t1:
            return 0.95

        # Fuzzy matching
        return SequenceMatcher(None, t1, t2).ratio()

    def find_in_ocr(self, value: str, ocr_text: str) -> Tuple[bool, float]:
        """
        Check if a value exists in OCR text using fuzzy matching.

        Returns:
            (found, confidence) tuple
        """
        if not value:
            return True, 1.0  # Empty value = nothing to find

        normalized_value = self.normalize_text(value)
        normalized_ocr = self.normalize_text(ocr_text)

        # Exact match
        if normalized_value in normalized_ocr:
            return True, 1.0

        # Fuzzy search - check each word/segment
        ocr_segments = normalized_ocr.split()
        value_segments = normalized_value.split()

        if not value_segments:
            return True, 1.0

        # Check if all value segments are found (possibly fuzzy)
        found_segments = 0
        total_similarity = 0.0

        for v_seg in value_segments:
            best_match = 0.0
            for o_seg in ocr_segments:
                sim = SequenceMatcher(None, v_seg, o_seg).ratio()
                best_match = max(best_match, sim)
            if best_match >= self.fuzzy_threshold:
                found_segments += 1
                total_similarity += best_match

        if found_segments == 0:
            return False, 0.0

        confidence = total_similarity / len(value_segments)
        return found_segments == len(value_segments), confidence

    def _find_ocr_raw_value(
        self,
        field_name: str,
        extracted_value: Any,
        ocr_text: str,
        format_type: Optional[str]
    ) -> Optional[str]:
        """
        Find the raw OCR value corresponding to an extracted value.

        For normalized fields (date, amount), search in OCR for values that
        match after normalization.
        """
        if not extracted_value or not format_type:
            return None

        extracted_str = str(extracted_value)

        if format_type == FORMAT_DATE:
            # Search for date patterns in OCR
            date_patterns = [
                r"\d{2}/\d{2}/\d{4}",  # DD/MM/YYYY
                r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
                r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY
                r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
                r"\d{2}\.\d{2}\.\d{4}",  # DD.MM.YYYY
            ]
            for pattern in date_patterns:
                for match in re.finditer(pattern, ocr_text):
                    normalized = normalize_date(match.group())
                    if normalized == extracted_str:
                        return match.group()

        elif format_type == FORMAT_AMOUNT:
            # Search for amount patterns in OCR
            # This is more complex - try to find the value directly
            extracted_normalized = normalize_amount(extracted_value)
            if extracted_normalized:
                # Search for numeric patterns
                amount_patterns = [
                    r"[\d\s.,]+(?:\s*[€$£¥])?",
                    r"[€$£¥]\s*[\d\s.,]+",
                ]
                for pattern in amount_patterns:
                    for match in re.finditer(pattern, ocr_text):
                        ocr_normalized = normalize_amount(match.group())
                        if ocr_normalized == extracted_normalized:
                            return match.group().strip()

        return None

    def compare_field(
        self,
        field_name: str,
        extracted_value: Any,
        ocr_text: str,
    ) -> Dict[str, Any]:
        """
        Compare a single extracted field against OCR text.

        Applies normalization based on field format before comparison.

        Returns:
            Field comparison result with:
            - ocr_raw: Original OCR value (if found)
            - ocr_normalized: Normalized OCR value
            - extracted: Extracted value
            - format_type: Detected format type
            - fuzzy_score: Similarity score (0-1)
            - format_valid: Whether format validation passed
            - flagged: Whether field needs HITL review
            - note: Explanation
        """
        # Determine format type
        format_type = get_field_format(field_name, extracted_value)

        # Handle null values
        if extracted_value is None:
            return {
                "field": field_name,
                "ocr_raw": None,
                "ocr_normalized": None,
                "extracted": None,
                "format_type": format_type,
                "fuzzy_score": 1.0,
                "format_valid": None,
                "flagged": False,
                "note": "Field is null"
            }

        extracted_str = str(extracted_value)

        # Try to find the raw OCR value
        ocr_raw = self._find_ocr_raw_value(field_name, extracted_value, ocr_text, format_type)

        # Normalize extracted value if format is known
        if format_type == FORMAT_DATE:
            extracted_normalized = normalize_date(extracted_str) or extracted_str
            format_valid = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", extracted_str))
        elif format_type == FORMAT_AMOUNT:
            extracted_normalized = normalize_amount(extracted_value) or extracted_str
            format_valid = isinstance(extracted_value, (int, float)) and extracted_value >= 0
        else:
            extracted_normalized = extracted_str
            format_valid = None  # No format rule for this field

        # Normalize OCR for comparison
        if format_type and ocr_raw:
            ocr_normalized = normalize_value(ocr_raw, format_type)
        else:
            ocr_normalized = None

        # Calculate fuzzy score
        if ocr_normalized and extracted_normalized:
            # Compare normalized values
            if ocr_normalized == extracted_normalized:
                fuzzy_score = 1.0
                note = "exact match after normalization"
            else:
                fuzzy_score = self.similarity(ocr_normalized, extracted_normalized)
                note = "fuzzy match after normalization" if fuzzy_score >= self.fuzzy_threshold else "mismatch after normalization"
        else:
            # No normalization - compare directly with OCR text
            found, fuzzy_score = self.find_in_ocr(extracted_str, ocr_text)
            if fuzzy_score == 1.0:
                note = "exact match"
            elif found:
                note = "fuzzy match"
            else:
                note = "not found in OCR"

        # HITL flagging
        flagged = fuzzy_score < self.hitl_threshold

        return {
            "field": field_name,
            "ocr_raw": ocr_raw,
            "ocr_normalized": ocr_normalized,
            "extracted": extracted_str,
            "format_type": format_type,
            "fuzzy_score": round(fuzzy_score, 3),
            "format_valid": format_valid,
            "flagged": flagged,
            "note": note
        }

    def calculate(
        self,
        extraction: Dict[str, Any],
        ocr_text: str,
    ) -> Dict[str, Any]:
        """
        Calculate confidence for all extracted fields.

        Args:
            extraction: Extracted data from GPT
            ocr_text: Raw OCR text from Mistral

        Returns:
            Confidence report with per-field details and summary
        """
        fields: Dict[str, Dict[str, Any]] = {}

        # Document number
        fields["document_number"] = self.compare_field(
            "document_number",
            extraction.get("document_number"),
            ocr_text
        )

        # Document date
        fields["document_date"] = self.compare_field(
            "document_date",
            extraction.get("document_date"),
            ocr_text
        )

        # Supplier name
        supplier = extraction.get("supplier") or {}
        fields["supplier_name"] = self.compare_field(
            "supplier_name",
            supplier.get("name") if isinstance(supplier, dict) else None,
            ocr_text
        )

        # Customer name
        customer = extraction.get("customer") or {}
        fields["customer_name"] = self.compare_field(
            "customer_name",
            customer.get("name") if isinstance(customer, dict) else None,
            ocr_text
        )

        # Total amount
        total = extraction.get("total_amount") or {}
        if isinstance(total, dict):
            amount_value = total.get("amount")
        else:
            amount_value = None
        fields["total_amount"] = self.compare_field(
            "total_amount",
            amount_value,
            ocr_text
        )

        # Line items (average confidence across items)
        line_items = extraction.get("line_items", [])
        if line_items:
            item_scores = []
            for i, item in enumerate(line_items):
                if isinstance(item, dict):
                    desc_result = self.compare_field(
                        f"line_item_{i}_description",
                        item.get("description"),
                        ocr_text
                    )
                    total_result = self.compare_field(
                        f"line_item_{i}_total",
                        item.get("total"),
                        ocr_text
                    )
                    item_scores.append((desc_result["fuzzy_score"] + total_result["fuzzy_score"]) / 2)

            if item_scores:
                avg_score = sum(item_scores) / len(item_scores)
                fields["line_items"] = {
                    "field": "line_items",
                    "ocr_raw": None,
                    "ocr_normalized": None,
                    "extracted": f"{len(line_items)} items",
                    "format_type": None,
                    "fuzzy_score": round(avg_score, 3),
                    "format_valid": None,
                    "flagged": avg_score < self.hitl_threshold,
                    "note": f"average across {len(line_items)} items"
                }

        # Build summary
        all_fields = list(fields.values())
        flagged_fields = [f["field"] for f in all_fields if f.get("flagged")]
        needs_review = len(flagged_fields) > 0

        # Count by confidence level
        high_conf = sum(1 for f in all_fields if f["fuzzy_score"] >= 0.9)
        medium_conf = sum(1 for f in all_fields if 0.5 <= f["fuzzy_score"] < 0.9)
        low_conf = sum(1 for f in all_fields if f["fuzzy_score"] < 0.5)

        # Count format validations
        format_checks = [f for f in all_fields if f["format_valid"] is not None]
        format_passed = sum(1 for f in format_checks if f["format_valid"])

        return {
            "fields": fields,
            "summary": {
                "total_fields": len(all_fields),
                "high_confidence": high_conf,
                "medium_confidence": medium_conf,
                "low_confidence": low_conf,
                "format_checks_passed": f"{format_passed}/{len(format_checks)}" if format_checks else "N/A",
                "flagged_fields": flagged_fields,
                "needs_review": needs_review,
                "hitl_threshold": self.hitl_threshold,
            }
        }

    # Keep legacy methods for backward compatibility
    def validate_format(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extraction format using regex rules.

        DEPRECATED: Use calculate() which includes format_valid per field.
        """
        checks = []

        # Date format validation (YYYY-MM-DD)
        date = extraction.get("document_date")
        if date is not None:
            is_valid = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(date)))
            checks.append({
                "field": "date_format",
                "value": date,
                "valid": is_valid,
                "expected": "YYYY-MM-DD",
            })

        # Amount validation (positive number)
        total = extraction.get("total_amount", {})
        if isinstance(total, dict):
            amount = total.get("amount")
            if amount is not None:
                is_valid = isinstance(amount, (int, float)) and amount >= 0
                checks.append({
                    "field": "amount_positive",
                    "value": amount,
                    "valid": is_valid,
                    "expected": ">= 0",
                })

        # Calculate score
        if not checks:
            score = 1.0
        else:
            valid_count = sum(1 for c in checks if c["valid"])
            score = valid_count / len(checks)

        return {
            "score": round(score, 3),
            "checks": checks,
            "passed": sum(1 for c in checks if c["valid"]),
            "total": len(checks),
        }

    def calculate_meta_score(
        self,
        anchoring_result: Dict[str, Any],
        format_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate weighted meta-score combining anchoring and format validation.

        DEPRECATED: The new calculate() method returns per-field scores without weighting.
        This method is kept for backward compatibility.
        """
        # Handle new format (fields dict) or old format (overall_confidence)
        if "fields" in anchoring_result:
            # New format - calculate average from fields
            fields = anchoring_result["fields"]
            scores = [f["fuzzy_score"] for f in fields.values()]
            anchoring_score = sum(scores) / len(scores) if scores else 1.0
            needs_review = anchoring_result.get("summary", {}).get("needs_review", False)
            flagged_fields = anchoring_result.get("summary", {}).get("flagged_fields", [])
        else:
            # Old format
            anchoring_score = anchoring_result.get("overall_confidence", 1.0)
            needs_review = anchoring_result.get("needs_review", False)
            flagged_fields = anchoring_result.get("flagged_fields", [])

        format_score = format_result.get("score", 1.0)

        # Weights
        weights = {
            "anchoring": 0.80,
            "format": 0.20,
        }

        # Calculate weighted score
        overall_score = (
            anchoring_score * weights["anchoring"]
            + format_score * weights["format"]
        )

        return {
            "overall_score": round(overall_score, 3),
            "method": "meta_score",
            "weights": weights,
            "needs_review": needs_review,
            "flagged_fields": flagged_fields,
            "components": {
                "anchoring": {
                    "score": anchoring_score,
                    "weight": weights["anchoring"],
                    "contribution": round(anchoring_score * weights["anchoring"], 3),
                },
                "format": {
                    "score": format_score,
                    "weight": weights["format"],
                    "contribution": round(format_score * weights["format"], 3),
                    "checks_passed": f"{format_result.get('passed', 0)}/{format_result.get('total', 0)}",
                },
            },
        }


def create_confidence_calculator(
    fuzzy_threshold: float = 0.7,
    hitl_threshold: float = 0.7,
) -> ConfidenceCalculator:
    """Factory function to create a confidence calculator.

    Args:
        fuzzy_threshold: Minimum similarity ratio (0-1) to consider a match.
                        Default 0.7 = 70% similar text is considered a match.
                        Lower = more permissive (tolerates OCR errors)
                        Higher = stricter (fewer false positives)
        hitl_threshold: Fields below this threshold are flagged for human review.
                       Default 0.7 = fields with < 70% confidence need review.
    """
    return ConfidenceCalculator(
        fuzzy_threshold=fuzzy_threshold,
        hitl_threshold=hitl_threshold,
    )
