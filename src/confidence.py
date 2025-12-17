"""
Confidence calculation by comparing OCR text with LLM extraction.

No self-evaluation from GPT - confidence is computed by matching extracted values
against the original OCR text.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


class ConfidenceCalculator:
    """
    Calculate extraction confidence by comparing OCR output with extracted values.
    
    Strategy:
    - Perfect match = 1.0
    - Fuzzy match = 0.5-0.99 (based on similarity)
    - No match / not found in OCR = 0.0
    """
    
    def __init__(self, fuzzy_threshold: float = 0.8):
        """
        Args:
            fuzzy_threshold: Minimum similarity ratio to consider a fuzzy match
        """
        self.fuzzy_threshold = fuzzy_threshold
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for comparison (lowercase, remove extra spaces, etc.)"""
        if text is None:
            return ""
        text = str(text).lower().strip()
        # Remove common OCR artifacts
        text = re.sub(r'\^?\{\}?\[?\]?', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove currency symbols for amount comparison
        text = re.sub(r'[$€£¥]', '', text)
        # Remove thousand separators
        text = re.sub(r',(\d{3})', r'\1', text)
        return text
    
    @staticmethod
    def normalize_amount(value: Any) -> str:
        """Normalize monetary amounts for comparison."""
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            # Format with 2 decimals, no thousand separator
            return f"{float(value):.2f}"
        # Try to extract number from string
        text = str(value)
        match = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
        if match:
            try:
                return f"{float(match.group()):.2f}"
            except ValueError:
                pass
        return ConfidenceCalculator.normalize(text)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not text1 and not text2:
            return 1.0  # Both empty = match
        if not text1 or not text2:
            return 0.0  # One empty = no match
        
        t1 = self.normalize(text1)
        t2 = self.normalize(text2)
        
        if t1 == t2:
            return 1.0
        
        # Check if one contains the other
        if t1 in t2 or t2 in t1:
            return 0.95
        
        # Fuzzy matching
        return SequenceMatcher(None, t1, t2).ratio()
    
    def find_in_ocr(self, value: str, ocr_text: str) -> Tuple[bool, float]:
        """
        Check if a value exists in OCR text.
        
        Returns:
            (found, confidence) tuple
        """
        if not value:
            return True, 1.0  # Empty value = nothing to find
        
        normalized_value = self.normalize(value)
        normalized_ocr = self.normalize(ocr_text)
        
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
    
    def compare_field(
        self, 
        field_name: str, 
        extracted_value: Any, 
        ocr_text: str,
        is_amount: bool = False
    ) -> Dict[str, Any]:
        """
        Compare a single extracted field against OCR text.
        
        Returns:
            Field comparison result with confidence score
        """
        if extracted_value is None:
            return {
                "field": field_name,
                "extracted": None,
                "found_in_ocr": True,  # Nothing to find
                "confidence": 1.0,
                "note": "Field is null"
            }
        
        # Convert to string for comparison
        if is_amount:
            str_value = self.normalize_amount(extracted_value)
        else:
            str_value = str(extracted_value)
        
        found, confidence = self.find_in_ocr(str_value, ocr_text)
        
        return {
            "field": field_name,
            "extracted": str_value,
            "found_in_ocr": found,
            "confidence": round(confidence, 3),
            "note": "exact match" if confidence == 1.0 else (
                "fuzzy match" if found else "not found in OCR"
            )
        }
    
    def calculate(
        self, 
        extraction: Dict[str, Any], 
        ocr_text: str,
        field_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate overall confidence by comparing extraction with OCR.
        
        Args:
            extraction: Extracted data from GPT
            ocr_text: Raw OCR text from Mistral
            field_weights: Optional weights per field (default: equal weights)
            
        Returns:
            Confidence report with overall score and per-field details
        """
        # Default weights (critical fields weighted higher)
        if field_weights is None:
            field_weights = {
                "document_number": 1.5,
                "document_date": 1.2,
                "supplier_name": 1.0,
                "customer_name": 1.0,
                "total_amount": 2.0,  # Most critical
                "line_items": 1.0,
            }
        
        comparisons = []
        
        # Document number
        comparisons.append(self.compare_field(
            "document_number", 
            extraction.get("document_number"), 
            ocr_text
        ))
        
        # Document date
        comparisons.append(self.compare_field(
            "document_date", 
            extraction.get("document_date"), 
            ocr_text
        ))
        
        # Supplier name
        supplier = extraction.get("supplier") or {}
        comparisons.append(self.compare_field(
            "supplier_name", 
            supplier.get("name") if isinstance(supplier, dict) else None, 
            ocr_text
        ))
        
        # Customer name
        customer = extraction.get("customer") or {}
        comparisons.append(self.compare_field(
            "customer_name", 
            customer.get("name") if isinstance(customer, dict) else None, 
            ocr_text
        ))
        
        # Total amount
        total = extraction.get("total_amount") or {}
        comparisons.append(self.compare_field(
            "total_amount", 
            total.get("amount") if isinstance(total, dict) else None, 
            ocr_text,
            is_amount=True
        ))
        
        # Line items (average confidence across items)
        line_items = extraction.get("line_items", [])
        if line_items:
            item_confidences = []
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
                        ocr_text,
                        is_amount=True
                    )
                    item_confidences.append((desc_result["confidence"] + total_result["confidence"]) / 2)
            
            if item_confidences:
                avg_item_conf = sum(item_confidences) / len(item_confidences)
                comparisons.append({
                    "field": "line_items",
                    "extracted": f"{len(line_items)} items",
                    "found_in_ocr": avg_item_conf > self.fuzzy_threshold,
                    "confidence": round(avg_item_conf, 3),
                    "note": f"average across {len(line_items)} items"
                })
        
        # Calculate weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for comp in comparisons:
            weight = field_weights.get(comp["field"], 1.0)
            weighted_sum += comp["confidence"] * weight
            total_weight += weight
        
        overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            "overall_confidence": round(overall_confidence, 3),
            "method": "ocr_extraction_comparison",
            "fields": {c["field"]: c for c in comparisons},
            "summary": {
                "total_fields": len(comparisons),
                "high_confidence": sum(1 for c in comparisons if c["confidence"] >= 0.9),
                "medium_confidence": sum(1 for c in comparisons if 0.5 <= c["confidence"] < 0.9),
                "low_confidence": sum(1 for c in comparisons if c["confidence"] < 0.5),
            }
        }


def create_confidence_calculator(fuzzy_threshold: float = 0.8) -> ConfidenceCalculator:
    """Factory function to create a confidence calculator."""
    return ConfidenceCalculator(fuzzy_threshold=fuzzy_threshold)
