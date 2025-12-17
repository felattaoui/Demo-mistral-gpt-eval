"""
Quality evaluation for extraction results using Azure AI Evaluation SDK.

Uses Entra ID authentication and GPT-4o as judge model.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from azure.identity import DefaultAzureCredential
from azure.ai.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
)

class ExtractionValidator:
    """Custom validators for extraction quality."""
    
    @staticmethod
    def validate_date(date_str: Optional[str]) -> Tuple[bool, str]:
        if not date_str:
            return True, "No date provided (optional)"
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return True, "Valid format"
        return False, f"Invalid format: {date_str}"
    
    @staticmethod
    def validate_amount(amount: Optional[float]) -> Tuple[bool, str]:
        if amount is None:
            return True, "No amount provided (optional)"
        if amount >= 0:
            return True, f"Valid amount: {amount}"
        return False, f"Negative amount: {amount}"
    
    @staticmethod
    def validate_confidence(score: Optional[float]) -> Tuple[bool, str]:
        if score is None:
            return False, "Missing confidence score"
        if 0 <= score <= 1:
            return True, f"Valid score: {score}"
        return False, f"Score out of range: {score}"
    
    def validate(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Run all validations."""
        results = {
            "validations": {},
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "validation_score": 0.0
        }
        
        validations = [
            ("confidence_score", self.validate_confidence, extraction.get("confidence_score")),
            ("document_date", self.validate_date, extraction.get("document_date")),
        ]
        
        # Check nested total_amount
        if extraction.get("total_amount"):
            validations.append(
                ("total_amount", self.validate_amount, extraction["total_amount"].get("amount"))
            )
        
        for name, validator, value in validations:
            is_valid, message = validator(value)
            results["validations"][name] = {"valid": is_valid, "message": message, "value": value}
            results["total_checks"] += 1
            if is_valid:
                results["passed_checks"] += 1
            else:
                results["failed_checks"] += 1
        
        if results["total_checks"] > 0:
            results["validation_score"] = results["passed_checks"] / results["total_checks"]
        
        return results


class QualityEvaluator:
    """Evaluate extraction quality using Azure AI Evaluation SDK."""
    
    def __init__(self, endpoint: str, deployment: str = "gpt-4o", tenant_id: Optional[str] = None):
        """
        Initialize the evaluator with Entra ID authentication.
        
        Args:
            endpoint: Azure OpenAI endpoint
            deployment: Eval model deployment (must be gpt-4o, SDK doesn't support gpt-5.x)
            tenant_id: Azure tenant ID (optional, avoids corporate tenant conflicts)
        """
        # Use specific tenant to avoid Microsoft corporate tenant conflict
        if tenant_id:
            credential = DefaultAzureCredential(
                exclude_shared_token_cache_credential=True,
                additionally_allowed_tenants=[tenant_id],
                interactive_browser_tenant_id=tenant_id,
            )
        else:
            credential = DefaultAzureCredential()
        
        self.model_config = {
            "azure_endpoint": endpoint,
            "azure_deployment": deployment,
            "api_version": "2024-12-01-preview"
        }
        
        # Initialize AI evaluators with Entra ID auth
        self.groundedness = GroundednessEvaluator(self.model_config, credential=credential)
        self.relevance = RelevanceEvaluator(self.model_config, credential=credential)
        self.coherence = CoherenceEvaluator(self.model_config, credential=credential)
        self.validator = ExtractionValidator()
    
    def evaluate(self, source_text: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            source_text: Original OCR text from the document
            extraction: Extracted structured data
            
        Returns:
            Evaluation results with AI metrics, validation metrics, and overall score
        """
        results = {
            "ai_metrics": {},
            "validation_metrics": {},
            "overall_score": 0.0
        }
        
        response_text = json.dumps(extraction, indent=2, ensure_ascii=False)
        query = "Extract structured data from the document."
        
        # AI metrics - using gpt-4o as judge
        # Groundedness: verifies extracted data is present in source document
        try:
            results["ai_metrics"]["groundedness"] = self.groundedness(
                query=query, context=source_text, response=response_text
            )
        except Exception as e:
            results["ai_metrics"]["groundedness"] = {"error": str(e)}
        
        # Relevance: verifies extraction answers the query
        try:
            results["ai_metrics"]["relevance"] = self.relevance(
                query=query, context=source_text, response=response_text
            )
        except Exception as e:
            results["ai_metrics"]["relevance"] = {"error": str(e)}
        
        # Coherence: verifies output is logically coherent
        try:
            results["ai_metrics"]["coherence"] = self.coherence(
                query=query, response=response_text
            )
        except Exception as e:
            results["ai_metrics"]["coherence"] = {"error": str(e)}
        
        # Custom validations (dates, amounts, confidence)
        results["validation_metrics"] = self.validator.validate(extraction)
        
        # Calculate overall score (weighted average)
        scores, weights = [], []
        
        for metric in ["groundedness", "relevance", "coherence"]:
            metric_result = results["ai_metrics"].get(metric, {})
            if "error" not in metric_result:
                score = metric_result.get(metric, 0) / 5  # Normalize to 0-1
                scores.append(score)
                weights.append(0.25)
        
        # Add validation score
        scores.append(results["validation_metrics"]["validation_score"])
        weights.append(0.15)
        
        # Add model's confidence score
        if extraction.get("confidence_score"):
            scores.append(extraction["confidence_score"])
            weights.append(0.1)
        
        if scores:
            results["overall_score"] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        return results


def create_evaluator(config: Any) -> Optional[QualityEvaluator]:
    """
    Create an evaluator from configuration.
    
    Args:
        config: Config object with Azure OpenAI settings
        
    Returns:
        QualityEvaluator instance or None if not configured
    """
    if not getattr(config, "eval_deployment", None):
        return None
    
    return QualityEvaluator(
        endpoint=config.aoai_endpoint,
        deployment=config.eval_deployment,
        tenant_id=getattr(config, "azure_tenant_id", None),
    )
