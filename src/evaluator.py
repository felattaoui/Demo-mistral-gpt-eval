"""
Quality evaluation for extraction results using Azure AI Foundry Evals API.

Uses the AIProjectClient and builtin evaluators to create cloud evaluation
jobs visible in Azure AI Foundry portal.

Requires:
- Azure AI Foundry project endpoint (PROJECT_ENDPOINT)
- Model deployment for LLM judge (EVAL_MODEL_DEPLOYMENT)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from azure.identity import DefaultAzureCredential

# Try to import Azure AI Projects SDK
try:
    from azure.ai.projects import AIProjectClient
    from openai.types.evals.create_eval_jsonl_run_data_source_param import (
        CreateEvalJSONLRunDataSourceParam,
        SourceFileContent,
        SourceFileContentContent,
    )
    FOUNDRY_SDK_AVAILABLE = True
except ImportError:
    FOUNDRY_SDK_AVAILABLE = False

# Legacy: Keep for backward compatibility
EVALUATION_SDK_AVAILABLE = FOUNDRY_SDK_AVAILABLE

# Builtin evaluator configurations
BUILTIN_METRICS = {
    "coherence": {
        "evaluator_name": "builtin.coherence",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
    },
    "fluency": {
        "evaluator_name": "builtin.fluency",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
    },
    "relevance": {
        "evaluator_name": "builtin.relevance",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
    },
    "groundedness": {
        "evaluator_name": "builtin.groundedness",
        "data_mapping": {
            "context": "{{item.context}}",
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
    },
}


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

    def validate(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Run all validations."""
        results = {
            "validations": {},
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "validation_score": 0.0
        }

        # confidence_score removed - calculated by OCR/Extraction comparison
        validations = [
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
    """
    Evaluate extraction quality using Azure AI Foundry Evals API.

    Creates cloud evaluation jobs visible in Foundry portal.
    Uses builtin evaluators for: groundedness, relevance, coherence.
    """

    def __init__(
        self,
        project_endpoint: str,
        deployment: str = "gpt-4o",
        tenant_id: Optional[str] = None,
    ):
        """
        Initialize the evaluator with Azure AI Foundry project.

        Args:
            project_endpoint: Azure AI Foundry project endpoint (PROJECT_ENDPOINT)
            deployment: Eval model deployment name for LLM judge (e.g., gpt-4o)
            tenant_id: Azure tenant ID (optional, avoids corporate tenant conflicts)
        """
        self.project_endpoint = project_endpoint
        self.deployment = deployment

        # Use specific tenant to avoid Microsoft corporate tenant conflict
        if tenant_id:
            self.credential = DefaultAzureCredential(
                exclude_shared_token_cache_credential=True,
                additionally_allowed_tenants=[tenant_id],
                interactive_browser_tenant_id=tenant_id,
            )
        else:
            self.credential = DefaultAzureCredential()

        # Initialize clients
        self._project_client = None
        self._openai_client = None

        self.validator = ExtractionValidator()

    def _get_project_client(self):
        """Get or create the AI Project client."""
        if self._project_client is None and FOUNDRY_SDK_AVAILABLE and self.project_endpoint:
            self._project_client = AIProjectClient(
                endpoint=self.project_endpoint,
                credential=self.credential,
            )
        return self._project_client

    def _get_openai_client(self):
        """Get the OpenAI client from AI Project client."""
        if self._openai_client is None:
            project_client = self._get_project_client()
            if project_client:
                self._openai_client = project_client.get_openai_client()
        return self._openai_client

    def _build_testing_criteria(self, metrics: list = None) -> list:
        """Build testing criteria using builtin evaluators."""
        if metrics is None:
            metrics = ["groundedness", "relevance", "coherence"]

        criteria = []
        for metric in metrics:
            if metric not in BUILTIN_METRICS:
                continue

            config = BUILTIN_METRICS[metric]
            criteria.append({
                "type": "azure_ai_evaluator",
                "name": metric,
                "evaluator_name": config["evaluator_name"],
                "initialization_parameters": {
                    "deployment_name": self.deployment,
                },
                "data_mapping": config["data_mapping"],
            })

        return criteria

    def _build_data_source_config(self) -> Dict:
        """Build data source config schema."""
        return {
            "type": "custom",
            "item_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "object"}}
                        ]
                    },
                    "context": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "object"}}
                        ]
                    },
                    "response": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "object"}}
                        ]
                    },
                },
                "required": ["query", "response"],
            },
            "include_sample_schema": True,
        }

    def _parse_eval_results(self, output_items: list) -> Dict[str, Any]:
        """Parse evaluation results from Evals API output."""
        ai_metrics = {}

        for output_item in output_items:
            # Handle both dict and object types
            if hasattr(output_item, "model_dump"):
                output_data = output_item.model_dump()
            elif hasattr(output_item, "__dict__"):
                output_data = output_item.__dict__
            else:
                output_data = output_item if isinstance(output_item, dict) else {}

            # Parse per-evaluator results from results array
            results_array = output_data.get("results", [])
            if isinstance(results_array, list):
                for result_item in results_array:
                    if hasattr(result_item, "model_dump"):
                        result_dict = result_item.model_dump()
                    elif hasattr(result_item, "__dict__"):
                        result_dict = result_item.__dict__
                    else:
                        result_dict = result_item if isinstance(result_item, dict) else {}

                    # Extract metric name
                    metric_name = result_dict.get("name", "")

                    # Get sample output
                    sample = result_dict.get("sample", {})
                    result_output = sample.get("output", [])

                    # Parse the result output
                    if isinstance(result_output, list):
                        for item in result_output:
                            if isinstance(item, dict):
                                content = item.get("content")
                                if content and isinstance(content, str):
                                    try:
                                        parsed = json.loads(content)
                                        if isinstance(parsed, dict) and "score" in parsed:
                                            ai_metrics[metric_name] = parsed["score"]
                                    except json.JSONDecodeError:
                                        # Handle S-tag format: <S2>score</S2>
                                        score_match = re.search(r'<S2>(\d+)</S2>', content)
                                        if score_match:
                                            ai_metrics[metric_name] = int(score_match.group(1))

                    # Try direct score access
                    if metric_name not in ai_metrics:
                        score = result_dict.get("score")
                        if score is not None:
                            ai_metrics[metric_name] = score

        return ai_metrics

    def evaluate(
        self,
        source_text: str,
        extraction: Dict[str, Any],
        run_name: Optional[str] = None,
        log_to_cloud: bool = True,
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation via Azure AI Foundry Evals API.

        Creates a cloud evaluation job visible in Foundry portal.

        Args:
            source_text: Original OCR text from the document
            extraction: Extracted structured data
            run_name: Name for the evaluation run (optional)
            log_to_cloud: Whether to run cloud evaluation (default: True)

        Returns:
            Evaluation results with AI metrics, validation metrics, and overall score
        """
        results = {
            "ai_metrics": {},
            "validation_metrics": {},
            "cloud_evaluation": {},
            "overall_score": 0.0
        }

        if run_name is None:
            run_name = f"extraction-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Prepare data
        clean_extraction = {k: v for k, v in extraction.items() if not k.startswith("_")}
        response_text = json.dumps(clean_extraction, indent=2, ensure_ascii=False)
        query = "Extract structured data from the document."

        # Run cloud evaluation via Azure AI Foundry Evals API
        if FOUNDRY_SDK_AVAILABLE and log_to_cloud and self.project_endpoint:
            try:
                client = self._get_openai_client()
                if client is None:
                    raise ValueError("Could not initialize OpenAI client from AIProjectClient")

                print(f"   Creating cloud evaluation job...")
                print(f"   Project endpoint: {self.project_endpoint[:60]}...")
                print(f"   Deployment: {self.deployment}")

                # Build data source config
                data_source_config = self._build_data_source_config()

                # Create eval group with builtin evaluators
                testing_criteria = self._build_testing_criteria()
                eval_obj = client.evals.create(
                    name=run_name,
                    data_source_config=data_source_config,
                    testing_criteria=testing_criteria,
                )
                eval_id = eval_obj.id
                print(f"   Eval created: {eval_id}")

                # Prepare inline data
                eval_item = {
                    "query": query,
                    "context": source_text,
                    "response": response_text,
                }

                # Submit evaluation run with inline data
                eval_run = client.evals.runs.create(
                    eval_id=eval_id,
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    metadata={"source": "DocumentPipeline"},
                    data_source=CreateEvalJSONLRunDataSourceParam(
                        type="jsonl",
                        source=SourceFileContent(
                            type="file_content",
                            content=[SourceFileContentContent(item=eval_item)],
                        ),
                    ),
                )
                run_id = eval_run.id
                print(f"   Run submitted: {run_id}")

                # Poll for completion
                print(f"   Waiting for completion...")
                timeout = 120
                poll_interval = 5
                start_time = time.time()

                while True:
                    run = client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id)

                    if run.status in ("completed", "failed"):
                        break

                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        results["cloud_evaluation"] = {
                            "status": "Timeout",
                            "eval_id": eval_id,
                            "run_id": run_id,
                            "error": f"Evaluation timed out after {timeout} seconds",
                        }
                        break

                    time.sleep(poll_interval)

                if run.status == "completed":
                    # Get output items
                    output_items = list(client.evals.runs.output_items.list(
                        run_id=run_id,
                        eval_id=eval_id
                    ))

                    # Parse results
                    results["ai_metrics"] = self._parse_eval_results(output_items)

                    # Build portal URL
                    report_url = getattr(run, "report_url", None)
                    portal_url = report_url or "https://ai.azure.com/resource/evaluation"

                    results["cloud_evaluation"] = {
                        "status": "Completed",
                        "method": "azure_ai_foundry_evals_api",
                        "eval_id": eval_id,
                        "run_id": run_id,
                        "portal_url": portal_url,
                        "logged_to_cloud": True,
                    }
                    print(f"   Cloud evaluation complete")

                elif run.status == "failed":
                    # Get error details
                    error_info = getattr(run, "error", None)
                    error_msg = str(error_info) if error_info else "Evaluation job failed"
                    results["cloud_evaluation"] = {
                        "status": "Failed",
                        "eval_id": eval_id,
                        "run_id": run_id,
                        "error": error_msg,
                    }
                    print(f"   Cloud evaluation failed: {error_msg[:100]}")

            except Exception as e:
                error_msg = str(e)
                print(f"   Cloud evaluation error: {error_msg[:200]}")
                results["cloud_evaluation"] = {
                    "status": "Error",
                    "error": error_msg[:500],
                }
        else:
            if not FOUNDRY_SDK_AVAILABLE:
                results["cloud_evaluation"] = {
                    "status": "NotConfigured",
                    "error": "azure-ai-projects SDK not installed",
                }
            elif not self.project_endpoint:
                results["cloud_evaluation"] = {
                    "status": "NotConfigured",
                    "error": "PROJECT_ENDPOINT not configured",
                }
            else:
                results["cloud_evaluation"] = {
                    "status": "Skipped",
                    "note": "Cloud evaluation disabled",
                }

        # Custom validations (dates, amounts)
        results["validation_metrics"] = self.validator.validate(extraction)

        # Calculate overall score (weighted average)
        scores, weights = [], []

        for metric in ["groundedness", "relevance", "coherence"]:
            value = results["ai_metrics"].get(metric)
            if isinstance(value, (int, float)):
                score = value / 5  # Normalize to 0-1
                scores.append(score)
                weights.append(0.25)

        # Add validation score
        scores.append(results["validation_metrics"]["validation_score"])
        weights.append(0.15)

        if scores:
            results["overall_score"] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        return results


def create_evaluator(config: Any) -> Optional[QualityEvaluator]:
    """
    Create an evaluator from configuration.

    Args:
        config: Config object with Azure settings

    Returns:
        QualityEvaluator instance or None if not configured
    """
    project_endpoint = getattr(config, "project_endpoint", None)
    eval_deployment = getattr(config, "eval_deployment", None)

    if not project_endpoint or not eval_deployment:
        return None

    return QualityEvaluator(
        project_endpoint=project_endpoint,
        deployment=eval_deployment,
        tenant_id=getattr(config, "azure_tenant_id", None),
    )


# Aliases pour compatibilite avec le notebook
CloudEvaluator = QualityEvaluator
CloudEvaluatorV2 = QualityEvaluator
