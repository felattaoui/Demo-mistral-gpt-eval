"""
Document extraction pipeline.

Orchestrates OCR, extraction, and evaluation into a simple interface.

New architecture:
1. Extraction (GPT) → JSON
2. Normalization (if format known) → Transform OCR to match extracted format
3. Fuzzy Match (on normalized values) → Per-field score
4. LLM-as-Judge (optional) → Groundedness, Relevance, Fluency

Output: Per-field confidence scores without weighted meta-score.
"""

import json
from typing import Dict, Any, Optional, Type

from pydantic import BaseModel

from config import Config
from utils import get_file_info, encode_file_to_base64, is_pdf
from ocr import MistralOCR, create_ocr_client
from extractor import StructuredExtractor, create_extractor
from evaluator import create_evaluator
from confidence import ConfidenceCalculator, create_confidence_calculator
from schemas import DocumentExtraction, get_strict_schema, EXTRACTION_SCHEMA


class DocumentPipeline:
    """
    Complete document extraction pipeline.

    Combines:
    1. Mistral Document AI for OCR
    2. GPT (Chat Completions API) for structured extraction
    3. Confidence scoring with normalization (per-field fuzzy match + format validation)
    4. Azure AI Foundry Cloud Evaluation for quality assessment (optional)
    """

    def __init__(self, config: Config):
        """
        Initialize the pipeline.

        Args:
            config: Configuration object with all settings
        """
        self.config = config

        # Initialize components
        print("🔧 Initializing pipeline components...")

        self.ocr = create_ocr_client(config)
        print(f"   ✅ OCR client ready (model: {config.mistral_model})")

        self.extractor = create_extractor(config)
        print(f"   ✅ Extractor ready (model: {config.aoai_deployment})")

        if config.eval_deployment and config.project_endpoint:
            self.evaluator = create_evaluator(config)
            print(f"   ✅ Cloud Evaluator ready (model: {config.eval_deployment})")
            print(f"      Results visible at: ai.azure.com > Evaluation")
        else:
            self.evaluator = None
            print("   ⚠️  Cloud Evaluator not configured (requires project_endpoint + eval_deployment)")

        # Confidence calculator (with normalization)
        self.confidence_calculator = create_confidence_calculator(fuzzy_threshold=0.7)
        print("   ✅ Confidence calculator ready (with normalization for dates/amounts)")

        print("✅ Pipeline ready\n")

    def process(
        self,
        file_path: str,
        schema: Optional[dict] = None,
        schema_name: str = "document_extraction",
        use_direct_pdf: bool = False,
        custom_instructions: Optional[str] = None,
        run_evaluation: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.

        Args:
            file_path: Path to the document (PDF or image)
            schema: JSON schema for extraction (default: DocumentExtraction)
            schema_name: Name for the schema
            use_direct_pdf: Skip OCR for PDFs (not supported with Chat Completions)
            custom_instructions: Custom extraction instructions
            run_evaluation: Run quality evaluation (requires evaluator)
            verbose: Print progress messages

        Returns:
            Dictionary with file_info, ocr_result, extraction, confidence, and evaluation
        """
        if schema is None:
            schema = EXTRACTION_SCHEMA

        results = {
            "file_info": get_file_info(file_path),
            "ocr_result": None,
            "extraction": None,
            "confidence": None,  # Per-field scores
            "evaluation": None,
        }

        if verbose:
            print(f"📄 Processing: {results['file_info']['name']}")
            print(f"   Size: {results['file_info']['size_mb']} MB")

        file_is_pdf = is_pdf(file_path)

        # Direct PDF mode is not supported with Chat Completions API
        if use_direct_pdf and file_is_pdf:
            print("   ⚠️  Direct PDF mode not supported with Chat Completions API")
            print("   ➡️  Falling back to OCR + extraction mode")
            use_direct_pdf = False

        # Standard mode (OCR + extraction)
        if verbose:
            print("\n🔄 Step 1: OCR with Mistral Document AI")

        ocr_result = self.ocr.extract_from_file(file_path)
        source_text = self.ocr.get_markdown_text(ocr_result)

        results["ocr_result"] = {
            "pages_processed": self.ocr.get_page_count(ocr_result),
            "text_length": len(source_text),
            "text_preview": source_text[:500] + "..." if len(source_text) > 500 else source_text,
            "full_text": source_text,
        }

        if verbose:
            print(f"   ✅ OCR complete ({results['ocr_result']['pages_processed']} pages)")

        # Determine extraction mode
        extraction_mode = self.config.extraction_mode

        if extraction_mode == "hybrid":
            if verbose:
                print("\n🔄 Step 2: Hybrid extraction with GPT (OCR + Image)")

            # Encode the original file for vision
            file_base64, mime_type = encode_file_to_base64(file_path)

            extraction = self.extractor.extract_hybrid(
                ocr_text=source_text,
                image_base64=file_base64,
                schema=schema,
                schema_name=schema_name,
                image_mime_type=mime_type,
                instructions=custom_instructions,
            )
        else:
            # text_only mode
            if verbose:
                print("\n🔄 Step 2: Structured extraction with GPT (text only)")

            extraction = self.extractor.extract(
                text=source_text,
                schema=schema,
                schema_name=schema_name,
                instructions=custom_instructions,
            )

        results["extraction"] = extraction

        # Calculate confidence score (new architecture: per-field with normalization)
        if verbose:
            print("\n🔄 Step 3: Calculating confidence (with normalization)")

        # Calculate per-field confidence with normalization
        confidence_result = self.confidence_calculator.calculate(extraction, source_text)

        # Store confidence data (new format: per-field)
        results["confidence"] = confidence_result

        if verbose:
            mode_label = "hybrid" if extraction_mode == "hybrid" else "text"
            print(f"   ✅ Extraction complete (mode: {mode_label})")

            # Summary stats
            summary = confidence_result.get("summary", {})
            total = summary.get("total_fields", 0)
            high = summary.get("high_confidence", 0)
            medium = summary.get("medium_confidence", 0)
            low = summary.get("low_confidence", 0)

            print(f"   📊 Confidence by field:")
            print(f"      ✅ High (≥90%): {high}/{total}")
            print(f"      ⚠️  Medium (50-89%): {medium}/{total}")
            print(f"      ❌ Low (<50%): {low}/{total}")

            # Format validation
            format_checks = summary.get("format_checks_passed", "N/A")
            print(f"      📋 Format checks: {format_checks}")

            # HITL warning
            if summary.get("needs_review"):
                flagged = summary.get("flagged_fields", [])
                print(f"   ⚠️  HITL: {len(flagged)} field(s) need review: {', '.join(flagged)}")

        # Evaluation (optional - Cloud Evaluation via Azure AI Foundry)
        if run_evaluation and self.evaluator:
            if verbose:
                print("\n🔄 Step 4: Cloud Evaluation (Azure AI Foundry)")

            evaluation = self.evaluator.evaluate(
                source_text=source_text,
                extraction=extraction,
            )
            results["evaluation"] = evaluation

            if verbose:
                cloud_status = evaluation.get("cloud_evaluation", {}).get("status", "Unknown")
                overall = evaluation.get("overall_score", 0)
                print(f"   ✅ Evaluation complete (status: {cloud_status})")
                if overall > 0:
                    print(f"   📊 Overall score: {overall:.1%}")
                portal_url = evaluation.get("cloud_evaluation", {}).get("portal_url")
                if portal_url:
                    print(f"   🔗 Portal: {portal_url}")

        if verbose:
            print("\n✅ Processing complete")

        return results

    def process_with_schema(
        self,
        file_path: str,
        schema_model: Type[BaseModel],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a document using a Pydantic model as schema.

        Args:
            file_path: Path to the document
            schema_model: Pydantic model class
            **kwargs: Additional arguments for process()

        Returns:
            Processing results
        """
        schema = get_strict_schema(schema_model)
        schema_name = schema_model.__name__.lower()
        return self.process(file_path, schema=schema, schema_name=schema_name, **kwargs)

    @staticmethod
    def display_results(results: Dict[str, Any]):
        """Display results in a formatted way."""
        print("\n" + "=" * 60)
        print("📊 EXTRACTION RESULTS")
        print("=" * 60)

        extraction = results.get("extraction", {})

        # Basic info
        print(f"\nDocument Type: {extraction.get('document_type', 'N/A')}")
        print(f"Document Number: {extraction.get('document_number', 'N/A')}")
        print(f"Document Date: {extraction.get('document_date', 'N/A')}")

        # Amount
        if extraction.get("total_amount"):
            amt = extraction["total_amount"]
            print(f"Total Amount: {amt.get('amount', 'N/A')} {amt.get('currency', '')}")

        # Confidence score (new format: per-field)
        confidence = results.get("confidence", {})
        if confidence:
            summary = confidence.get("summary", {})
            fields = confidence.get("fields", {})

            print(f"\n📊 CONFIDENCE (Per-Field)")
            print("-" * 40)

            # Display each field
            for field_name, field_data in fields.items():
                score = field_data.get("fuzzy_score", 0)
                note = field_data.get("note", "")
                flagged = field_data.get("flagged", False)
                format_valid = field_data.get("format_valid")
                ocr_raw = field_data.get("ocr_raw")
                extracted = field_data.get("extracted")

                # Icon based on score and flagged status
                if flagged:
                    icon = "⚠️"
                elif score >= 0.9:
                    icon = "✅"
                elif score >= 0.5:
                    icon = "📝"
                else:
                    icon = "❌"

                # Format validation icon
                if format_valid is True:
                    fmt_icon = "✓"
                elif format_valid is False:
                    fmt_icon = "✗"
                else:
                    fmt_icon = "-"

                print(f"   {icon} {field_name}: {score:.0%} [{fmt_icon}]")
                print(f"      Extracted: {extracted}")
                if ocr_raw and ocr_raw != extracted:
                    print(f"      OCR raw: {ocr_raw}")
                print(f"      Note: {note}")

            # Summary
            print(f"\n   Summary:")
            print(f"      ✅ High confidence (≥90%): {summary.get('high_confidence', 0)}")
            print(f"      ⚠️  Medium confidence: {summary.get('medium_confidence', 0)}")
            print(f"      ❌ Low confidence: {summary.get('low_confidence', 0)}")
            print(f"      📋 Format checks: {summary.get('format_checks_passed', 'N/A')}")

            # HITL status
            if summary.get("needs_review"):
                flagged = summary.get("flagged_fields", [])
                print(f"\n   ⚠️  HITL: {len(flagged)} field(s) flagged for review:")
                for field in flagged:
                    print(f"      • {field}")
            else:
                print(f"\n   ✅ HITL: No fields flagged for review")

        # Line items
        items = extraction.get("line_items", [])
        if items:
            print(f"\nLine Items ({len(items)}):")
            for i, item in enumerate(items, 1):
                desc = item.get("description", "N/A")[:40]
                total = item.get("total", "N/A")
                print(f"   {i}. {desc}... → {total}")

        # Evaluation (Cloud Evaluation)
        if results.get("evaluation"):
            eval_data = results["evaluation"]

            print("\n" + "=" * 60)
            print("📈 EVALUATION RESULTS (Azure AI Foundry Cloud Evaluation)")
            print("=" * 60)

            # Cloud evaluation status
            cloud_eval = eval_data.get("cloud_evaluation", {})
            status = cloud_eval.get("status", "Unknown")
            print(f"\nStatus: {status}")

            if cloud_eval.get("error"):
                print(f"   ❌ Error: {cloud_eval['error']}")

            # AI metrics (if available)
            ai_metrics = eval_data.get("ai_metrics", {})
            if ai_metrics:
                print("\nAI Metrics:")
                for metric, data in ai_metrics.items():
                    if isinstance(data, dict) and metric in data:
                        score = data[metric]
                        icon = "✅" if isinstance(score, (int, float)) and score >= 4 else "⚠️"
                        print(f"   {icon} {metric.capitalize()}: {score}/5")

            # Validation
            val = eval_data.get("validation_metrics", {})
            if val.get("total_checks", 0) > 0:
                print(f"\nValidation: {val.get('passed_checks', 0)}/{val.get('total_checks', 0)} checks passed")

            # Overall score
            overall = eval_data.get("overall_score", 0)
            if overall > 0:
                icon = "🏆" if overall >= 0.9 else "✅" if overall >= 0.8 else "⚠️" if overall >= 0.7 else "❌"
                print(f"\n{icon} OVERALL SCORE: {overall:.1%}")

            # Portal URL (clickable link to view results in Foundry)
            portal_url = cloud_eval.get("portal_url")
            if portal_url:
                print(f"\n🔗 View in Foundry Portal:")
                print(f"   {portal_url}")

    @staticmethod
    def export_results(results: Dict[str, Any], output_path: str = "extraction_results.json"):
        """Export results to a JSON file."""
        # Create a copy to avoid modifying the original
        export_data = json.loads(json.dumps(results, default=str))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Results exported to: {output_path}")
