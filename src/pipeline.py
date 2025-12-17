"""
Document extraction pipeline.

Orchestrates OCR, extraction, and evaluation into a simple interface.
"""

import json
from typing import Dict, Any, Optional, Type

from pydantic import BaseModel

from config import Config
from utils import get_file_info, encode_file_to_base64, is_pdf
from ocr import MistralOCR, create_ocr_client
from extractor import StructuredExtractor, create_extractor
from evaluator import QualityEvaluator, create_evaluator
from schemas import DocumentExtraction, get_strict_schema, EXTRACTION_SCHEMA


class DocumentPipeline:
    """
    Complete document extraction pipeline.
    
    Combines:
    1. Mistral Document AI for OCR
    2. GPT-5.1 (Responses API) for structured extraction
    3. Azure AI Evaluation SDK for quality assessment
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
        
        if config.eval_deployment:
            self.evaluator = create_evaluator(config)
            print(f"   ✅ Evaluator ready (model: {config.eval_deployment})")
        else:
            self.evaluator = None
            print("   ⚠️  Evaluator not configured (optional)")
        
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
            use_direct_pdf: Use Responses API's native PDF support (skip OCR)
            custom_instructions: Custom extraction instructions
            run_evaluation: Run quality evaluation (requires evaluator)
            verbose: Print progress messages
            
        Returns:
            Dictionary with file_info, ocr_result, extraction, and evaluation
        """
        if schema is None:
            schema = EXTRACTION_SCHEMA
        
        results = {
            "file_info": get_file_info(file_path),
            "ocr_result": None,
            "extraction": None,
            "evaluation": None,
        }
        
        if verbose:
            print(f"📄 Processing: {results['file_info']['name']}")
            print(f"   Size: {results['file_info']['size_mb']} MB")
        
        file_is_pdf = is_pdf(file_path)
        
        # Direct PDF mode (skip OCR)
        if use_direct_pdf and file_is_pdf:
            if verbose:
                print("\n🔄 Mode: Direct PDF extraction (Responses API)")
            
            pdf_base64, _ = encode_file_to_base64(file_path)
            extraction = self.extractor.extract_from_pdf(
                pdf_base64=pdf_base64,
                schema=schema,
                schema_name=schema_name,
                filename=results["file_info"]["name"],
                instructions=custom_instructions,
            )
            results["extraction"] = extraction
            source_text = "[PDF processed directly by model]"
            
            if verbose:
                print("   ✅ Extraction complete")
        
        # Standard mode (OCR + extraction)
        else:
            if verbose:
                print("\n🔄 Step 1: OCR with Mistral Document AI")
            
            ocr_result = self.ocr.extract_from_file(file_path)
            source_text = self.ocr.get_markdown_text(ocr_result)
            
            results["ocr_result"] = {
                "pages_processed": self.ocr.get_page_count(ocr_result),
                "text_length": len(source_text),
                "text_preview": source_text[:500] + "..." if len(source_text) > 500 else source_text,
            }
            
            if verbose:
                print(f"   ✅ OCR complete ({results['ocr_result']['pages_processed']} pages)")
                print("\n🔄 Step 2: Structured extraction with GPT-5.1")
            
            extraction = self.extractor.extract(
                text=source_text,
                schema=schema,
                schema_name=schema_name,
                instructions=custom_instructions,
            )
            results["extraction"] = extraction
            
            if verbose:
                confidence = extraction.get("confidence_score", "N/A")
                print(f"   ✅ Extraction complete (confidence: {confidence})")
        
        # Evaluation (optional)
        if run_evaluation and self.evaluator:
            if verbose:
                print("\n🔄 Step 3: Quality evaluation")
            
            evaluation = self.evaluator.evaluate(
                source_text=source_text,
                extraction=extraction,
            )
            results["evaluation"] = evaluation
            
            if verbose:
                print(f"   ✅ Evaluation complete (score: {evaluation['overall_score']:.1%})")
        
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
        
        # Confidence
        print(f"Confidence: {extraction.get('confidence_score', 'N/A')}")
        
        # Line items
        items = extraction.get("line_items", [])
        if items:
            print(f"\nLine Items ({len(items)}):")
            for i, item in enumerate(items, 1):
                desc = item.get("description", "N/A")[:40]
                total = item.get("total", "N/A")
                print(f"   {i}. {desc}... → {total}")
        
        # Evaluation
        if results.get("evaluation"):
            eval_data = results["evaluation"]
            
            print("\n" + "=" * 60)
            print("📈 EVALUATION RESULTS")
            print("=" * 60)
            
            # AI metrics
            for metric, data in eval_data.get("ai_metrics", {}).items():
                if "error" not in data:
                    score = data.get(metric, "N/A")
                    icon = "✅" if isinstance(score, (int, float)) and score >= 4 else "⚠️"
                    print(f"   {icon} {metric.capitalize()}: {score}/5")
                else:
                    print(f"   ❌ {metric.capitalize()}: Error")
            
            # Validation
            val = eval_data.get("validation_metrics", {})
            print(f"\nValidation: {val.get('passed_checks', 0)}/{val.get('total_checks', 0)} checks passed")
            
            # Overall
            overall = eval_data.get("overall_score", 0)
            icon = "🏆" if overall >= 0.9 else "✅" if overall >= 0.8 else "⚠️" if overall >= 0.7 else "❌"
            print(f"\n{icon} OVERALL SCORE: {overall:.1%}")
    
    @staticmethod
    def export_results(results: Dict[str, Any], output_path: str = "extraction_results.json"):
        """Export results to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results exported to: {output_path}")