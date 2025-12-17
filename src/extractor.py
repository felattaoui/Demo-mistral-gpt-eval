"""
Structured data extraction using Azure OpenAI Responses API.

Uses Entra ID authentication (same pattern as the original notebook).
"""

import json
from typing import Dict, Any, Optional

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


class StructuredExtractor:
    """
    Extract structured data using Azure OpenAI Responses API.
    
    Features:
    - Entra ID authentication with automatic token refresh
    - Strict JSON Schema validation (Structured Outputs)
    - Support for text and image inputs
    """
    
    DEFAULT_INSTRUCTIONS = """You are a document processing expert. Extract information 
from the provided document according to the specified schema. Be precise and extract 
only information that is explicitly present in the document. If a field cannot be 
found, use null or an appropriate default value."""

    HYBRID_INSTRUCTIONS = """You are a document processing expert specialized in accurate data extraction.

You are provided with:
1. The ORIGINAL IMAGE of the document
2. The OCR TEXT extracted by a separate OCR system (may contain errors)
3. A JSON SCHEMA defining the expected output structure

Your task:
- Use the ORIGINAL IMAGE as the source of truth for visual verification
- Use the OCR TEXT as a helpful reference (but be aware it may have OCR errors)
- Extract information according to the SCHEMA
- If the OCR text and image disagree, trust the image
- For visual elements (checkboxes, signatures, logos), rely on the image
- Be precise: only extract information explicitly visible in the document
- If a field cannot be found, use null

Output must strictly conform to the provided JSON schema."""
    
    def __init__(self, endpoint: str, deployment: str):
        """
        Initialize the extractor with Entra ID authentication.
        
        Args:
            endpoint: Azure OpenAI endpoint (e.g., https://xxx.cognitiveservices.azure.com)
            deployment: Model deployment name (e.g., gpt-5.1)
        """
        # Get token provider for Entra ID auth (same as original notebook)
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Build base URL for Responses API
        endpoint = endpoint.rstrip("/")
        base_url = f"{endpoint}/openai/v1/"
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=token_provider()  # Get the token
        )
        self.deployment = deployment
    
    def extract(
        self,
        text: str,
        schema: dict,
        schema_name: str = "extraction",
        instructions: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_mime_type: str = "image/png",
    ) -> Dict[str, Any]:
        """
        Extract structured data from text.
        
        Args:
            text: Source text to extract from (e.g., OCR output)
            schema: JSON schema defining the extraction structure
            schema_name: Name for the schema
            instructions: Custom system instructions (optional)
            image_base64: Optional base64-encoded image for multimodal extraction
            image_mime_type: MIME type of the image
            
        Returns:
            Extracted data conforming to the schema, with metadata
        """
        if instructions is None:
            instructions = self.DEFAULT_INSTRUCTIONS
        
        # Build input content
        input_content = []
        
        if image_base64:
            input_content.append({
                "type": "input_image",
                "image_url": f"data:{image_mime_type};base64,{image_base64}",
            })
        
        input_content.append({
            "type": "input_text",
            "text": f"Document content:\n\n---\n{text}\n---\n\nExtract the information according to the defined schema.",
        })
        
        # Call Responses API
        response = self.client.responses.create(
            model=self.deployment,
            instructions=instructions,
            input=[{"role": "user", "content": input_content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=0,
        )
        
        # Parse and enrich result
        result = json.loads(response.output_text)
        result["_metadata"] = {
            "model": self.deployment,
            "response_id": response.id,
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else None,
                "output_tokens": response.usage.output_tokens if response.usage else None,
            },
        }
        
        return result
    
    def extract_hybrid(
        self,
        ocr_text: str,
        image_base64: str,
        schema: dict,
        schema_name: str = "extraction",
        image_mime_type: str = "image/png",
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured data using both OCR text and original image.
        
        This hybrid approach allows the model to:
        - Verify OCR accuracy against the original image
        - Extract visual elements not captured by OCR
        - Cross-reference text and image for higher confidence
        
        Args:
            ocr_text: Text extracted by OCR (e.g., Mistral output)
            image_base64: Base64-encoded original document image
            schema: JSON schema defining the extraction structure
            schema_name: Name for the schema
            image_mime_type: MIME type of the image
            instructions: Custom system instructions (optional)
            
        Returns:
            Extracted data conforming to the schema, with metadata
        """
        if instructions is None:
            instructions = self.HYBRID_INSTRUCTIONS
        
        # Build input content with image FIRST, then OCR text
        input_content = [
            # 1. Original image for visual verification
            {
                "type": "input_image",
                "image_url": f"data:{image_mime_type};base64,{image_base64}",
            },
            # 2. OCR text as reference
            {
                "type": "input_text",
                "text": f"""## OCR Text (from Mistral Document AI)
The following text was extracted by OCR. Use it as reference but verify against the image:

---
{ocr_text}
---

## Task
Extract the information according to the defined schema. 
Use the image as the source of truth, and the OCR text as helpful context.""",
            },
        ]
        
        # Call Responses API
        response = self.client.responses.create(
            model=self.deployment,
            instructions=instructions,
            input=[{"role": "user", "content": input_content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=0,
        )
        
        # Parse and enrich result
        result = json.loads(response.output_text)
        result["_metadata"] = {
            "model": self.deployment,
            "response_id": response.id,
            "mode": "hybrid_ocr_vision",
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else None,
                "output_tokens": response.usage.output_tokens if response.usage else None,
            },
        }
        
        return result
    
    def extract_from_pdf(
        self,
        pdf_base64: str,
        schema: dict,
        schema_name: str = "extraction",
        filename: str = "document.pdf",
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured data directly from a PDF.
        
        Uses the Responses API's native PDF support (no OCR step needed).
        
        Args:
            pdf_base64: Base64-encoded PDF content
            schema: JSON schema for extraction
            schema_name: Name for the schema
            filename: Original filename for context
            instructions: Custom instructions (optional)
            
        Returns:
            Extracted data with metadata
        """
        if instructions is None:
            instructions = self.DEFAULT_INSTRUCTIONS
        
        # Build input with PDF file
        input_content = [
            {
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:application/pdf;base64,{pdf_base64}",
            },
            {
                "type": "input_text",
                "text": "Extract the information from this document according to the defined schema.",
            },
        ]
        
        response = self.client.responses.create(
            model=self.deployment,
            instructions=instructions,
            input=[{"role": "user", "content": input_content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=0,
        )
        
        result = json.loads(response.output_text)
        result["_metadata"] = {
            "model": self.deployment,
            "response_id": response.id,
            "mode": "direct_pdf",
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else None,
                "output_tokens": response.usage.output_tokens if response.usage else None,
            },
        }
        
        return result


def create_extractor(config) -> StructuredExtractor:
    """
    Create an extractor from configuration.
    
    Args:
        config: Config object with Azure OpenAI settings
        
    Returns:
        Configured StructuredExtractor instance
    """
    return StructuredExtractor(
        endpoint=config.aoai_endpoint,
        deployment=config.aoai_deployment,
    )
