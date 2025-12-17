"""
Mistral Document AI OCR client for Azure.

Extracts text from documents (PDF, images) using Mistral's OCR model.
"""

import requests
from typing import Dict, Any, Optional

from utils import encode_file_to_base64


class MistralOCR:
    """
    Client for Mistral Document AI on Azure.
    
    Supports PDF and image files. Returns markdown-formatted text.
    """
    
    def __init__(self, endpoint: str, api_key: str, model: str = "mistral-document-ai-2505-2"):
        """
        Initialize the OCR client.
        
        Args:
            endpoint: Azure AI Services endpoint (e.g., https://xxx.services.ai.azure.com)
            api_key: API key for authentication
            model: Model name (default: mistral-document-ai-2505-2)
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
    
    def extract_text(
        self,
        document_base64: str,
        mime_type: str,
        include_images: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text from a base64-encoded document.
        
        Args:
            document_base64: Base64-encoded document content
            mime_type: MIME type of the document
            include_images: Whether to include extracted images in response
            
        Returns:
            OCR result with pages and markdown text
        """
        url = f"{self.endpoint}/providers/mistral/azure/ocr"
        
        # Build payload based on document type
        if "pdf" in mime_type:
            document_url = f"data:{mime_type};base64,{document_base64}"
            payload = {
                "model": self.model,
                "document": {"type": "document_url", "document_url": document_url},
                "include_image_base64": include_images,
            }
        else:
            image_url = f"data:{mime_type};base64,{document_base64}"
            payload = {
                "model": self.model,
                "document": {"type": "image_url", "image_url": image_url},
                "include_image_base64": include_images,
            }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        return response.json()
    
    def extract_from_file(self, file_path: str, include_images: bool = False) -> Dict[str, Any]:
        """
        Extract text from a file.
        
        Args:
            file_path: Path to the document file
            include_images: Whether to include extracted images
            
        Returns:
            OCR result with pages and markdown text
        """
        base64_content, mime_type = encode_file_to_base64(file_path)
        return self.extract_text(base64_content, mime_type, include_images)
    
    def get_markdown_text(self, ocr_result: Dict[str, Any]) -> str:
        """
        Extract combined markdown text from OCR result.
        
        Args:
            ocr_result: Result from extract_text or extract_from_file
            
        Returns:
            Combined markdown text from all pages
        """
        pages = ocr_result.get("pages", [])
        return "\n\n".join(page.get("markdown", "") for page in pages)
    
    def get_page_count(self, ocr_result: Dict[str, Any]) -> int:
        """Get number of pages in OCR result."""
        return len(ocr_result.get("pages", []))


def create_ocr_client(config) -> MistralOCR:
    """
    Create an OCR client from configuration.
    
    Args:
        config: Config object with Mistral settings
        
    Returns:
        Configured MistralOCR instance
    """
    return MistralOCR(
        endpoint=config.mistral_endpoint,
        api_key=config.mistral_api_key,
        model=config.mistral_model,
    )