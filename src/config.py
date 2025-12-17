"""
Configuration management for Azure Document Extraction Pipeline.

Loads settings from environment variables (.env file).
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration loaded from environment variables."""
    
    # Mistral Document AI
    mistral_endpoint: str
    mistral_api_key: str
    mistral_model: str
    
    # Azure OpenAI (Responses API)
    aoai_endpoint: str
    aoai_deployment: str
    
    # Azure Tenant (for Entra ID auth)
    azure_tenant_id: Optional[str] = None
    
    # Evaluation (optional)
    eval_deployment: Optional[str] = None
    
    # Extraction mode: "text_only", "hybrid", or "vision_only"
    extraction_mode: str = "hybrid"
    
    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """
        Load configuration from environment variables.
        
        Args:
            env_path: Optional path to .env file. If None, searches in current directory.
        """
        load_dotenv(env_path)
        
        return cls(
            mistral_endpoint=os.getenv("MISTRAL_ENDPOINT", ""),
            mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
            mistral_model=os.getenv("MISTRAL_MODEL", "mistral-document-ai-2505-2"),
            aoai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            aoai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1"),
            azure_tenant_id=os.getenv("AZURE_TENANT_ID"),
            # Optional: if not set, the pipeline won't initialize evaluation.
            eval_deployment=os.getenv("EVAL_MODEL_DEPLOYMENT"),
            # Extraction mode: text_only (OCR only), hybrid (OCR + image), vision_only (image only)
            extraction_mode=os.getenv("EXTRACTION_MODE", "hybrid"),
        )
    
    @property
    def aoai_base_url(self) -> str:
        """Base URL for Responses API."""
        endpoint = self.aoai_endpoint.rstrip("/")
        return f"{endpoint}/openai/v1/"
    
    def validate(self) -> dict:
        """Validate configuration and return status."""
        return {
            "mistral_configured": bool(self.mistral_endpoint and self.mistral_api_key),
            "aoai_configured": bool(self.aoai_endpoint),
            "eval_configured": bool(self.eval_deployment),
        }
    
    def show_status(self):
        """Print configuration status."""
        status = self.validate()
        
        print("=" * 50)
        print("Configuration Status")
        print("=" * 50)
        
        # Mistral
        icon = "✅" if status["mistral_configured"] else "❌"
        print(f"{icon} Mistral Document AI")
        if self.mistral_endpoint:
            print(f"   Endpoint: {self.mistral_endpoint[:50]}...")
        print(f"   Model: {self.mistral_model}")
        
        # Azure OpenAI
        icon = "✅" if status["aoai_configured"] else "❌"
        print(f"\n{icon} Azure OpenAI (Responses API)")
        if self.aoai_endpoint:
            print(f"   Endpoint: {self.aoai_endpoint[:50]}...")
        print(f"   Base URL: {self.aoai_base_url}")
        print(f"   Deployment: {self.aoai_deployment}")
        
        # Evaluation
        icon = "✅" if status["eval_configured"] else "⚠️"
        status_text = "Configured" if status["eval_configured"] else "Optional - not set"
        print(f"\n{icon} Evaluation: {status_text}")
        if self.eval_deployment:
            print(f"   Deployment: {self.eval_deployment}")
        
        print("=" * 50)
        
        return status
