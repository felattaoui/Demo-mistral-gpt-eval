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

    # Azure OpenAI (Chat Completions API)
    aoai_endpoint: str
    aoai_deployment: str

    # Azure Tenant (for Entra ID auth)
    azure_tenant_id: Optional[str] = None

    # Evaluation model (optional, for LLM-as-Judge)
    eval_deployment: Optional[str] = None

    # Azure AI Foundry Project (for cloud evaluation)
    project_endpoint: Optional[str] = None
    
    # Model endpoint and API key for cloud evaluation (optional)
    model_endpoint: Optional[str] = None
    model_api_key: Optional[str] = None
    
    # Azure subscription and resource group (for portal URL)
    azure_subscription_id: Optional[str] = None
    azure_resource_group: Optional[str] = None

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
            # Evaluation model (for LLM-as-Judge)
            eval_deployment=os.getenv("EVAL_MODEL_DEPLOYMENT"),
            # Azure AI Foundry Project endpoint for cloud evaluation
            project_endpoint=os.getenv("PROJECT_ENDPOINT"),
            # Model endpoint and API key for cloud evaluation
            model_endpoint=os.getenv("MODEL_ENDPOINT"),
            model_api_key=os.getenv("MODEL_API_KEY"),
            # Azure subscription and resource group (for portal URL)
            azure_subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            azure_resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
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
            "eval_configured": bool(self.eval_deployment and self.project_endpoint),
            "project_configured": bool(self.project_endpoint),
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
        print(f"\n{icon} Azure OpenAI (Chat Completions API)")
        if self.aoai_endpoint:
            print(f"   Endpoint: {self.aoai_endpoint[:50]}...")
        print(f"   Deployment: {self.aoai_deployment}")

        # Azure AI Foundry Project
        icon = "✅" if status["project_configured"] else "⚠️"
        status_text = "Configured" if status["project_configured"] else "Not set (cloud eval disabled)"
        print(f"\n{icon} Foundry Project: {status_text}")
        if self.project_endpoint:
            print(f"   Endpoint: {self.project_endpoint[:60]}...")

        # Evaluation
        icon = "✅" if status["eval_configured"] else "⚠️"
        status_text = "Configured" if status["eval_configured"] else "Not set (requires project + model)"
        print(f"\n{icon} Cloud Evaluation: {status_text}")
        if self.eval_deployment:
            print(f"   Judge Model: {self.eval_deployment}")
        
        print("=" * 50)
        
        return status
