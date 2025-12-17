"""Pydantic schemas used for Structured Outputs extraction."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field


class Money(BaseModel):
    amount: float = Field(description="Amount")
    currency: str = Field(default="USD", description="Currency code (e.g., EUR, USD)")


class Party(BaseModel):
    name: str = Field(description="Name")
    address: Optional[str] = Field(default=None, description="Address")
    tax_id: Optional[str] = Field(default=None, description="Tax identifier")


class LineItem(BaseModel):
    description: str = Field(description="Item description")
    quantity: Optional[float] = Field(default=None, description="Quantity")
    unit_price: Optional[float] = Field(default=None, description="Unit price")
    total: Optional[float] = Field(default=None, description="Line total")


class DocumentExtraction(BaseModel):
    """Generic schema for invoices/receipts/other business documents."""

    document_type: str = Field(description="Type of document (invoice, receipt, etc.)")
    document_number: Optional[str] = Field(default=None, description="Document identifier / number")
    document_date: Optional[str] = Field(default=None, description="Document date (YYYY-MM-DD if possible)")

    supplier: Optional[Party] = Field(default=None, description="Supplier / issuer")
    customer: Optional[Party] = Field(default=None, description="Customer / recipient")

    total_amount: Optional[Money] = Field(default=None, description="Total amount")
    line_items: List[LineItem] = Field(default_factory=list, description="Extracted line items")

    confidence_score: float = Field(description="Confidence score in [0,1] for the extraction")


def make_schema_strict(schema: dict) -> dict:
    """
    Transform a Pydantic JSON schema to be compatible with Azure OpenAI Structured Outputs.
    
    Azure OpenAI strict mode requires:
    1. additionalProperties: false on all objects
    2. All properties must be in the 'required' array
    """
    schema = schema.copy()
    
    def process_object(obj: dict) -> dict:
        """Recursively process an object schema."""
        if not isinstance(obj, dict):
            return obj
            
        obj = obj.copy()
        
        # If this is an object type, add additionalProperties: false and make all fields required
        if obj.get("type") == "object" and "properties" in obj:
            obj["additionalProperties"] = False
            # Make all properties required
            obj["required"] = list(obj["properties"].keys())
            # Process nested properties
            for prop_name, prop_value in obj["properties"].items():
                obj["properties"][prop_name] = process_object(prop_value)
        
        # Process $defs
        if "$defs" in obj:
            for def_name, def_value in obj["$defs"].items():
                obj["$defs"][def_name] = process_object(def_value)
        
        # Process items in arrays
        if "items" in obj:
            obj["items"] = process_object(obj["items"])
        
        # Process anyOf
        if "anyOf" in obj:
            obj["anyOf"] = [process_object(item) for item in obj["anyOf"]]
        
        # Process $ref (no change needed, but process siblings)
        if "$ref" in obj:
            pass  # $ref objects don't need processing
            
        return obj
    
    return process_object(schema)


def get_strict_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Generate a JSON schema compatible with Structured Outputs strict mode."""
    raw_schema = model_cls.model_json_schema(ref_template="#/$defs/{model}")
    return make_schema_strict(raw_schema)


# Default schema used by the pipeline
EXTRACTION_SCHEMA: Dict[str, Any] = get_strict_schema(DocumentExtraction)
