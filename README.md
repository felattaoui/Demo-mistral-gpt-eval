# Azure Document Extraction Pipeline

Pipeline for extracting structured data from documents (PDFs, images) using Azure AI services.

## Features

- **OCR** with Mistral Document AI
- **Structured extraction** with GPT-5.1 (Chat Completions API + Structured Outputs)
- **Cloud evaluation** with Azure AI Foundry Evals API (builtin evaluators)
- **Confidence scoring** with OCR vs Extraction anchoring + format validation
- **Entra ID authentication** with automatic token refresh
- **Customizable schemas** via Pydantic

## Structure

```
Demo-mistral-gpt-eval/
├── src/
│   ├── config.py          # Configuration (.env)
│   ├── utils.py           # Utilities (base64, file info)
│   ├── ocr.py             # Mistral OCR client
│   ├── extractor.py       # GPT extraction (Chat Completions API)
│   ├── evaluator.py       # Cloud evaluation (Azure AI Foundry)
│   ├── confidence.py      # Confidence score (anchoring + format)
│   ├── schemas.py         # Pydantic schemas
│   ├── field_formats.py   # Field format specifications
│   ├── normalizers.py     # Extracted value normalization
│   └── pipeline.py        # Complete pipeline
│
├── azure_document_extraction_pipeline.ipynb  # Main notebook
├── output/                # Exported results
│
├── .env                   # Configuration (not versioned)
├── requirements.txt       # Python dependencies
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone the project
git clone https://github.com/felattaoui/Demo-mistral-gpt-eval.git
cd Demo-mistral-gpt-eval

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file at the project root:

```env
# ----------------------------------------------
# Mistral Document AI (OCR)
# ----------------------------------------------
MISTRAL_ENDPOINT=https://your-resource.services.ai.azure.com
MISTRAL_API_KEY=your-api-key
MISTRAL_MODEL=mistral-document-ai-2505-2

# ----------------------------------------------
# Azure OpenAI (Extraction)
# ----------------------------------------------
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-5.1

# ----------------------------------------------
# Azure AI Foundry (Cloud Evaluation)
# ----------------------------------------------
PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/project-name
EVAL_MODEL_DEPLOYMENT=gpt-4.1

# ----------------------------------------------
# Azure Tenant (for Entra ID)
# ----------------------------------------------
AZURE_TENANT_ID=your-tenant-id

# ----------------------------------------------
# Extraction mode: text_only or hybrid
# ----------------------------------------------
EXTRACTION_MODE=hybrid
```

### 3. Azure Authentication

```bash
# Log in with Azure CLI
az login --tenant your-tenant-id
```

### 4. Usage

```python
import sys
sys.path.insert(0, "src")

from config import Config
from pipeline import DocumentPipeline

# Load configuration
config = Config.from_env()

# Create the pipeline
pipeline = DocumentPipeline(config)

# Process a document
results = pipeline.process("invoice.png", run_evaluation=True, verbose=True)

# Display results
pipeline.display_results(results)

# Export
pipeline.export_results(results, "output/results.json")
```

## Azure Prerequisites

### Required Deployments

| Service | Model | Usage |
|---------|--------|-------|
| Azure AI Services | Mistral Document AI | Document OCR |
| Azure OpenAI | GPT-5.1 | Structured extraction |
| Azure AI Foundry | gpt-4.1 | Cloud evaluation (LLM-as-Judge) |

### Authentication

- **Azure OpenAI**: Entra ID (DefaultAzureCredential)
- **Mistral OCR**: API Key (Bearer token)
- **Evaluation**: Entra ID with specific tenant

## Extraction Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `text_only` | OCR text only → GPT | Documents with clear text |
| `hybrid` | OCR text + image → GPT | Documents with visual elements (checkboxes, signatures) |

## Evaluation Metrics

### Cloud Evaluation (Azure AI Foundry)

| Metric | Description | Scale |
|--------|-------------|-------|
| Groundedness | Data present in OCR source | 1-5 |
| Relevance | Extraction relevance | 1-5 |
| Coherence | JSON result coherence | 1-5 |

### Local Confidence (Anchoring)

| Metric | Description | Scale |
|--------|-------------|-------|
| Anchoring | OCR vs Extraction match | 0-1 |
| Format Validation | Format compliance (dates, amounts) | pass/fail |
| HITL Flag | Fields requiring human review | boolean |

## Custom Schemas

Create your own extraction schemas with Pydantic:

```python
from pydantic import BaseModel, Field
from typing import Optional

class ContractExtraction(BaseModel):
    """Contract extraction schema."""

    contract_number: str = Field(description="Contract number")
    parties: list[str] = Field(description="Contract parties")
    effective_date: Optional[str] = Field(default=None, description="Effective date (YYYY-MM-DD)")
    termination_date: Optional[str] = Field(default=None, description="Termination date (YYYY-MM-DD)")
    total_value: Optional[float] = Field(default=None, description="Total value")

# Use the schema
results = pipeline.process_with_schema(
    file_path="contract.pdf",
    schema_model=ContractExtraction
)
```

## Azure Documentation

- [Azure OpenAI Chat Completions](https://learn.microsoft.com/azure/ai-services/openai/how-to/chatgpt)
- [Structured Outputs](https://learn.microsoft.com/azure/ai-services/openai/how-to/structured-outputs)
- [Mistral OCR on Azure](https://learn.microsoft.com/azure/ai-foundry/how-to/use-image-models)
- [Azure AI Foundry Evals API](https://learn.microsoft.com/azure/ai-foundry/how-to/evaluate-sdk)

## License

MIT
