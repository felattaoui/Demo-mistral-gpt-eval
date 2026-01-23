# Azure Document Extraction Pipeline

Pipeline d'extraction de données structurées à partir de documents (PDF, images) utilisant les services Azure AI.

## Fonctionnalités

- **OCR** avec Mistral Document AI
- **Extraction structurée** avec GPT-5.1 (Chat Completions API + Structured Outputs)
- **Évaluation cloud** avec Azure AI Foundry Evals API (builtin evaluators)
- **Confidence scoring** avec anchoring OCR vs Extraction + validation de format
- **Authentification Entra ID** avec refresh automatique du token
- **Schémas personnalisables** via Pydantic

## Structure

```
Demo-mistral-gpt-eval/
├── src/
│   ├── config.py          # Configuration (.env)
│   ├── utils.py           # Utilitaires (base64, file info)
│   ├── ocr.py             # Client Mistral OCR
│   ├── extractor.py       # Extraction GPT (Chat Completions API)
│   ├── evaluator.py       # Évaluation cloud (Azure AI Foundry)
│   ├── confidence.py      # Score de confiance (anchoring + format)
│   ├── schemas.py         # Schémas Pydantic
│   ├── field_formats.py   # Spécifications de format par champ
│   ├── normalizers.py     # Normalisation des valeurs extraites
│   └── pipeline.py        # Pipeline complet
│
├── azure_document_extraction_pipeline.ipynb  # Notebook principal
├── output/                # Résultats exportés
│
├── .env                   # Configuration (non versionné)
├── requirements.txt       # Dépendances Python
└── README.md
```

## Démarrage Rapide

### 1. Installation

```bash
# Cloner le projet
git clone https://github.com/felattaoui/Demo-mistral-gpt-eval.git
cd Demo-mistral-gpt-eval

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration

Créer un fichier `.env` à la racine du projet :

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
# Azure AI Foundry (Évaluation Cloud)
# ----------------------------------------------
PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/project-name
EVAL_MODEL_DEPLOYMENT=gpt-4.1

# ----------------------------------------------
# Azure Tenant (pour Entra ID)
# ----------------------------------------------
AZURE_TENANT_ID=your-tenant-id

# ----------------------------------------------
# Mode d'extraction: text_only ou hybrid
# ----------------------------------------------
EXTRACTION_MODE=hybrid
```

### 3. Authentification Azure

```bash
# Se connecter avec Azure CLI
az login --tenant your-tenant-id
```

### 4. Utilisation

```python
import sys
sys.path.insert(0, "src")

from config import Config
from pipeline import DocumentPipeline

# Charger la config
config = Config.from_env()

# Créer le pipeline
pipeline = DocumentPipeline(config)

# Traiter un document
results = pipeline.process("invoice.png", run_evaluation=True, verbose=True)

# Afficher les résultats
pipeline.display_results(results)

# Exporter
pipeline.export_results(results, "output/results.json")
```

## Prérequis Azure

### Déploiements nécessaires

| Service | Modèle | Usage |
|---------|--------|-------|
| Azure AI Services | Mistral Document AI | OCR du document |
| Azure OpenAI | GPT-5.1 | Extraction structurée |
| Azure AI Foundry | gpt-4.1 | Évaluation cloud (LLM-as-Judge) |

### Authentification

- **Azure OpenAI** : Entra ID (DefaultAzureCredential)
- **Mistral OCR** : Clé API (Bearer token)
- **Évaluation** : Entra ID avec tenant spécifique

## Modes d'Extraction

| Mode | Description | Quand l'utiliser |
|------|-------------|------------------|
| `text_only` | OCR texte uniquement → GPT | Documents avec texte clair |
| `hybrid` | OCR texte + image → GPT | Documents avec éléments visuels (cases à cocher, signatures) |

## Métriques d'Évaluation

### Évaluation Cloud (Azure AI Foundry)

| Métrique | Description | Échelle |
|----------|-------------|---------|
| Groundedness | Données présentes dans le source OCR | 1-5 |
| Relevance | Pertinence de l'extraction | 1-5 |
| Coherence | Cohérence du résultat JSON | 1-5 |

### Confidence Locale (Anchoring)

| Métrique | Description | Échelle |
|----------|-------------|---------|
| Anchoring | Correspondance OCR vs Extraction | 0-1 |
| Format Validation | Respect des formats (dates, montants) | pass/fail |
| HITL Flag | Champs nécessitant review humaine | boolean |

## Schémas Personnalisés

Créez vos propres schémas d'extraction avec Pydantic :

```python
from pydantic import BaseModel, Field
from typing import Optional

class ContractExtraction(BaseModel):
    """Extraction de contrat."""

    contract_number: str = Field(description="Numéro de contrat")
    parties: list[str] = Field(description="Parties au contrat")
    effective_date: Optional[str] = Field(default=None, description="Date d'effet (YYYY-MM-DD)")
    termination_date: Optional[str] = Field(default=None, description="Date de fin (YYYY-MM-DD)")
    total_value: Optional[float] = Field(default=None, description="Valeur totale")

# Utiliser le schéma
results = pipeline.process_with_schema(
    file_path="contract.pdf",
    schema_model=ContractExtraction
)
```

## Documentation Azure

- [Azure OpenAI Chat Completions](https://learn.microsoft.com/azure/ai-services/openai/how-to/chatgpt)
- [Structured Outputs](https://learn.microsoft.com/azure/ai-services/openai/how-to/structured-outputs)
- [Mistral OCR on Azure](https://learn.microsoft.com/azure/ai-foundry/how-to/use-image-models)
- [Azure AI Foundry Evals API](https://learn.microsoft.com/azure/ai-foundry/how-to/evaluate-sdk)

## Licence

MIT
