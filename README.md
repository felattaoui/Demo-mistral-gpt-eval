# 🔍 Azure Document Extraction Pipeline

Pipeline d'extraction de données structurées à partir de documents (PDF, images) utilisant les services Azure AI.

## ✨ Fonctionnalités

- **OCR** avec Mistral Document AI
- **Extraction structurée** avec GPT-5.1 (Responses API + Structured Outputs)
- **Évaluation de qualité** avec Azure AI Evaluation SDK
- **Authentification Entra ID** avec refresh automatique du token
- **Schémas personnalisables** via Pydantic

## 📁 Structure

```
azure-document-extraction/
├── src/
│   ├── config.py       # Configuration (.env)
│   ├── utils.py        # Utilitaires (base64, file info)
│   ├── ocr.py          # Client Mistral OCR
│   ├── extractor.py    # Extraction GPT-5.1 (Responses API)
│   ├── evaluator.py    # Évaluation qualité
│   ├── schemas.py      # Schémas Pydantic
│   └── pipeline.py     # Pipeline complet
│
├── notebooks/
│   └── tutorial.ipynb  # Notebook pédagogique
│
├── examples/           # Documents de test
├── output/             # Résultats exportés
│
├── .env.example        # Template de configuration
├── requirements.txt    # Dépendances Python
└── README.md
```

## 🚀 Démarrage Rapide

### 1. Installation

```bash
# Cloner le projet
git clone <repo>
cd azure-document-extraction

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copier le template
cp .env.example .env

# Éditer avec vos valeurs Azure
nano .env
```

### 3. Authentification Azure

```bash
# Se connecter (Entra ID)
az login
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
results = pipeline.process("examples/invoice.png", verbose=True)

# Afficher les résultats
pipeline.display_results(results)
```

## 📋 Prérequis Azure

### Déploiements nécessaires

| Service | Modèle | Usage |
|---------|--------|-------|
| Azure AI Services | Mistral Document AI | OCR |
| Azure OpenAI | GPT-5.1 | Extraction |
| Azure OpenAI | GPT-4o | Évaluation (optionnel) |

### Authentification

Ce projet utilise **Entra ID** (DefaultAzureCredential) pour Azure OpenAI.
Les clés API sont utilisées uniquement pour Mistral OCR.

## 🎨 Schémas Personnalisés

Créez vos propres schémas d'extraction :

```python
from pydantic import BaseModel, Field
from typing import Optional

class ContractExtraction(BaseModel):
    \"\"\"Extraction de contrat.\"\"\"
    
    contract_number: str = Field(description="Numéro de contrat")
    parties: list[str] = Field(description="Parties au contrat")
    effective_date: Optional[str] = Field(default=None)
    termination_date: Optional[str] = Field(default=None)
    total_value: Optional[float] = Field(default=None)
    confidence_score: float = Field(description="Score 0-1")

# Utiliser le schéma
results = pipeline.process_with_schema(
    file_path="contract.pdf",
    schema_model=ContractExtraction
)
```

## 📊 Métriques d'Évaluation

| Métrique | Description | Échelle |
|----------|-------------|---------|
| Groundedness | Données présentes dans le source | 1-5 |
| Relevance | Pertinence de l'extraction | 1-5 |
| Coherence | Cohérence du résultat | 1-5 |
| Validation | Respect des formats (dates, etc.) | 0-100% |

## 🔧 Configuration Avancée

### Variables d'environnement

```env
# Mistral OCR
MISTRAL_ENDPOINT=https://xxx.services.ai.azure.com
MISTRAL_API_KEY=your-key
MISTRAL_MODEL=mistral-document-ai-2505-2

# Azure OpenAI (Entra ID - pas de clé)
AZURE_OPENAI_ENDPOINT=https://xxx.cognitiveservices.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-5.1

# Azure AI Foundry Cloud Evaluation (optionnel)
PROJECT_ENDPOINT=https://xxx.services.ai.azure.com/api/projects/project-name
EVAL_MODEL_DEPLOYMENT=gpt-4o
MODEL_ENDPOINT=https://xxx.services.ai.azure.com   # Optional: defaults to derived from PROJECT_ENDPOINT
MODEL_API_KEY=your-key                              # Optional: uses DefaultAzureCredential if not set
```

### Mode PDF Direct

Pour les PDFs, vous pouvez bypasser l'OCR :

```python
results = pipeline.process(
    file_path="document.pdf",
    use_direct_pdf=True  # Utilise le support PDF natif de GPT-5.1
)
```

## 📚 Documentation

- [Azure OpenAI Responses API](https://learn.microsoft.com/azure/ai-services/openai/how-to/responses)
- [Structured Outputs](https://learn.microsoft.com/azure/ai-services/openai/how-to/structured-outputs)
- [Mistral OCR on Azure](https://learn.microsoft.com/azure/ai-foundry/how-to/use-image-models)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/azure/ai-studio/how-to/evaluate-sdk)

## 📝 Licence

MIT
