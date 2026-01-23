"""
Configuration des formats de champs pour la normalisation.

Ce module définit:
- Les types de formats supportés (date, amount, phone)
- La configuration explicite par champ
- L'auto-détection en fallback basée sur le nom du champ
"""

from __future__ import annotations

from typing import Any, Optional


# Formats supportés
FORMAT_DATE = "date"
FORMAT_AMOUNT = "amount"
FORMAT_PHONE = "phone"

# Configuration explicite (prioritaire)
# Mapping: field_name -> format_type
FIELD_FORMATS = {
    "document_date": FORMAT_DATE,
    "total_amount": FORMAT_AMOUNT,
    "subtotal": FORMAT_AMOUNT,
    "tax_amount": FORMAT_AMOUNT,
    # Ajouter d'autres champs au besoin
}


def auto_detect_format(field_name: str, value: Any = None) -> Optional[str]:
    """
    Auto-détection du format basée sur le nom du champ.

    Utilisé en fallback si le champ n'est pas dans FIELD_FORMATS.

    Args:
        field_name: Nom du champ
        value: Valeur extraite (optionnel, pour détection future basée sur valeur)

    Returns:
        Type de format détecté ou None si pas de format connu
    """
    field_lower = field_name.lower()

    # Détection par nom de champ
    if "date" in field_lower:
        return FORMAT_DATE

    if any(kw in field_lower for kw in ["amount", "total", "price", "cost", "subtotal", "tax"]):
        return FORMAT_AMOUNT

    if any(kw in field_lower for kw in ["phone", "tel", "fax", "mobile"]):
        return FORMAT_PHONE

    return None  # Pas de normalisation


def get_field_format(field_name: str, value: Any = None) -> Optional[str]:
    """
    Récupère le format d'un champ (config explicite ou auto-détection).

    Args:
        field_name: Nom du champ
        value: Valeur extraite (optionnel)

    Returns:
        Type de format ou None si pas de format connu
    """
    # Config explicite prioritaire
    if field_name in FIELD_FORMATS:
        return FIELD_FORMATS[field_name]

    # Fallback auto-détection
    return auto_detect_format(field_name, value)


def register_field_format(field_name: str, format_type: str) -> None:
    """
    Enregistre dynamiquement un format pour un champ.

    Args:
        field_name: Nom du champ
        format_type: Type de format (FORMAT_DATE, FORMAT_AMOUNT, FORMAT_PHONE)
    """
    FIELD_FORMATS[field_name] = format_type
