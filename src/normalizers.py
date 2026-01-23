"""
Fonctions de normalisation par type de format.

Transforme les valeurs OCR pour correspondre au format extrait.
Cela permet de comparer des valeurs sémantiquement équivalentes:
- "15/01/2024" vs "2024-01-15" → même date
- "1,234.56 €" vs "1234.56" → même montant
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


def normalize_date(ocr_value: str) -> Optional[str]:
    """
    Normalise une date OCR vers ISO format (YYYY-MM-DD).

    Supporte plusieurs formats d'entrée:
    - DD/MM/YYYY (européen)
    - MM/DD/YYYY (américain)
    - DD-MM-YYYY
    - YYYY/MM/DD
    - YYYY-MM-DD (déjà ISO)

    Args:
        ocr_value: Texte OCR contenant une date

    Returns:
        Date au format ISO (YYYY-MM-DD) ou None si pas de date trouvée
    """
    if not ocr_value:
        return None

    ocr_str = str(ocr_value).strip()

    # Patterns de date courants avec leur format strptime
    # Note: On essaie d'abord les formats européens (DD/MM/YYYY) car plus courants en France
    patterns: List[Tuple[str, str]] = [
        # Formats avec /
        (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"),  # DD/MM/YYYY (européen)
        (r"(\d{4})/(\d{2})/(\d{2})", "%Y/%m/%d"),  # YYYY/MM/DD

        # Formats avec -
        (r"(\d{2})-(\d{2})-(\d{4})", "%d-%m-%Y"),  # DD-MM-YYYY
        (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),  # YYYY-MM-DD (ISO)

        # Formats avec .
        (r"(\d{2})\.(\d{2})\.(\d{4})", "%d.%m.%Y"),  # DD.MM.YYYY (allemand)
    ]

    for pattern, date_format in patterns:
        match = re.search(pattern, ocr_str)
        if match:
            try:
                dt = datetime.strptime(match.group(), date_format)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                # Format invalide (ex: 31/02/2024), essayer le pattern suivant
                continue

    return None  # Pas de format reconnu


def normalize_amount(ocr_value: Any) -> Optional[str]:
    """
    Normalise un montant OCR vers format numérique standardisé.

    Gère les formats:
    - "1,234.56 €" → "1234.56" (US)
    - "1.234,56 €" → "1234.56" (européen)
    - "1 234,56" → "1234.56" (français avec espace)
    - "$1,234.56" → "1234.56"

    Args:
        ocr_value: Texte OCR ou nombre contenant un montant

    Returns:
        Montant formaté avec 2 décimales ou None si invalide
    """
    if ocr_value is None:
        return None

    # Si c'est déjà un nombre
    if isinstance(ocr_value, (int, float)):
        return f"{float(ocr_value):.2f}"

    ocr_str = str(ocr_value).strip()
    if not ocr_str:
        return None

    # Retirer symboles monétaires et espaces
    cleaned = re.sub(r'[$€£¥\s]', '', ocr_str)

    # Retirer les lettres (ex: "EUR", "USD")
    cleaned = re.sub(r'[A-Za-z]', '', cleaned)

    if not cleaned:
        return None

    # Gérer format européen (1.234,56) vs US (1,234.56)
    if ',' in cleaned and '.' in cleaned:
        # Les deux séparateurs présents
        if cleaned.rfind(',') > cleaned.rfind('.'):
            # Format européen: le . est séparateur de milliers, la , est décimale
            # 1.234,56 → 1234.56
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            # Format US: la , est séparateur de milliers, le . est décimale
            # 1,234.56 → 1234.56
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        # Virgule seule
        # Vérifier si c'est un séparateur de milliers ou décimal
        parts = cleaned.split(',')
        if len(parts) == 2 and len(parts[1]) == 2:
            # Probablement décimale européenne (ex: 1234,56)
            cleaned = cleaned.replace(',', '.')
        elif len(parts) == 2 and len(parts[1]) == 3:
            # Probablement séparateur de milliers (ex: 1,234)
            cleaned = cleaned.replace(',', '')
        else:
            # Par défaut, traiter comme décimale
            cleaned = cleaned.replace(',', '.')
    # Si seulement des points, c'est soit décimal soit milliers
    # On garde tel quel car le point est le séparateur standard

    try:
        value = float(cleaned)
        return f"{value:.2f}"
    except ValueError:
        return None


def normalize_phone(ocr_value: str) -> Optional[str]:
    """
    Normalise un numéro de téléphone (chiffres uniquement).

    Exemples:
    - "06 12 34 56 78" → "0612345678"
    - "+33 6 12 34 56 78" → "33612345678"
    - "(01) 234-5678" → "012345678"

    Args:
        ocr_value: Texte OCR contenant un numéro de téléphone

    Returns:
        Numéro avec uniquement les chiffres ou None si vide
    """
    if not ocr_value:
        return None

    # Garder uniquement les chiffres
    digits = re.sub(r'\D', '', str(ocr_value))

    return digits if digits else None


# Registry des normalizers par type de format
NORMALIZERS: Dict[str, Callable[[Any], Optional[str]]] = {
    "date": normalize_date,
    "amount": normalize_amount,
    "phone": normalize_phone,
}


def normalize_value(value: Any, format_type: str) -> Optional[str]:
    """
    Normalise une valeur selon son type de format.

    Args:
        value: Valeur à normaliser
        format_type: Type de format ("date", "amount", "phone")

    Returns:
        Valeur normalisée ou None si le format n'est pas supporté ou la valeur invalide
    """
    normalizer = NORMALIZERS.get(format_type)
    if normalizer:
        return normalizer(value)
    return None


def find_and_normalize_in_text(
    text: str,
    format_type: str,
    target_value: str
) -> Optional[str]:
    """
    Cherche une valeur dans le texte OCR et la normalise.

    Utile quand on veut trouver la valeur OCR correspondant à une extraction.
    Par exemple, si on a extrait "2024-01-15", on cherche dans l'OCR
    toutes les dates et on retourne celle qui match après normalisation.

    Args:
        text: Texte OCR complet
        format_type: Type de format recherché
        target_value: Valeur extraite à matcher

    Returns:
        Valeur OCR normalisée qui correspond ou None
    """
    if format_type == "date":
        # Chercher toutes les dates potentielles dans le texte
        date_patterns = [
            r"\d{2}/\d{2}/\d{4}",
            r"\d{4}/\d{2}/\d{2}",
            r"\d{2}-\d{2}-\d{4}",
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}\.\d{2}\.\d{4}",
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                normalized = normalize_date(match.group())
                if normalized == target_value:
                    return normalized

    elif format_type == "amount":
        # Chercher toutes les valeurs numériques potentielles
        amount_patterns = [
            r"[\d\s,.]+(?:\s*[€$£¥])?",
            r"[€$£¥]\s*[\d\s,.]+",
        ]
        for pattern in amount_patterns:
            for match in re.finditer(pattern, text):
                normalized = normalize_amount(match.group())
                if normalized == target_value:
                    return normalized

    return None
