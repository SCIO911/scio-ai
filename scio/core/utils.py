"""
SCIO Utilities

Allgemeine Hilfsfunktionen für das Framework.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """
    Generiert eine eindeutige ID.

    Args:
        prefix: Optionales Präfix (z.B. 'exp', 'agent')

    Returns:
        Eindeutige ID im Format: prefix-uuid4[:8]
    """
    short_uuid = str(uuid.uuid4())[:8]
    if prefix:
        return f"{prefix}-{short_uuid}"
    return short_uuid


def now_utc() -> datetime:
    """Gibt aktuelle UTC-Zeit zurück."""
    return datetime.now(timezone.utc)


def hash_content(content: str | bytes, algorithm: str = "sha256") -> str:
    """
    Berechnet Hash eines Inhalts.

    Args:
        content: Zu hashender Inhalt
        algorithm: Hash-Algorithmus (sha256, sha512, md5)

    Returns:
        Hexadezimaler Hash-String
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    hasher = hashlib.new(algorithm)
    hasher.update(content)
    return hasher.hexdigest()


def ensure_path(path: Path | str) -> Path:
    """
    Stellt sicher, dass ein Pfad existiert.

    Args:
        path: Pfad zum Erstellen

    Returns:
        Path-Objekt
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Führt zwei Dictionaries rekursiv zusammen.

    Args:
        base: Basis-Dictionary
        override: Überschreibungs-Dictionary

    Returns:
        Zusammengeführtes Dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Kürzt einen String auf maximale Länge."""
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def safe_get(data: dict[str, Any], *keys: str, default: T = None) -> T | Any:
    """
    Sicherer Zugriff auf verschachtelte Dictionary-Werte.

    Args:
        data: Dictionary
        *keys: Schlüssel-Pfad
        default: Standardwert wenn nicht gefunden

    Example:
        value = safe_get(data, "level1", "level2", "key", default="fallback")
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
