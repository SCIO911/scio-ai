"""
SCIO Memory Store

Persistenter Speicher für Agenten-Kontext und Zwischenergebnisse.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc, hash_content

logger = get_logger(__name__)


@dataclass
class MemoryEntry:
    """Ein einzelner Memory-Eintrag."""

    key: str
    value: Any
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    ttl_seconds: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        age = (now_utc() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class MemoryStore:
    """
    In-Memory Store mit optionaler Persistenz.

    Features:
    - Key-Value Speicherung
    - TTL Support
    - Tag-basierte Suche
    - Optionale Disk-Persistenz
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self._store: dict[str, MemoryEntry] = {}
        self.persist_path = persist_path
        self.logger = get_logger(__name__, component="memory_store")

        if persist_path and persist_path.exists():
            self._load_from_disk()

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Speichert einen Wert.

        Args:
            key: Schlüssel
            value: Wert
            ttl_seconds: Time-to-live in Sekunden
            tags: Optionale Tags für Suche
            metadata: Optionale Metadaten
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            ttl_seconds=ttl_seconds,
            tags=tags or [],
            metadata=metadata or {},
        )

        if key in self._store:
            entry.created_at = self._store[key].created_at

        self._store[key] = entry
        self.logger.debug("Memory set", key=key, ttl=ttl_seconds)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Liest einen Wert.

        Args:
            key: Schlüssel
            default: Standardwert wenn nicht gefunden

        Returns:
            Gespeicherter Wert oder default
        """
        entry = self._store.get(key)

        if entry is None:
            return default

        if entry.is_expired:
            self.delete(key)
            return default

        return entry.value

    def get_entry(self, key: str) -> Optional[MemoryEntry]:
        """Gibt den vollständigen Entry zurück."""
        entry = self._store.get(key)
        if entry and entry.is_expired:
            self.delete(key)
            return None
        return entry

    def delete(self, key: str) -> bool:
        """Löscht einen Eintrag."""
        if key in self._store:
            del self._store[key]
            self.logger.debug("Memory deleted", key=key)
            return True
        return False

    def exists(self, key: str) -> bool:
        """Prüft ob ein Schlüssel existiert."""
        entry = self._store.get(key)
        if entry and entry.is_expired:
            self.delete(key)
            return False
        return entry is not None

    def find_by_tag(self, tag: str) -> list[MemoryEntry]:
        """Findet Einträge nach Tag."""
        self._cleanup_expired()
        return [e for e in self._store.values() if tag in e.tags]

    def find_by_prefix(self, prefix: str) -> list[MemoryEntry]:
        """Findet Einträge nach Key-Prefix."""
        self._cleanup_expired()
        return [e for e in self._store.values() if e.key.startswith(prefix)]

    def clear(self) -> None:
        """Löscht alle Einträge."""
        self._store.clear()
        self.logger.info("Memory cleared")

    def keys(self) -> list[str]:
        """Gibt alle Schlüssel zurück."""
        self._cleanup_expired()
        return list(self._store.keys())

    def size(self) -> int:
        """Gibt die Anzahl der Einträge zurück."""
        return len(self._store)

    def persist(self) -> None:
        """Speichert auf Disk (wenn persist_path gesetzt)."""
        if self.persist_path is None:
            return

        self._cleanup_expired()

        data = {key: entry.to_dict() for key, entry in self._store.items()}

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info("Memory persisted", path=str(self.persist_path))

    def _load_from_disk(self) -> None:
        """Lädt von Disk."""
        try:
            with open(self.persist_path) as f:
                data = json.load(f)

            for key, entry_data in data.items():
                self._store[key] = MemoryEntry(
                    key=entry_data["key"],
                    value=entry_data["value"],
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    updated_at=datetime.fromisoformat(entry_data["updated_at"]),
                    ttl_seconds=entry_data.get("ttl_seconds"),
                    tags=entry_data.get("tags", []),
                    metadata=entry_data.get("metadata", {}),
                )

            self.logger.info(
                "Memory loaded from disk",
                path=str(self.persist_path),
                entries=len(self._store),
            )

        except Exception as e:
            self.logger.error("Failed to load memory", error=str(e))

    def _cleanup_expired(self) -> None:
        """Entfernt abgelaufene Einträge."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for key in expired:
            del self._store[key]
