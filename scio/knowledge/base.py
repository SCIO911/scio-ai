"""
SCIO Knowledge Base

Zentrale Wissensspeicherung mit Vektorsuche und Metadaten.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
import numpy as np

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class KnowledgeType(str, Enum):
    """Typen von Wissenseinträgen."""
    FACT = "fact"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    RULE = "rule"
    EXPERIENCE = "experience"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"
    CODE = "code"
    DOCUMENT = "document"


class ConfidenceLevel(str, Enum):
    """Vertrauensstufen für Wissen."""
    VERIFIED = "verified"        # Verifiziert und bestätigt
    HIGH = "high"               # Hohe Konfidenz
    MEDIUM = "medium"           # Mittlere Konfidenz
    LOW = "low"                 # Niedrige Konfidenz
    UNCERTAIN = "uncertain"     # Unsicher
    CONTRADICTED = "contradicted"  # Widersprüchlich


@dataclass
class KnowledgeEntry:
    """Ein Wissenseintrag in der Knowledge Base."""

    id: str
    content: str
    knowledge_type: KnowledgeType
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    embedding: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    tags: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert den Eintrag."""
        return {
            "id": self.id,
            "content": self.content,
            "knowledge_type": self.knowledge_type.value,
            "confidence": self.confidence.value,
            "metadata": self.metadata,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "related_ids": self.related_ids,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeEntry":
        """Deserialisiert einen Eintrag."""
        return cls(
            id=data["id"],
            content=data["content"],
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            confidence=ConfidenceLevel(data.get("confidence", "medium")),
            metadata=data.get("metadata", {}),
            source=data.get("source"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=data.get("tags", []),
            related_ids=data.get("related_ids", []),
            version=data.get("version", 1),
        )

    def content_hash(self) -> str:
        """Berechnet einen Hash des Inhalts."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class KnowledgeQuery:
    """Abfrage an die Knowledge Base."""

    query: str
    knowledge_types: Optional[list[KnowledgeType]] = None
    min_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    tags: Optional[list[str]] = None
    limit: int = 10
    threshold: float = 0.7
    include_related: bool = False
    time_range: Optional[tuple[datetime, datetime]] = None

    def matches_entry(self, entry: KnowledgeEntry) -> bool:
        """Prüft ob ein Eintrag die Filter erfüllt."""
        # Typ-Filter
        if self.knowledge_types and entry.knowledge_type not in self.knowledge_types:
            return False

        # Konfidenz-Filter
        confidence_order = [c.value for c in ConfidenceLevel]
        if confidence_order.index(entry.confidence.value) > confidence_order.index(self.min_confidence.value):
            return False

        # Tag-Filter
        if self.tags and not any(t in entry.tags for t in self.tags):
            return False

        # Zeit-Filter
        if self.time_range:
            if entry.created_at < self.time_range[0] or entry.created_at > self.time_range[1]:
                return False

        return True


@dataclass
class KnowledgeResult:
    """Ergebnis einer Wissensabfrage."""

    entry: KnowledgeEntry
    score: float
    relevance_explanation: Optional[str] = None
    related_entries: list["KnowledgeResult"] = field(default_factory=list)


class KnowledgeBase:
    """
    Zentrale Wissensdatenbank für SCIO.

    Features:
    - Vektorbasierte semantische Suche
    - Hybride Suche (Vektor + Keyword)
    - Wissenstypen und Konfidenzlevel
    - Versionierung und Geschichte
    - Beziehungen zwischen Einträgen
    - Persistente Speicherung
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedding_dim: int = 384,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.db_path = db_path or Path.home() / ".scio" / "knowledge.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.embedding_fn = embedding_fn or self._default_embedding

        # In-Memory Caches
        self._entries: dict[str, KnowledgeEntry] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._index_dirty = True

        # SQLite für Persistenz
        self._init_db()
        self._load_from_db()

        logger.info(
            "KnowledgeBase initialized",
            entries=len(self._entries),
            db_path=str(self.db_path),
        )

    def _init_db(self) -> None:
        """Initialisiert die SQLite-Datenbank."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    knowledge_type TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    metadata TEXT,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags TEXT,
                    related_ids TEXT,
                    version INTEGER DEFAULT 1,
                    embedding BLOB
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_type
                ON knowledge_entries(knowledge_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON knowledge_entries(created_at)
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
                USING fts5(id, content, tags)
            """)

    def _load_from_db(self) -> None:
        """Lädt alle Einträge aus der Datenbank."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM knowledge_entries")
            for row in cursor:
                entry = KnowledgeEntry(
                    id=row[0],
                    content=row[1],
                    knowledge_type=KnowledgeType(row[2]),
                    confidence=ConfidenceLevel(row[3]),
                    metadata=json.loads(row[4]) if row[4] else {},
                    source=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    tags=json.loads(row[8]) if row[8] else [],
                    related_ids=json.loads(row[9]) if row[9] else [],
                    version=row[10],
                )
                if row[11]:
                    entry.embedding = np.frombuffer(row[11], dtype=np.float32)
                    self._embeddings[entry.id] = entry.embedding
                self._entries[entry.id] = entry

    def _default_embedding(self, text: str) -> np.ndarray:
        """Einfache Hash-basierte Embedding-Funktion als Fallback."""
        # Tokenisiere
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        for i, word in enumerate(words):
            # Deterministischer Hash pro Wort
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for j in range(self.embedding_dim):
                embedding[j] += ((h >> (j % 32)) & 1) * (1.0 / (i + 1))

        # Normalisiere
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def add(
        self,
        content: str,
        knowledge_type: KnowledgeType = KnowledgeType.FACT,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        metadata: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
        tags: Optional[list[str]] = None,
        related_ids: Optional[list[str]] = None,
    ) -> KnowledgeEntry:
        """
        Fügt neues Wissen hinzu.

        Args:
            content: Der Wissensinhalt
            knowledge_type: Typ des Wissens
            confidence: Vertrauensstufe
            metadata: Zusätzliche Metadaten
            source: Quelle des Wissens
            tags: Tags für Kategorisierung
            related_ids: IDs verwandter Einträge

        Returns:
            Der erstellte KnowledgeEntry
        """
        entry_id = generate_id("know")
        embedding = self.embedding_fn(content)

        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            knowledge_type=knowledge_type,
            confidence=confidence,
            embedding=embedding,
            metadata=metadata or {},
            source=source,
            tags=tags or [],
            related_ids=related_ids or [],
        )

        self._entries[entry_id] = entry
        self._embeddings[entry_id] = embedding
        self._index_dirty = True

        # Persistiere
        self._save_entry(entry)

        logger.debug("Knowledge added", id=entry_id, type=knowledge_type.value)
        return entry

    def _save_entry(self, entry: KnowledgeEntry) -> None:
        """Speichert einen Eintrag in der Datenbank."""
        with sqlite3.connect(self.db_path) as conn:
            embedding_bytes = entry.embedding.tobytes() if entry.embedding is not None else None
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_entries
                (id, content, knowledge_type, confidence, metadata, source,
                 created_at, updated_at, tags, related_ids, version, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.content,
                entry.knowledge_type.value,
                entry.confidence.value,
                json.dumps(entry.metadata),
                entry.source,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                json.dumps(entry.tags),
                json.dumps(entry.related_ids),
                entry.version,
                embedding_bytes,
            ))
            # Update FTS
            conn.execute("DELETE FROM knowledge_fts WHERE id = ?", (entry.id,))
            conn.execute(
                "INSERT INTO knowledge_fts (id, content, tags) VALUES (?, ?, ?)",
                (entry.id, entry.content, " ".join(entry.tags)),
            )

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Holt einen Eintrag nach ID."""
        return self._entries.get(entry_id)

    def update(
        self,
        entry_id: str,
        content: Optional[str] = None,
        confidence: Optional[ConfidenceLevel] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[KnowledgeEntry]:
        """Aktualisiert einen bestehenden Eintrag."""
        entry = self._entries.get(entry_id)
        if not entry:
            return None

        if content is not None:
            entry.content = content
            entry.embedding = self.embedding_fn(content)
            self._embeddings[entry_id] = entry.embedding
            self._index_dirty = True

        if confidence is not None:
            entry.confidence = confidence

        if metadata is not None:
            entry.metadata.update(metadata)

        if tags is not None:
            entry.tags = tags

        entry.updated_at = now_utc()
        entry.version += 1

        self._save_entry(entry)
        logger.debug("Knowledge updated", id=entry_id, version=entry.version)

        return entry

    def delete(self, entry_id: str) -> bool:
        """Löscht einen Eintrag."""
        if entry_id not in self._entries:
            return False

        del self._entries[entry_id]
        if entry_id in self._embeddings:
            del self._embeddings[entry_id]
        self._index_dirty = True

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))
            conn.execute("DELETE FROM knowledge_fts WHERE id = ?", (entry_id,))

        logger.debug("Knowledge deleted", id=entry_id)
        return True

    def search(self, query: KnowledgeQuery) -> list[KnowledgeResult]:
        """
        Sucht nach relevantem Wissen.

        Kombiniert Vektorsuche mit Keyword-Suche für beste Ergebnisse.
        """
        results = []
        query_embedding = self.embedding_fn(query.query)

        # Berechne Ähnlichkeiten
        similarities: list[tuple[str, float]] = []
        for entry_id, embedding in self._embeddings.items():
            entry = self._entries[entry_id]
            if not query.matches_entry(entry):
                continue

            # Cosine Similarity
            similarity = float(np.dot(query_embedding, embedding))
            if similarity >= query.threshold:
                similarities.append((entry_id, similarity))

        # Keyword-Boost
        query_words = set(query.query.lower().split())
        for entry_id, base_sim in similarities:
            entry = self._entries[entry_id]
            content_words = set(entry.content.lower().split())
            keyword_overlap = len(query_words & content_words) / max(len(query_words), 1)
            boosted_sim = base_sim * 0.7 + keyword_overlap * 0.3

            results.append(KnowledgeResult(
                entry=entry,
                score=boosted_sim,
            ))

        # Sortiere nach Score
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:query.limit]

        # Lade verwandte Einträge falls gewünscht
        if query.include_related:
            for result in results:
                for related_id in result.entry.related_ids:
                    related_entry = self._entries.get(related_id)
                    if related_entry:
                        result.related_entries.append(KnowledgeResult(
                            entry=related_entry,
                            score=0.5,  # Standard-Score für Related
                        ))

        logger.debug(
            "Knowledge search",
            query=query.query[:50],
            results=len(results),
        )

        return results

    def keyword_search(self, keywords: str, limit: int = 10) -> list[KnowledgeEntry]:
        """Führt eine reine Keyword-Suche durch."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM knowledge_fts WHERE knowledge_fts MATCH ? LIMIT ?",
                (keywords, limit),
            )
            entry_ids = [row[0] for row in cursor]

        return [self._entries[eid] for eid in entry_ids if eid in self._entries]

    def find_related(self, entry_id: str, limit: int = 5) -> list[KnowledgeResult]:
        """Findet verwandte Einträge basierend auf Ähnlichkeit."""
        entry = self._entries.get(entry_id)
        if not entry or entry.embedding is None:
            return []

        results = []
        for other_id, embedding in self._embeddings.items():
            if other_id == entry_id:
                continue

            similarity = float(np.dot(entry.embedding, embedding))
            if similarity > 0.5:
                results.append(KnowledgeResult(
                    entry=self._entries[other_id],
                    score=similarity,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def link_entries(self, entry_id: str, related_id: str) -> bool:
        """Verknüpft zwei Einträge bidirektional."""
        entry1 = self._entries.get(entry_id)
        entry2 = self._entries.get(related_id)

        if not entry1 or not entry2:
            return False

        if related_id not in entry1.related_ids:
            entry1.related_ids.append(related_id)
            self._save_entry(entry1)

        if entry_id not in entry2.related_ids:
            entry2.related_ids.append(entry_id)
            self._save_entry(entry2)

        return True

    def get_by_type(
        self,
        knowledge_type: KnowledgeType,
        limit: int = 100,
    ) -> list[KnowledgeEntry]:
        """Holt alle Einträge eines Typs."""
        return [
            e for e in self._entries.values()
            if e.knowledge_type == knowledge_type
        ][:limit]

    def get_by_tags(self, tags: list[str], match_all: bool = False) -> list[KnowledgeEntry]:
        """Holt Einträge mit bestimmten Tags."""
        results = []
        for entry in self._entries.values():
            if match_all:
                if all(t in entry.tags for t in tags):
                    results.append(entry)
            else:
                if any(t in entry.tags for t in tags):
                    results.append(entry)
        return results

    def export_to_json(self, path: Path) -> int:
        """Exportiert alle Einträge als JSON."""
        data = [e.to_dict() for e in self._entries.values()]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return len(data)

    def import_from_json(self, path: Path) -> int:
        """Importiert Einträge aus JSON."""
        data = json.loads(path.read_text())
        count = 0
        for item in data:
            if item["id"] not in self._entries:
                entry = KnowledgeEntry.from_dict(item)
                entry.embedding = self.embedding_fn(entry.content)
                self._entries[entry.id] = entry
                self._embeddings[entry.id] = entry.embedding
                self._save_entry(entry)
                count += 1
        self._index_dirty = True
        return count

    def stats(self) -> dict[str, Any]:
        """Gibt Statistiken über die Wissensbasis zurück."""
        type_counts = {}
        confidence_counts = {}

        for entry in self._entries.values():
            type_counts[entry.knowledge_type.value] = type_counts.get(entry.knowledge_type.value, 0) + 1
            confidence_counts[entry.confidence.value] = confidence_counts.get(entry.confidence.value, 0) + 1

        return {
            "total_entries": len(self._entries),
            "by_type": type_counts,
            "by_confidence": confidence_counts,
            "embedding_dim": self.embedding_dim,
            "db_path": str(self.db_path),
        }

    def clear(self) -> None:
        """Löscht alle Einträge."""
        self._entries.clear()
        self._embeddings.clear()
        self._index_dirty = True

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM knowledge_entries")
            conn.execute("DELETE FROM knowledge_fts")

        logger.info("KnowledgeBase cleared")
