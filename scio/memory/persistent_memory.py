"""
SCIO Persistent Memory - Unbegrenztes Langzeit-Gedächtnis

Features:
- Episodisches Gedächtnis (Ereignisse, Erfahrungen)
- Semantisches Gedächtnis (Fakten, Konzepte)
- Prozedurales Gedächtnis (Skills, Prozesse)
- Arbeitsgedächtnis (aktuelle Aufgabe)
- Automatische Konsolidierung
- Intelligentes Vergessen
- Kontextbasiertes Abrufen
"""

import asyncio
import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import math

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MemoryType(str, Enum):
    """Typen von Erinnerungen."""
    EPISODIC = "episodic"       # Ereignisse und Erfahrungen
    SEMANTIC = "semantic"       # Fakten und Wissen
    PROCEDURAL = "procedural"   # Fähigkeiten und Prozesse
    WORKING = "working"         # Kurzzeit / Arbeitsgedächtnis


class MemoryStrength(str, Enum):
    """Stärke einer Erinnerung."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    PERMANENT = "permanent"


@dataclass
class Memory:
    """Eine einzelne Erinnerung."""

    id: str
    content: str
    memory_type: MemoryType
    importance: float = 0.5  # 0-1
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)  # IDs verwandter Erinnerungen

    # Zeitliche Informationen
    created_at: datetime = field(default_factory=now_utc)
    last_accessed: datetime = field(default_factory=now_utc)
    access_count: int = 0

    # Gedächtnisstärke
    strength: float = 1.0  # Zerfällt mit der Zeit
    consolidation_level: int = 0  # 0=working, 1=short-term, 2=long-term

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "metadata": self.metadata,
            "tags": self.tags,
            "associations": self.associations,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "strength": self.strength,
            "consolidation_level": self.consolidation_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            associations=data.get("associations", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            strength=data.get("strength", 1.0),
            consolidation_level=data.get("consolidation_level", 0),
        )

    def decay(self, hours_passed: float) -> None:
        """Lässt die Erinnerung mit der Zeit verblassen."""
        # Ebbinghaus Vergessenskurve: R = e^(-t/S)
        # S = Stärke der Erinnerung (höher = langsamer Verfall)
        stability = 24 * (1 + self.importance) * (1 + math.log1p(self.access_count))
        self.strength = math.exp(-hours_passed / stability)

    def reinforce(self, boost: float = 0.2) -> None:
        """Verstärkt die Erinnerung durch Abruf."""
        self.strength = min(1.0, self.strength + boost)
        self.access_count += 1
        self.last_accessed = now_utc()


@dataclass
class MemoryConfig:
    """Konfiguration für das Gedächtnissystem."""

    # Speicherung
    db_path: Optional[Path] = None
    max_memories: int = 1_000_000
    max_working_memory: int = 100

    # Konsolidierung
    consolidation_interval_hours: float = 1.0
    short_term_threshold: float = 0.5
    long_term_threshold: float = 0.8

    # Vergessen
    decay_enabled: bool = True
    decay_interval_hours: float = 24.0
    forget_threshold: float = 0.1

    # Abruf
    default_recall_limit: int = 10
    similarity_threshold: float = 0.7

    # Embedding
    embedding_dim: int = 768
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================================
# MEMORY INDEX
# ============================================================================

class MemoryIndex:
    """
    Index für schnellen Zugriff auf Erinnerungen.

    Unterstützt:
    - Semantische Suche (Embedding-basiert)
    - Tag-basierte Suche
    - Zeitbasierte Suche
    - Assoziative Suche
    """

    def __init__(self):
        self._by_type: Dict[MemoryType, set] = defaultdict(set)
        self._by_tag: Dict[str, set] = defaultdict(set)
        self._by_date: Dict[str, set] = defaultdict(set)  # YYYY-MM-DD -> IDs
        self._associations: Dict[str, set] = defaultdict(set)

        # Simple vector index (in production würde man FAISS/Annoy verwenden)
        self._embeddings: Dict[str, List[float]] = {}

    def add(self, memory: Memory) -> None:
        """Fügt eine Erinnerung zum Index hinzu."""
        self._by_type[memory.memory_type].add(memory.id)

        for tag in memory.tags:
            self._by_tag[tag.lower()].add(memory.id)

        date_key = memory.created_at.strftime("%Y-%m-%d")
        self._by_date[date_key].add(memory.id)

        for assoc_id in memory.associations:
            self._associations[memory.id].add(assoc_id)
            self._associations[assoc_id].add(memory.id)

        if memory.embedding:
            self._embeddings[memory.id] = memory.embedding

    def remove(self, memory_id: str) -> None:
        """Entfernt eine Erinnerung aus dem Index."""
        for type_set in self._by_type.values():
            type_set.discard(memory_id)

        for tag_set in self._by_tag.values():
            tag_set.discard(memory_id)

        for date_set in self._by_date.values():
            date_set.discard(memory_id)

        if memory_id in self._associations:
            for assoc_id in self._associations[memory_id]:
                self._associations[assoc_id].discard(memory_id)
            del self._associations[memory_id]

        self._embeddings.pop(memory_id, None)

    def search_by_type(self, memory_type: MemoryType) -> set:
        """Sucht nach Typ."""
        return self._by_type.get(memory_type, set())

    def search_by_tag(self, tag: str) -> set:
        """Sucht nach Tag."""
        return self._by_tag.get(tag.lower(), set())

    def search_by_date(self, date: datetime) -> set:
        """Sucht nach Datum."""
        date_key = date.strftime("%Y-%m-%d")
        return self._by_date.get(date_key, set())

    def search_by_date_range(self, start: datetime, end: datetime) -> set:
        """Sucht in einem Datumsbereich."""
        results = set()
        current = start
        while current <= end:
            results |= self.search_by_date(current)
            current += timedelta(days=1)
        return results

    def get_associations(self, memory_id: str) -> set:
        """Holt assoziierte Erinnerungen."""
        return self._associations.get(memory_id, set())

    def search_by_similarity(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Semantische Suche basierend auf Embedding-Ähnlichkeit."""
        if not self._embeddings:
            return []

        results = []

        for memory_id, embedding in self._embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= threshold:
                results.append((memory_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Berechnet Cosinus-Ähnlichkeit."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# ============================================================================
# PERSISTENT MEMORY
# ============================================================================

class PersistentMemory:
    """
    Unbegrenztes Langzeit-Gedächtnis für SCIO.

    Implementiert die volle Gedächtnishierarchie:
    - Arbeitsgedächtnis (aktuelle Aufgabe)
    - Kurzzeitgedächtnis (Minuten bis Stunden)
    - Langzeitgedächtnis (permanent)

    Mit automatischer Konsolidierung und intelligentem Vergessen.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()

        # Speicherpfad
        if self.config.db_path is None:
            self.config.db_path = Path.home() / ".scio" / "memory.db"
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Memory Store
        self._memories: Dict[str, Memory] = {}
        self._working_memory: List[str] = []  # IDs, begrenzte Kapazität

        # Index
        self._index = MemoryIndex()

        # Background Tasks
        self._consolidation_task: Optional[asyncio.Task] = None
        self._decay_task: Optional[asyncio.Task] = None
        self._running = False

        # Embedding Callback (wird von außen gesetzt)
        self._embedding_callback: Optional[Callable] = None

        # Initialize DB
        self._init_db()
        self._load_from_db()

        logger.info(
            "PersistentMemory initialized",
            total_memories=len(self._memories),
            db_path=str(self.config.db_path),
        )

    def _init_db(self) -> None:
        """Initialisiert die SQLite-Datenbank."""
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    embedding BLOB,
                    metadata TEXT,
                    tags TEXT,
                    associations TEXT,
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    strength REAL DEFAULT 1.0,
                    consolidation_level INTEGER DEFAULT 0
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_importance ON memories(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_strength ON memories(strength)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_created ON memories(created_at)")

    def _load_from_db(self) -> None:
        """Lädt Erinnerungen aus der Datenbank."""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.execute("SELECT * FROM memories WHERE strength > ?", (self.config.forget_threshold,))

            for row in cursor:
                memory = Memory(
                    id=row[0],
                    content=row[1],
                    memory_type=MemoryType(row[2]),
                    importance=row[3] or 0.5,
                    embedding=json.loads(row[4]) if row[4] else None,
                    metadata=json.loads(row[5]) if row[5] else {},
                    tags=json.loads(row[6]) if row[6] else [],
                    associations=json.loads(row[7]) if row[7] else [],
                    created_at=datetime.fromisoformat(row[8]) if row[8] else now_utc(),
                    last_accessed=datetime.fromisoformat(row[9]) if row[9] else now_utc(),
                    access_count=row[10] or 0,
                    strength=row[11] or 1.0,
                    consolidation_level=row[12] or 0,
                )

                self._memories[memory.id] = memory
                self._index.add(memory)

    def _save_memory(self, memory: Memory) -> None:
        """Speichert eine Erinnerung in der Datenbank."""
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, memory_type, importance, embedding, metadata, tags,
                 associations, created_at, last_accessed, access_count, strength, consolidation_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.content,
                memory.memory_type.value,
                memory.importance,
                json.dumps(memory.embedding) if memory.embedding else None,
                json.dumps(memory.metadata),
                json.dumps(memory.tags),
                json.dumps(memory.associations),
                memory.created_at.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                memory.strength,
                memory.consolidation_level,
            ))

    def _delete_memory(self, memory_id: str) -> None:
        """Löscht eine Erinnerung aus der Datenbank."""
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    # ========================================================================
    # PUBLIC API - REMEMBER
    # ========================================================================

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        associations: Optional[List[str]] = None,
    ) -> Memory:
        """
        Speichert eine neue Erinnerung.

        Args:
            content: Der Inhalt der Erinnerung
            memory_type: Typ der Erinnerung
            importance: Wichtigkeit (0-1)
            tags: Optionale Tags für Kategorisierung
            metadata: Optionale Metadaten
            associations: IDs verwandter Erinnerungen

        Returns:
            Die erstellte Memory-Instanz
        """
        # Erstelle Embedding wenn Callback verfügbar
        embedding = None
        if self._embedding_callback:
            try:
                embedding = await self._embedding_callback(content)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        # Erstelle Memory
        memory_id = generate_id("mem")
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            tags=tags or [],
            metadata=metadata or {},
            associations=associations or [],
        )

        # Speichere
        self._memories[memory_id] = memory
        self._index.add(memory)
        self._save_memory(memory)

        # Working Memory Management
        if memory_type == MemoryType.WORKING:
            self._working_memory.append(memory_id)
            if len(self._working_memory) > self.config.max_working_memory:
                old_id = self._working_memory.pop(0)
                if old_id in self._memories:
                    self._memories[old_id].memory_type = MemoryType.EPISODIC
                    self._save_memory(self._memories[old_id])

        logger.debug("Memory stored", id=memory_id, type=memory_type.value)
        return memory

    # ========================================================================
    # PUBLIC API - RECALL
    # ========================================================================

    async def recall(
        self,
        query: str,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        min_strength: float = 0.0,
    ) -> List[Memory]:
        """
        Ruft Erinnerungen basierend auf einer Abfrage ab.

        Kombiniert semantische Suche mit Filtern.

        Args:
            query: Suchanfrage
            limit: Maximale Anzahl Ergebnisse
            memory_types: Filter nach Typen
            tags: Filter nach Tags
            min_importance: Minimum Wichtigkeit
            min_strength: Minimum Stärke

        Returns:
            Liste von Erinnerungen, sortiert nach Relevanz
        """
        candidates = set(self._memories.keys())

        # Filter by type
        if memory_types:
            type_matches = set()
            for mt in memory_types:
                type_matches |= self._index.search_by_type(mt)
            candidates &= type_matches

        # Filter by tags
        if tags:
            for tag in tags:
                candidates &= self._index.search_by_tag(tag)

        # Filter by importance and strength
        candidates = {
            mid for mid in candidates
            if self._memories[mid].importance >= min_importance
            and self._memories[mid].strength >= min_strength
        }

        # Semantic search if embedding available
        if self._embedding_callback and query:
            try:
                query_embedding = await self._embedding_callback(query)
                similarity_results = self._index.search_by_similarity(
                    query_embedding, top_k=limit * 2
                )

                # Combine with candidates
                semantic_ids = {mid for mid, _ in similarity_results if mid in candidates}
                candidates = semantic_ids if semantic_ids else candidates

            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Get memories and score them
        results = []
        for mid in candidates:
            memory = self._memories[mid]

            # Score based on multiple factors
            score = self._calculate_relevance_score(memory, query)
            results.append((memory, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Reinforce accessed memories
        final_results = []
        for memory, score in results[:limit]:
            memory.reinforce()
            self._save_memory(memory)
            final_results.append(memory)

        return final_results

    def _calculate_relevance_score(self, memory: Memory, query: str) -> float:
        """Berechnet einen Relevanz-Score für eine Erinnerung."""
        score = 0.0

        # Textuelle Ähnlichkeit
        query_words = set(query.lower().split())
        content_words = set(memory.content.lower().split())
        if query_words:
            word_overlap = len(query_words & content_words) / len(query_words)
            score += word_overlap * 0.3

        # Importance
        score += memory.importance * 0.2

        # Strength (Gedächtnisstärke)
        score += memory.strength * 0.2

        # Recency
        age_hours = (now_utc() - memory.last_accessed).total_seconds() / 3600
        recency = math.exp(-age_hours / 168)  # Halbwertszeit: 1 Woche
        score += recency * 0.15

        # Access frequency
        frequency = math.log1p(memory.access_count) / 10
        score += min(frequency, 0.15)

        return score

    async def recall_by_association(
        self,
        memory_id: str,
        depth: int = 2,
        limit: int = 20,
    ) -> List[Memory]:
        """
        Ruft assoziierte Erinnerungen ab.

        Folgt dem assoziativen Netzwerk.
        """
        visited = set()
        results = []

        def traverse(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited:
                return
            visited.add(current_id)

            memory = self._memories.get(current_id)
            if memory:
                results.append(memory)
                memory.reinforce(boost=0.1)
                self._save_memory(memory)

                # Follow associations
                for assoc_id in self._index.get_associations(current_id):
                    traverse(assoc_id, current_depth + 1)

        traverse(memory_id, 0)

        # Sort by relevance
        results.sort(key=lambda m: m.strength * m.importance, reverse=True)
        return results[:limit]

    async def recall_recent(
        self,
        hours: float = 24.0,
        limit: int = 50,
    ) -> List[Memory]:
        """Ruft kürzlich erstellte/zugegriffene Erinnerungen ab."""
        cutoff = now_utc() - timedelta(hours=hours)

        results = [
            m for m in self._memories.values()
            if m.last_accessed >= cutoff or m.created_at >= cutoff
        ]

        results.sort(key=lambda m: m.last_accessed, reverse=True)
        return results[:limit]

    # ========================================================================
    # PUBLIC API - FORGET
    # ========================================================================

    async def forget(
        self,
        memory_id: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> int:
        """
        Vergisst Erinnerungen.

        Args:
            memory_id: Spezifische Erinnerung vergessen
            threshold: Alle unter Schwellwert vergessen

        Returns:
            Anzahl vergessener Erinnerungen
        """
        forgotten = 0

        if memory_id:
            if memory_id in self._memories:
                self._index.remove(memory_id)
                del self._memories[memory_id]
                self._delete_memory(memory_id)
                forgotten = 1

        elif threshold is not None:
            to_forget = [
                mid for mid, m in self._memories.items()
                if m.strength < threshold
            ]

            for mid in to_forget:
                self._index.remove(mid)
                del self._memories[mid]
                self._delete_memory(mid)
                forgotten += 1

        logger.info(f"Forgotten {forgotten} memories")
        return forgotten

    # ========================================================================
    # CONSOLIDATION & DECAY
    # ========================================================================

    async def consolidate(self) -> Dict[str, int]:
        """
        Konsolidiert Erinnerungen.

        Bewegt Erinnerungen von Working -> Short-term -> Long-term
        basierend auf Wichtigkeit und Häufigkeit des Abrufs.
        """
        stats = {"promoted": 0, "maintained": 0}

        for memory in self._memories.values():
            old_level = memory.consolidation_level

            # Calculate new consolidation level
            consolidation_score = (
                memory.importance * 0.3 +
                memory.strength * 0.3 +
                min(memory.access_count / 10, 0.4)
            )

            if consolidation_score >= self.config.long_term_threshold:
                memory.consolidation_level = 2  # Long-term
            elif consolidation_score >= self.config.short_term_threshold:
                memory.consolidation_level = 1  # Short-term
            else:
                memory.consolidation_level = 0  # Working

            if memory.consolidation_level > old_level:
                stats["promoted"] += 1
                # Boost strength for promoted memories
                memory.strength = min(1.0, memory.strength + 0.1)
            else:
                stats["maintained"] += 1

            self._save_memory(memory)

        logger.info("Memory consolidation complete", **stats)
        return stats

    async def apply_decay(self) -> Dict[str, int]:
        """
        Wendet Vergessenskurve auf alle Erinnerungen an.
        """
        if not self.config.decay_enabled:
            return {"decayed": 0, "forgotten": 0}

        stats = {"decayed": 0, "forgotten": 0}

        for memory in list(self._memories.values()):
            hours_since_access = (now_utc() - memory.last_accessed).total_seconds() / 3600
            old_strength = memory.strength

            memory.decay(hours_since_access)
            stats["decayed"] += 1

            # Remove if too weak
            if memory.strength < self.config.forget_threshold:
                await self.forget(memory_id=memory.id)
                stats["forgotten"] += 1
            else:
                self._save_memory(memory)

        logger.info("Memory decay applied", **stats)
        return stats

    # ========================================================================
    # BACKGROUND TASKS
    # ========================================================================

    async def start_background_tasks(self) -> None:
        """Startet Background-Tasks für Konsolidierung und Verfall."""
        if self._running:
            return

        self._running = True

        async def consolidation_loop():
            while self._running:
                await asyncio.sleep(self.config.consolidation_interval_hours * 3600)
                try:
                    await self.consolidate()
                except Exception as e:
                    logger.error(f"Consolidation failed: {e}")

        async def decay_loop():
            while self._running:
                await asyncio.sleep(self.config.decay_interval_hours * 3600)
                try:
                    await self.apply_decay()
                except Exception as e:
                    logger.error(f"Decay failed: {e}")

        self._consolidation_task = asyncio.create_task(consolidation_loop())
        self._decay_task = asyncio.create_task(decay_loop())

        logger.info("Memory background tasks started")

    async def stop_background_tasks(self) -> None:
        """Stoppt Background-Tasks."""
        self._running = False

        if self._consolidation_task:
            self._consolidation_task.cancel()
        if self._decay_task:
            self._decay_task.cancel()

        logger.info("Memory background tasks stopped")

    # ========================================================================
    # UTILITY
    # ========================================================================

    def set_embedding_callback(self, callback: Callable) -> None:
        """Setzt den Callback für Embedding-Generierung."""
        self._embedding_callback = callback

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über das Gedächtnis zurück."""
        type_counts = defaultdict(int)
        level_counts = defaultdict(int)
        total_strength = 0.0

        for m in self._memories.values():
            type_counts[m.memory_type.value] += 1
            level_counts[m.consolidation_level] += 1
            total_strength += m.strength

        return {
            "total_memories": len(self._memories),
            "working_memory_size": len(self._working_memory),
            "by_type": dict(type_counts),
            "by_consolidation_level": {
                "working": level_counts[0],
                "short_term": level_counts[1],
                "long_term": level_counts[2],
            },
            "average_strength": total_strength / max(len(self._memories), 1),
            "db_path": str(self.config.db_path),
        }

    def get_working_memory(self) -> List[Memory]:
        """Gibt das aktuelle Arbeitsgedächtnis zurück."""
        return [
            self._memories[mid]
            for mid in self._working_memory
            if mid in self._memories
        ]

    async def create_association(self, memory_id1: str, memory_id2: str) -> bool:
        """Erstellt eine Assoziation zwischen zwei Erinnerungen."""
        if memory_id1 not in self._memories or memory_id2 not in self._memories:
            return False

        m1 = self._memories[memory_id1]
        m2 = self._memories[memory_id2]

        if memory_id2 not in m1.associations:
            m1.associations.append(memory_id2)
        if memory_id1 not in m2.associations:
            m2.associations.append(memory_id1)

        self._index.add(m1)
        self._index.add(m2)
        self._save_memory(m1)
        self._save_memory(m2)

        return True

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Memory]:
        """Convenience-Methode für einfache Suche."""
        return await self.recall(query, limit=limit)

    def clear(self) -> None:
        """Löscht alle Erinnerungen."""
        self._memories.clear()
        self._working_memory.clear()
        self._index = MemoryIndex()

        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute("DELETE FROM memories")

        logger.info("Memory cleared")


# ============================================================================
# SINGLETON & CONVENIENCE
# ============================================================================

_default_memory: Optional[PersistentMemory] = None


def get_memory(config: Optional[MemoryConfig] = None) -> PersistentMemory:
    """Gibt eine Singleton-Instanz des Gedächtnisses zurück."""
    global _default_memory
    if _default_memory is None:
        _default_memory = PersistentMemory(config)
    return _default_memory


async def remember(
    content: str,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    importance: float = 0.5,
    **kwargs,
) -> Memory:
    """Convenience-Funktion zum Speichern."""
    return await get_memory().remember(content, memory_type, importance, **kwargs)


async def recall(query: str, limit: int = 10, **kwargs) -> List[Memory]:
    """Convenience-Funktion zum Abrufen."""
    return await get_memory().recall(query, limit, **kwargs)


async def forget(memory_id: Optional[str] = None, threshold: Optional[float] = None) -> int:
    """Convenience-Funktion zum Vergessen."""
    return await get_memory().forget(memory_id, threshold)
