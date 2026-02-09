"""
SCIO Knowledge Graph

Graph-basierte Wissensrepräsentation mit Entitäten und Relationen.
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Iterator, Callable
import heapq

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class EntityType(str, Enum):
    """Typen von Entitäten im Knowledge Graph."""
    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    OBJECT = "object"
    PROCESS = "process"
    ATTRIBUTE = "attribute"
    VALUE = "value"
    CODE = "code"
    DOCUMENT = "document"
    AGENT = "agent"
    TOOL = "tool"
    EXPERIMENT = "experiment"


class RelationType(str, Enum):
    """Typen von Relationen."""
    # Hierarchische Beziehungen
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    SUBCLASS_OF = "subclass_of"
    INSTANCE_OF = "instance_of"

    # Kausale Beziehungen
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    ENABLES = "enables"
    PREVENTS = "prevents"
    REQUIRES = "requires"

    # Temporale Beziehungen
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    STARTS = "starts"
    ENDS = "ends"

    # Assoziative Beziehungen
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    DEPENDS_ON = "depends_on"
    USES = "uses"
    USED_BY = "used_by"
    CREATED_BY = "created_by"
    CREATES = "creates"

    # Attributive Beziehungen
    HAS_PROPERTY = "has_property"
    HAS_VALUE = "has_value"
    HAS_TYPE = "has_type"

    # Räumliche Beziehungen
    LOCATED_IN = "located_in"
    CONTAINS = "contains"
    NEAR = "near"


@dataclass
class Entity:
    """Eine Entität im Knowledge Graph."""

    id: str
    name: str
    entity_type: EntityType
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    confidence: float = 1.0
    source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
        )


@dataclass
class Relation:
    """Eine Relation zwischen zwei Entitäten."""

    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            bidirectional=data.get("bidirectional", False),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class Triple:
    """Ein Wissens-Triple (Subject, Predicate, Object)."""

    subject: Entity
    predicate: Relation
    object: Entity

    def to_string(self) -> str:
        return f"({self.subject.name}) --[{self.predicate.relation_type.value}]--> ({self.object.name})"


@dataclass
class GraphPath:
    """Ein Pfad durch den Graphen."""

    entities: list[Entity]
    relations: list[Relation]
    total_weight: float = 0.0

    def __len__(self) -> int:
        return len(self.relations)

    def to_string(self) -> str:
        if not self.entities:
            return ""
        parts = [self.entities[0].name]
        for rel, ent in zip(self.relations, self.entities[1:]):
            parts.append(f" --[{rel.relation_type.value}]--> {ent.name}")
        return "".join(parts)


class KnowledgeGraph:
    """
    Leistungsstarker Knowledge Graph für SCIO.

    Features:
    - Entitäten mit Typen und Eigenschaften
    - Typisierte Relationen
    - Graphtraversierung und Pfadsuche
    - Subgraph-Extraktion
    - Pattern-Matching
    - Inferenz über Relationen
    - Persistente Speicherung
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".scio" / "knowledge_graph.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-Memory Strukturen
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}

        # Indexe für schnellen Zugriff
        self._outgoing: dict[str, list[str]] = defaultdict(list)  # entity_id -> [relation_ids]
        self._incoming: dict[str, list[str]] = defaultdict(list)  # entity_id -> [relation_ids]
        self._by_type: dict[EntityType, set[str]] = defaultdict(set)
        self._by_name: dict[str, set[str]] = defaultdict(set)

        self._init_db()
        self._load_from_db()

        logger.info(
            "KnowledgeGraph initialized",
            entities=len(self._entities),
            relations=len(self._relations),
        )

    def _init_db(self) -> None:
        """Initialisiert die SQLite-Datenbank."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    properties TEXT,
                    embedding BLOB,
                    created_at TEXT,
                    updated_at TEXT,
                    confidence REAL DEFAULT 1.0,
                    source TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT,
                    weight REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    bidirectional INTEGER DEFAULT 0,
                    created_at TEXT,
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_source ON relations(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_target ON relations(target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relation_type)")

    def _load_from_db(self) -> None:
        """Lädt den Graph aus der Datenbank."""
        with sqlite3.connect(self.db_path) as conn:
            # Lade Entitäten
            for row in conn.execute("SELECT * FROM entities"):
                entity = Entity(
                    id=row[0],
                    name=row[1],
                    entity_type=EntityType(row[2]),
                    properties=json.loads(row[3]) if row[3] else {},
                    created_at=datetime.fromisoformat(row[5]) if row[5] else now_utc(),
                    updated_at=datetime.fromisoformat(row[6]) if row[6] else now_utc(),
                    confidence=row[7] or 1.0,
                    source=row[8],
                )
                self._entities[entity.id] = entity
                self._by_type[entity.entity_type].add(entity.id)
                self._by_name[entity.name.lower()].add(entity.id)

            # Lade Relationen
            for row in conn.execute("SELECT * FROM relations"):
                relation = Relation(
                    id=row[0],
                    source_id=row[1],
                    target_id=row[2],
                    relation_type=RelationType(row[3]),
                    properties=json.loads(row[4]) if row[4] else {},
                    weight=row[5] or 1.0,
                    confidence=row[6] or 1.0,
                    bidirectional=bool(row[7]),
                    created_at=datetime.fromisoformat(row[8]) if row[8] else now_utc(),
                )
                self._relations[relation.id] = relation
                self._outgoing[relation.source_id].append(relation.id)
                self._incoming[relation.target_id].append(relation.id)
                if relation.bidirectional:
                    self._outgoing[relation.target_id].append(relation.id)
                    self._incoming[relation.source_id].append(relation.id)

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        properties: Optional[dict[str, Any]] = None,
        confidence: float = 1.0,
        source: Optional[str] = None,
    ) -> Entity:
        """Fügt eine neue Entität hinzu."""
        entity_id = generate_id("ent")
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            confidence=confidence,
            source=source,
        )

        self._entities[entity_id] = entity
        self._by_type[entity_type].add(entity_id)
        self._by_name[name.lower()].add(entity_id)
        self._save_entity(entity)

        logger.debug("Entity added", id=entity_id, name=name, type=entity_type.value)
        return entity

    def _save_entity(self, entity: Entity) -> None:
        """Speichert eine Entität."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO entities
                (id, name, entity_type, properties, created_at, updated_at, confidence, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id,
                entity.name,
                entity.entity_type.value,
                json.dumps(entity.properties),
                entity.created_at.isoformat(),
                entity.updated_at.isoformat(),
                entity.confidence,
                entity.source,
            ))

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Optional[dict[str, Any]] = None,
        weight: float = 1.0,
        confidence: float = 1.0,
        bidirectional: bool = False,
    ) -> Optional[Relation]:
        """Fügt eine neue Relation hinzu."""
        if source_id not in self._entities or target_id not in self._entities:
            logger.warning("Cannot create relation: entity not found")
            return None

        relation_id = generate_id("rel")
        relation = Relation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight,
            confidence=confidence,
            bidirectional=bidirectional,
        )

        self._relations[relation_id] = relation
        self._outgoing[source_id].append(relation_id)
        self._incoming[target_id].append(relation_id)
        if bidirectional:
            self._outgoing[target_id].append(relation_id)
            self._incoming[source_id].append(relation_id)

        self._save_relation(relation)

        logger.debug(
            "Relation added",
            id=relation_id,
            type=relation_type.value,
            source=source_id,
            target=target_id,
        )
        return relation

    def _save_relation(self, relation: Relation) -> None:
        """Speichert eine Relation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO relations
                (id, source_id, target_id, relation_type, properties, weight, confidence, bidirectional, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relation.id,
                relation.source_id,
                relation.target_id,
                relation.relation_type.value,
                json.dumps(relation.properties),
                relation.weight,
                relation.confidence,
                int(relation.bidirectional),
                relation.created_at.isoformat(),
            ))

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Holt eine Entität nach ID."""
        return self._entities.get(entity_id)

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Holt eine Relation nach ID."""
        return self._relations.get(relation_id)

    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        properties: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Entity]:
        """Sucht Entitäten nach Kriterien."""
        candidates = set(self._entities.keys())

        if name:
            name_lower = name.lower()
            name_matches = self._by_name.get(name_lower, set())
            # Auch partielle Matches
            for n, ids in self._by_name.items():
                if name_lower in n:
                    name_matches = name_matches | ids
            candidates &= name_matches

        if entity_type:
            candidates &= self._by_type.get(entity_type, set())

        results = []
        for eid in candidates:
            entity = self._entities[eid]

            if properties:
                if not all(entity.properties.get(k) == v for k, v in properties.items()):
                    continue

            results.append(entity)
            if len(results) >= limit:
                break

        return results

    def get_neighbors(
        self,
        entity_id: str,
        relation_types: Optional[list[RelationType]] = None,
        direction: str = "both",  # "out", "in", "both"
    ) -> list[tuple[Entity, Relation]]:
        """Holt alle Nachbarn einer Entität."""
        neighbors = []

        if direction in ("out", "both"):
            for rel_id in self._outgoing.get(entity_id, []):
                rel = self._relations[rel_id]
                if relation_types and rel.relation_type not in relation_types:
                    continue
                target_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                target = self._entities.get(target_id)
                if target:
                    neighbors.append((target, rel))

        if direction in ("in", "both"):
            for rel_id in self._incoming.get(entity_id, []):
                rel = self._relations[rel_id]
                if relation_types and rel.relation_type not in relation_types:
                    continue
                if rel.source_id != entity_id:  # Vermeide Duplikate
                    source = self._entities.get(rel.source_id)
                    if source:
                        neighbors.append((source, rel))

        return neighbors

    def get_triples(
        self,
        subject_id: Optional[str] = None,
        predicate_type: Optional[RelationType] = None,
        object_id: Optional[str] = None,
    ) -> list[Triple]:
        """Holt Triples nach Pattern."""
        triples = []

        for rel in self._relations.values():
            if subject_id and rel.source_id != subject_id:
                continue
            if predicate_type and rel.relation_type != predicate_type:
                continue
            if object_id and rel.target_id != object_id:
                continue

            subject = self._entities.get(rel.source_id)
            obj = self._entities.get(rel.target_id)
            if subject and obj:
                triples.append(Triple(subject=subject, predicate=rel, object=obj))

        return triples

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relation_types: Optional[list[RelationType]] = None,
    ) -> Optional[GraphPath]:
        """Findet den kürzesten Pfad zwischen zwei Entitäten (Dijkstra)."""
        if start_id not in self._entities or end_id not in self._entities:
            return None

        # Priority Queue: (distance, entity_id, path)
        heap = [(0.0, start_id, [], [])]
        visited = set()

        while heap:
            dist, current_id, entities_path, relations_path = heapq.heappop(heap)

            if current_id in visited:
                continue
            visited.add(current_id)

            current_entity = self._entities[current_id]
            new_entities_path = entities_path + [current_entity]

            if current_id == end_id:
                return GraphPath(
                    entities=new_entities_path,
                    relations=relations_path,
                    total_weight=dist,
                )

            if len(new_entities_path) > max_depth:
                continue

            for neighbor, rel in self.get_neighbors(current_id, relation_types):
                if neighbor.id not in visited:
                    new_dist = dist + (1.0 / rel.weight)
                    heapq.heappush(
                        heap,
                        (new_dist, neighbor.id, new_entities_path, relations_path + [rel]),
                    )

        return None

    def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 4,
        max_paths: int = 10,
    ) -> list[GraphPath]:
        """Findet alle Pfade zwischen zwei Entitäten."""
        paths = []

        def dfs(current_id: str, target_id: str, path_entities: list, path_relations: list, visited: set) -> None:
            if len(paths) >= max_paths:
                return

            if current_id == target_id:
                paths.append(GraphPath(
                    entities=path_entities.copy(),
                    relations=path_relations.copy(),
                    total_weight=sum(1.0 / r.weight for r in path_relations),
                ))
                return

            if len(path_entities) > max_depth:
                return

            for neighbor, rel in self.get_neighbors(current_id):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    path_entities.append(neighbor)
                    path_relations.append(rel)
                    dfs(neighbor.id, target_id, path_entities, path_relations, visited)
                    path_entities.pop()
                    path_relations.pop()
                    visited.remove(neighbor.id)

        start_entity = self._entities.get(start_id)
        if start_entity:
            dfs(start_id, end_id, [start_entity], [], {start_id})

        return paths

    def get_subgraph(
        self,
        center_id: str,
        depth: int = 2,
        relation_types: Optional[list[RelationType]] = None,
    ) -> tuple[list[Entity], list[Relation]]:
        """Extrahiert einen Subgraphen um eine Entität."""
        entities = set()
        relations = set()
        to_visit = [(center_id, 0)]
        visited = set()

        while to_visit:
            current_id, current_depth = to_visit.pop(0)
            if current_id in visited or current_depth > depth:
                continue

            visited.add(current_id)
            entity = self._entities.get(current_id)
            if entity:
                entities.add(entity.id)

            if current_depth < depth:
                for neighbor, rel in self.get_neighbors(current_id, relation_types):
                    relations.add(rel.id)
                    if neighbor.id not in visited:
                        to_visit.append((neighbor.id, current_depth + 1))

        return (
            [self._entities[eid] for eid in entities],
            [self._relations[rid] for rid in relations],
        )

    def infer_relations(
        self,
        entity_id: str,
        inference_rules: Optional[dict[tuple[RelationType, RelationType], RelationType]] = None,
    ) -> list[Relation]:
        """
        Inferiert neue Relationen basierend auf Transitivitätsregeln.

        Beispiel: Wenn A is_a B und B is_a C, dann A is_a C
        """
        if not inference_rules:
            inference_rules = {
                (RelationType.IS_A, RelationType.IS_A): RelationType.IS_A,
                (RelationType.PART_OF, RelationType.PART_OF): RelationType.PART_OF,
                (RelationType.CAUSES, RelationType.CAUSES): RelationType.CAUSES,
                (RelationType.REQUIRES, RelationType.REQUIRES): RelationType.REQUIRES,
            }

        inferred = []
        entity = self._entities.get(entity_id)
        if not entity:
            return inferred

        # Hole alle ausgehenden Relationen
        for rel1_id in self._outgoing.get(entity_id, []):
            rel1 = self._relations[rel1_id]
            intermediate_id = rel1.target_id if rel1.source_id == entity_id else rel1.source_id

            # Hole Relationen vom Zwischenknoten
            for rel2_id in self._outgoing.get(intermediate_id, []):
                rel2 = self._relations[rel2_id]
                final_id = rel2.target_id if rel2.source_id == intermediate_id else rel2.source_id

                # Prüfe Inferenzregel
                key = (rel1.relation_type, rel2.relation_type)
                if key in inference_rules and final_id != entity_id:
                    inferred_type = inference_rules[key]
                    # Prüfe ob Relation bereits existiert
                    exists = any(
                        r.source_id == entity_id and r.target_id == final_id and r.relation_type == inferred_type
                        for r in self._relations.values()
                    )
                    if not exists:
                        new_rel = self.add_relation(
                            entity_id,
                            final_id,
                            inferred_type,
                            properties={"inferred": True, "via": intermediate_id},
                            confidence=min(rel1.confidence, rel2.confidence) * 0.9,
                        )
                        if new_rel:
                            inferred.append(new_rel)

        return inferred

    def merge_entities(self, entity_ids: list[str], new_name: Optional[str] = None) -> Optional[Entity]:
        """Führt mehrere Entitäten zu einer zusammen."""
        if len(entity_ids) < 2:
            return None

        entities = [self._entities.get(eid) for eid in entity_ids]
        entities = [e for e in entities if e is not None]
        if len(entities) < 2:
            return None

        # Erstelle neue Entität
        primary = entities[0]
        merged = self.add_entity(
            name=new_name or primary.name,
            entity_type=primary.entity_type,
            properties={k: v for e in entities for k, v in e.properties.items()},
            confidence=sum(e.confidence for e in entities) / len(entities),
        )

        # Übertrage alle Relationen
        for entity in entities:
            for rel_id in self._outgoing.get(entity.id, []):
                rel = self._relations[rel_id]
                target_id = rel.target_id if rel.target_id not in entity_ids else merged.id
                self.add_relation(merged.id, target_id, rel.relation_type, rel.properties)

            for rel_id in self._incoming.get(entity.id, []):
                rel = self._relations[rel_id]
                source_id = rel.source_id if rel.source_id not in entity_ids else merged.id
                self.add_relation(source_id, merged.id, rel.relation_type, rel.properties)

            # Lösche alte Entität
            self.delete_entity(entity.id)

        return merged

    def delete_entity(self, entity_id: str) -> bool:
        """Löscht eine Entität und alle zugehörigen Relationen."""
        entity = self._entities.get(entity_id)
        if not entity:
            return False

        # Lösche Relationen
        rel_ids_to_delete = (
            self._outgoing.get(entity_id, []) +
            self._incoming.get(entity_id, [])
        )
        for rel_id in set(rel_ids_to_delete):
            self.delete_relation(rel_id)

        # Lösche Entität
        del self._entities[entity_id]
        self._by_type[entity.entity_type].discard(entity_id)
        self._by_name[entity.name.lower()].discard(entity_id)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))

        return True

    def delete_relation(self, relation_id: str) -> bool:
        """Löscht eine Relation."""
        relation = self._relations.get(relation_id)
        if not relation:
            return False

        del self._relations[relation_id]
        self._outgoing[relation.source_id] = [r for r in self._outgoing[relation.source_id] if r != relation_id]
        self._incoming[relation.target_id] = [r for r in self._incoming[relation.target_id] if r != relation_id]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM relations WHERE id = ?", (relation_id,))

        return True

    def stats(self) -> dict[str, Any]:
        """Gibt Statistiken über den Graphen zurück."""
        type_counts = defaultdict(int)
        rel_counts = defaultdict(int)

        for e in self._entities.values():
            type_counts[e.entity_type.value] += 1

        for r in self._relations.values():
            rel_counts[r.relation_type.value] += 1

        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entity_types": dict(type_counts),
            "relation_types": dict(rel_counts),
            "avg_relations_per_entity": len(self._relations) / max(len(self._entities), 1),
        }

    def export_to_json(self, path: Path) -> None:
        """Exportiert den Graph als JSON."""
        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations.values()],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def clear(self) -> None:
        """Löscht den gesamten Graphen."""
        self._entities.clear()
        self._relations.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._by_type.clear()
        self._by_name.clear()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM entities")
            conn.execute("DELETE FROM relations")

        logger.info("KnowledgeGraph cleared")
