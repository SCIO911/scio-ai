#!/usr/bin/env python3
"""
SCIO - Knowledge Graph
Entitäten, Relationen und semantische Inferenz
"""

import json
import logging
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Eine Entität im Wissensgraphen"""
    id: str
    type: str  # z.B. "model", "worker", "capability", "user"
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Relation:
    """Eine Beziehung zwischen Entitäten"""
    id: str
    type: str  # z.B. "has_capability", "requires", "produces"
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.id)


class KnowledgeGraph:
    """
    SCIO Knowledge Graph

    Speichert und verwaltet:
    - Entitäten (Modelle, Worker, Fähigkeiten, etc.)
    - Relationen zwischen Entitäten
    - Ontologie (Hierarchie von Typen)
    - Inferenzregeln
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}

        # Indizes für schnelle Suche
        self.entities_by_type: Dict[str, Set[str]] = defaultdict(set)
        self.outgoing_relations: Dict[str, Set[str]] = defaultdict(set)
        self.incoming_relations: Dict[str, Set[str]] = defaultdict(set)
        self.relations_by_type: Dict[str, Set[str]] = defaultdict(set)

        # Ontologie: type -> parent_types
        self.type_hierarchy: Dict[str, Set[str]] = defaultdict(set)

        # Inferenzregeln
        self.inference_rules: List[Dict] = []

        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert den Knowledge Graph"""
        try:
            self._setup_base_ontology()
            self._setup_inference_rules()
            self._load_graph()
            self._initialized = True
            logger.info("Knowledge Graph initialisiert")
            return True
        except Exception as e:
            logger.error(f"Knowledge Graph Fehler: {e}")
            return False

    def _setup_base_ontology(self):
        """Erstellt Basis-Ontologie"""
        # AI-Komponenten Hierarchie
        self.type_hierarchy["llm_model"] = {"model", "ai_component"}
        self.type_hierarchy["image_model"] = {"model", "ai_component"}
        self.type_hierarchy["audio_model"] = {"model", "ai_component"}
        self.type_hierarchy["embedding_model"] = {"model", "ai_component"}

        self.type_hierarchy["worker"] = {"service", "ai_component"}
        self.type_hierarchy["llm_worker"] = {"worker"}
        self.type_hierarchy["vision_worker"] = {"worker"}
        self.type_hierarchy["audio_worker"] = {"worker"}

        self.type_hierarchy["capability"] = {"concept"}
        self.type_hierarchy["text_generation"] = {"capability"}
        self.type_hierarchy["image_generation"] = {"capability"}
        self.type_hierarchy["speech_recognition"] = {"capability"}

        self.type_hierarchy["resource"] = {"concept"}
        self.type_hierarchy["gpu"] = {"resource", "hardware"}
        self.type_hierarchy["memory"] = {"resource", "hardware"}

    def _setup_inference_rules(self):
        """Erstellt Inferenzregeln"""
        # Wenn A has_capability C und C requires R, dann A requires R
        self.inference_rules.append({
            "name": "capability_requires_propagation",
            "pattern": [
                ("A", "has_capability", "C"),
                ("C", "requires", "R")
            ],
            "infer": ("A", "requires", "R")
        })

        # Wenn A is_a B und B has_property P, dann A has_property P
        self.inference_rules.append({
            "name": "property_inheritance",
            "pattern": [
                ("A", "is_a", "B"),
                ("B", "has_property", "P")
            ],
            "infer": ("A", "has_property", "P")
        })

    def _load_graph(self):
        """Lädt gespeicherten Graph"""
        try:
            from backend.config import Config
            graph_file = Config.DATA_DIR / "knowledge_graph.json"
            if graph_file.exists():
                with open(graph_file, 'r') as f:
                    data = json.load(f)

                for e in data.get("entities", []):
                    entity = Entity(
                        id=e["id"],
                        type=e["type"],
                        name=e["name"],
                        properties=e.get("properties", {})
                    )
                    self.add_entity(entity)

                for r in data.get("relations", []):
                    relation = Relation(
                        id=r["id"],
                        type=r["type"],
                        source_id=r["source_id"],
                        target_id=r["target_id"],
                        properties=r.get("properties", {}),
                        weight=r.get("weight", 1.0)
                    )
                    self.add_relation(relation)

                logger.info(f"Graph geladen: {len(self.entities)} Entitäten, {len(self.relations)} Relationen")
        except Exception:
            pass

    def save_graph(self):
        """Speichert den Graph"""
        try:
            from backend.config import Config
            graph_file = Config.DATA_DIR / "knowledge_graph.json"

            data = {
                "entities": [
                    {
                        "id": e.id,
                        "type": e.type,
                        "name": e.name,
                        "properties": e.properties
                    }
                    for e in self.entities.values()
                ],
                "relations": [
                    {
                        "id": r.id,
                        "type": r.type,
                        "source_id": r.source_id,
                        "target_id": r.target_id,
                        "properties": r.properties,
                        "weight": r.weight
                    }
                    for r in self.relations.values()
                ],
                "timestamp": datetime.now().isoformat()
            }

            with open(graph_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Graph speichern fehlgeschlagen: {e}")

    def add_entity(self, entity: Entity) -> bool:
        """Fügt eine Entität hinzu"""
        if entity.id in self.entities:
            return False

        self.entities[entity.id] = entity
        self.entities_by_type[entity.type].add(entity.id)

        # Auch Parent-Typen registrieren
        for parent_type in self._get_all_parent_types(entity.type):
            self.entities_by_type[parent_type].add(entity.id)

        return True

    def add_relation(self, relation: Relation) -> bool:
        """Fügt eine Relation hinzu"""
        if relation.source_id not in self.entities:
            return False
        if relation.target_id not in self.entities:
            return False

        self.relations[relation.id] = relation
        self.outgoing_relations[relation.source_id].add(relation.id)
        self.incoming_relations[relation.target_id].add(relation.id)
        self.relations_by_type[relation.type].add(relation.id)

        return True

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Gibt eine Entität zurück"""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Gibt alle Entitäten eines Typs zurück"""
        entity_ids = self.entities_by_type.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_related(self,
                    entity_id: str,
                    relation_type: str = None,
                    direction: str = "outgoing") -> List[Tuple[Relation, Entity]]:
        """
        Gibt verwandte Entitäten zurück

        Args:
            entity_id: Ausgangs-Entität
            relation_type: Optional: Nur bestimmte Relationstypen
            direction: "outgoing", "incoming" oder "both"

        Returns:
            Liste von (Relation, Entität) Tupeln
        """
        results = []

        if direction in ["outgoing", "both"]:
            for rel_id in self.outgoing_relations.get(entity_id, set()):
                rel = self.relations.get(rel_id)
                if rel and (relation_type is None or rel.type == relation_type):
                    target = self.entities.get(rel.target_id)
                    if target:
                        results.append((rel, target))

        if direction in ["incoming", "both"]:
            for rel_id in self.incoming_relations.get(entity_id, set()):
                rel = self.relations.get(rel_id)
                if rel and (relation_type is None or rel.type == relation_type):
                    source = self.entities.get(rel.source_id)
                    if source:
                        results.append((rel, source))

        return results

    def query(self,
              entity_type: str = None,
              relation_type: str = None,
              properties: Dict[str, Any] = None) -> List[Entity]:
        """
        Flexible Abfrage des Graphs

        Args:
            entity_type: Filtern nach Typ
            relation_type: Nur Entitäten mit bestimmten Relationen
            properties: Filtern nach Properties

        Returns:
            Passende Entitäten
        """
        # Start mit allen oder typ-gefilterten Entitäten
        if entity_type:
            candidates = self.get_entities_by_type(entity_type)
        else:
            candidates = list(self.entities.values())

        results = []
        for entity in candidates:
            # Property-Filter
            if properties:
                match = all(
                    entity.properties.get(k) == v
                    for k, v in properties.items()
                )
                if not match:
                    continue

            # Relation-Filter
            if relation_type:
                has_relation = bool(self.get_related(entity.id, relation_type))
                if not has_relation:
                    continue

            results.append(entity)

        return results

    def find_path(self,
                  source_id: str,
                  target_id: str,
                  max_depth: int = 5) -> Optional[List[Tuple[str, Relation]]]:
        """
        Findet einen Pfad zwischen zwei Entitäten

        Returns:
            Liste von (entity_id, relation) Tupeln oder None
        """
        if source_id == target_id:
            return []

        visited = set()
        queue = [(source_id, [])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            if current_id in visited:
                continue
            visited.add(current_id)

            for rel_id in self.outgoing_relations.get(current_id, set()):
                rel = self.relations.get(rel_id)
                if not rel:
                    continue

                new_path = path + [(current_id, rel)]

                if rel.target_id == target_id:
                    return new_path + [(target_id, None)]

                queue.append((rel.target_id, new_path))

        return None

    def infer(self) -> List[Relation]:
        """
        Wendet Inferenzregeln an und erzeugt neue Relationen

        Returns:
            Liste neu inferierter Relationen
        """
        new_relations = []

        for rule in self.inference_rules:
            # Einfache Pattern-Matching Implementierung
            pattern = rule["pattern"]
            infer = rule["infer"]

            # Sammle alle Matches für das Pattern
            # (Vereinfachte Version - vollständige Implementierung wäre komplexer)

            if len(pattern) == 2:
                # Zwei-Schritt Pattern
                p1_source, p1_rel, p1_target = pattern[0]
                p2_source, p2_rel, p2_target = pattern[1]

                # Finde erste Relationen
                for rel1 in self.relations_by_type.get(p1_rel, set()):
                    r1 = self.relations.get(rel1)
                    if not r1:
                        continue

                    # Finde zweite Relationen die vom Ziel der ersten starten
                    for rel2 in self.outgoing_relations.get(r1.target_id, set()):
                        r2 = self.relations.get(rel2)
                        if not r2 or r2.type != p2_rel:
                            continue

                        # Inferenz erzeugen
                        infer_source = r1.source_id if infer[0] == "A" else r2.source_id
                        infer_target = r2.target_id if infer[2] == "R" else r1.target_id

                        # Prüfen ob Relation schon existiert
                        exists = any(
                            r.source_id == infer_source and
                            r.target_id == infer_target and
                            r.type == infer[1]
                            for r in self.relations.values()
                        )

                        if not exists:
                            new_rel = Relation(
                                id=f"inferred_{len(self.relations)}_{len(new_relations)}",
                                type=infer[1],
                                source_id=infer_source,
                                target_id=infer_target,
                                properties={"inferred": True, "rule": rule["name"]}
                            )
                            if self.add_relation(new_rel):
                                new_relations.append(new_rel)

        return new_relations

    def _get_all_parent_types(self, entity_type: str) -> Set[str]:
        """Gibt alle Parent-Typen zurück (transitiv)"""
        parents = set()
        to_process = [entity_type]

        while to_process:
            current = to_process.pop()
            direct_parents = self.type_hierarchy.get(current, set())
            for parent in direct_parents:
                if parent not in parents:
                    parents.add(parent)
                    to_process.append(parent)

        return parents

    def is_type(self, entity_id: str, entity_type: str) -> bool:
        """Prüft ob Entität einen bestimmten Typ hat (inkl. Vererbung)"""
        entity = self.entities.get(entity_id)
        if not entity:
            return False

        if entity.type == entity_type:
            return True

        return entity_type in self._get_all_parent_types(entity.type)

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        return {
            "entities_count": len(self.entities),
            "relations_count": len(self.relations),
            "entity_types": {t: len(ids) for t, ids in self.entities_by_type.items()},
            "relation_types": {t: len(ids) for t, ids in self.relations_by_type.items()},
            "type_hierarchy_depth": len(self.type_hierarchy),
            "inference_rules": len(self.inference_rules)
        }


# Singleton
_knowledge_graph: Optional[KnowledgeGraph] = None

def get_knowledge_graph() -> KnowledgeGraph:
    """Gibt Singleton-Instanz zurück"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph
