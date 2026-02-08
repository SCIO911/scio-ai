#!/usr/bin/env python3
"""
SCIO - Decision Engine
Entscheidungsbäume, Heuristiken und intelligente Entscheidungsfindung
"""

import json
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Typen von Entscheidungen"""
    DETERMINISTIC = 'deterministic'  # Regelbasiert
    PROBABILISTIC = 'probabilistic'  # Wahrscheinlichkeitsbasiert
    HEURISTIC = 'heuristic'          # Heuristik-basiert
    LEARNED = 'learned'              # Durch Erfahrung gelernt


@dataclass
class DecisionNode:
    """Knoten im Entscheidungsbaum"""
    id: str
    condition: str  # Python-Ausdruck oder Funktion
    true_branch: Optional[str] = None  # ID des nächsten Knotens
    false_branch: Optional[str] = None
    action: Optional[str] = None  # Aktion wenn Blattknoten
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Ergebnis einer Entscheidung"""
    action: str
    confidence: float
    reasoning: List[str]
    context: Dict[str, Any]
    decision_type: DecisionType
    processing_time_ms: float


class DecisionTree:
    """
    Entscheidungsbaum für strukturierte Entscheidungsfindung
    """

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, DecisionNode] = {}
        self.root_id: Optional[str] = None

    def add_node(self, node: DecisionNode, is_root: bool = False):
        """Fügt einen Knoten hinzu"""
        self.nodes[node.id] = node
        if is_root:
            self.root_id = node.id

    def evaluate(self, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Evaluiert den Baum und gibt Aktion + Reasoning zurück
        """
        if not self.root_id:
            return "no_action", ["No root node defined"]

        reasoning = []
        current_id = self.root_id

        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break

            # Blattknoten - Aktion ausführen
            if node.action:
                reasoning.append(f"→ Action: {node.action}")
                return node.action, reasoning

            # Bedingung evaluieren
            try:
                result = eval(node.condition, {"__builtins__": {}}, context)
                reasoning.append(f"Check: {node.condition} = {result}")

                if result:
                    current_id = node.true_branch
                else:
                    current_id = node.false_branch
            except Exception as e:
                reasoning.append(f"Error evaluating {node.condition}: {e}")
                return "error", reasoning

        return "no_action", reasoning


class Heuristic:
    """
    Heuristik für schnelle Entscheidungen basierend auf Erfahrungswerten
    """

    def __init__(self, name: str, score_func: Callable[[Dict], float]):
        self.name = name
        self.score_func = score_func
        self.usage_count = 0
        self.success_rate = 1.0

    def evaluate(self, context: Dict[str, Any]) -> float:
        """Berechnet Heuristik-Score"""
        self.usage_count += 1
        return self.score_func(context)

    def update_success(self, success: bool):
        """Aktualisiert Erfolgsrate"""
        alpha = 0.1  # Lernrate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)


class DecisionEngine:
    """
    SCIO Decision Engine

    Kombiniert mehrere Entscheidungsmethoden:
    - Entscheidungsbäume für strukturierte Logik
    - Heuristiken für schnelle Bewertungen
    - Regelbasierte Entscheidungen
    - Lernbasierte Entscheidungen aus Erfahrung
    """

    def __init__(self):
        self.trees: Dict[str, DecisionTree] = {}
        self.heuristics: Dict[str, Heuristic] = {}
        self.decision_history: List[Decision] = []
        self._history_lock = threading.Lock()
        self.learned_patterns: Dict[str, Dict] = {}
        self._patterns_lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert die Decision Engine mit Standard-Bäumen"""
        try:
            # Standard-Entscheidungsbaum für Worker-Auswahl
            self._create_worker_selection_tree()

            # Standard-Entscheidungsbaum für Ressourcen-Management
            self._create_resource_management_tree()

            # Standard-Heuristiken
            self._create_default_heuristics()

            self._initialized = True
            logger.info("Decision Engine initialisiert")
            return True
        except Exception as e:
            logger.error(f"Decision Engine Fehler: {e}")
            return False

    def _create_worker_selection_tree(self):
        """Erstellt Entscheidungsbaum für Worker-Auswahl"""
        tree = DecisionTree("worker_selection")

        # Root: Prüfe Task-Typ
        tree.add_node(DecisionNode(
            id="check_task_type",
            condition="task_type in ['llm', 'chat', 'completion']",
            true_branch="check_model_size",
            false_branch="check_vision"
        ), is_root=True)

        # LLM Branch
        tree.add_node(DecisionNode(
            id="check_model_size",
            condition="model_params > 13_000_000_000",
            true_branch="use_large_llm",
            false_branch="use_small_llm"
        ))

        tree.add_node(DecisionNode(
            id="use_large_llm",
            action="llm_inference_large",
            metadata={"vram_required": 24}
        ))

        tree.add_node(DecisionNode(
            id="use_small_llm",
            action="llm_inference_small",
            metadata={"vram_required": 8}
        ))

        # Vision Branch
        tree.add_node(DecisionNode(
            id="check_vision",
            condition="task_type in ['ocr', 'caption', 'detect', 'vqa']",
            true_branch="use_vision",
            false_branch="check_audio"
        ))

        tree.add_node(DecisionNode(
            id="use_vision",
            action="vision_worker",
            metadata={"capabilities": ["ocr", "caption", "detect"]}
        ))

        # Audio Branch
        tree.add_node(DecisionNode(
            id="check_audio",
            condition="task_type in ['stt', 'tts', 'transcribe']",
            true_branch="use_audio",
            false_branch="check_code"
        ))

        tree.add_node(DecisionNode(
            id="use_audio",
            action="audio_worker"
        ))

        # Code Branch
        tree.add_node(DecisionNode(
            id="check_code",
            condition="task_type in ['code', 'generate_code', 'review']",
            true_branch="use_code",
            false_branch="use_default"
        ))

        tree.add_node(DecisionNode(
            id="use_code",
            action="code_worker"
        ))

        tree.add_node(DecisionNode(
            id="use_default",
            action="llm_inference"
        ))

        self.trees["worker_selection"] = tree

    def _create_resource_management_tree(self):
        """Erstellt Entscheidungsbaum für Ressourcen-Management"""
        tree = DecisionTree("resource_management")

        tree.add_node(DecisionNode(
            id="check_vram",
            condition="vram_usage > 0.9",
            true_branch="need_cleanup",
            false_branch="check_queue"
        ), is_root=True)

        tree.add_node(DecisionNode(
            id="need_cleanup",
            condition="active_jobs > 5",
            true_branch="defer_job",
            false_branch="unload_models"
        ))

        tree.add_node(DecisionNode(
            id="defer_job",
            action="defer_to_queue",
            metadata={"priority": "low"}
        ))

        tree.add_node(DecisionNode(
            id="unload_models",
            action="unload_least_used",
            metadata={"keep_count": 2}
        ))

        tree.add_node(DecisionNode(
            id="check_queue",
            condition="queue_size > 10",
            true_branch="scale_workers",
            false_branch="process_normal"
        ))

        tree.add_node(DecisionNode(
            id="scale_workers",
            action="increase_workers",
            metadata={"max_workers": 12}
        ))

        tree.add_node(DecisionNode(
            id="process_normal",
            action="process_immediately"
        ))

        self.trees["resource_management"] = tree

    def _create_default_heuristics(self):
        """Erstellt Standard-Heuristiken"""

        # Prioritäts-Heuristik
        self.heuristics["priority_score"] = Heuristic(
            "priority_score",
            lambda ctx: (
                ctx.get("priority", 0) * 10 +
                (100 - ctx.get("queue_position", 0)) +
                (50 if ctx.get("is_premium", False) else 0) +
                (30 if ctx.get("is_retry", False) else 0)
            )
        )

        # Ressourcen-Kosten-Heuristik
        self.heuristics["resource_cost"] = Heuristic(
            "resource_cost",
            lambda ctx: (
                ctx.get("estimated_vram_gb", 8) * 10 +
                ctx.get("estimated_time_s", 30) * 0.5 +
                ctx.get("token_count", 1000) * 0.01
            )
        )

        # Erfolgswahrscheinlichkeit
        self.heuristics["success_probability"] = Heuristic(
            "success_probability",
            lambda ctx: (
                min(1.0, 0.5 +
                    0.2 * (1 if ctx.get("model_loaded", False) else 0) +
                    0.2 * (1 if ctx.get("vram_available", 0) > ctx.get("vram_required", 8) else 0) +
                    0.1 * min(1, ctx.get("similar_success_count", 0) / 10)
                )
            )
        )

    def decide(self,
               tree_name: str,
               context: Dict[str, Any],
               use_heuristics: bool = True) -> Decision:
        """
        Trifft eine Entscheidung basierend auf Baum und Kontext

        Args:
            tree_name: Name des Entscheidungsbaums
            context: Kontext-Daten für die Entscheidung
            use_heuristics: Ob Heuristiken berücksichtigt werden sollen

        Returns:
            Decision mit Aktion und Begründung
        """
        start_time = time.time()
        reasoning = []

        tree = self.trees.get(tree_name)
        if not tree:
            return Decision(
                action="unknown",
                confidence=0.0,
                reasoning=[f"Tree '{tree_name}' not found"],
                context=context,
                decision_type=DecisionType.DETERMINISTIC,
                processing_time_ms=0
            )

        # Baum evaluieren
        action, tree_reasoning = tree.evaluate(context)
        reasoning.extend(tree_reasoning)

        # Heuristiken anwenden
        confidence = 1.0
        if use_heuristics:
            heuristic_scores = {}
            for name, heuristic in self.heuristics.items():
                try:
                    score = heuristic.evaluate(context)
                    heuristic_scores[name] = score
                    reasoning.append(f"Heuristic {name}: {score:.2f}")
                except Exception as e:
                    reasoning.append(f"Heuristic {name} error: {e}")

            # Confidence aus Success-Probability Heuristik
            if "success_probability" in heuristic_scores:
                confidence = heuristic_scores["success_probability"]

        processing_time = (time.time() - start_time) * 1000

        decision = Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            context=context,
            decision_type=DecisionType.HEURISTIC if use_heuristics else DecisionType.DETERMINISTIC,
            processing_time_ms=processing_time
        )

        # Historie speichern (thread-safe)
        with self._history_lock:
            self.decision_history.append(decision)
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]

        return decision

    def decide_best_action(self,
                           actions: List[str],
                           context: Dict[str, Any],
                           score_func: Optional[Callable] = None) -> Tuple[str, float]:
        """
        Wählt die beste Aktion aus einer Liste

        Args:
            actions: Mögliche Aktionen
            context: Kontext für Bewertung
            score_func: Optionale Scoring-Funktion

        Returns:
            (beste_aktion, score)
        """
        if not actions:
            return "none", 0.0

        if score_func is None:
            # Standard: Priority-Heuristik
            score_func = self.heuristics.get("priority_score")
            if score_func:
                score_func = score_func.evaluate

        if score_func is None:
            # Fallback: Erste Aktion
            return actions[0], 1.0

        best_action = actions[0]
        best_score = float('-inf')

        for action in actions:
            ctx = {**context, "action": action}
            try:
                score = score_func(ctx)
                if score > best_score:
                    best_score = score
                    best_action = action
            except Exception:
                continue

        return best_action, best_score

    def learn_from_outcome(self, decision: Decision, success: bool, feedback: Dict = None):
        """
        Lernt aus dem Ergebnis einer Entscheidung

        Args:
            decision: Die getroffene Entscheidung
            success: Ob die Entscheidung erfolgreich war
            feedback: Optionales Feedback
        """
        # Heuristiken aktualisieren
        for heuristic in self.heuristics.values():
            heuristic.update_success(success)

        # Pattern speichern (thread-safe)
        pattern_key = f"{decision.action}_{hash(frozenset(decision.context.items())) % 10000}"

        with self._patterns_lock:
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    "action": decision.action,
                    "success_count": 0,
                    "failure_count": 0,
                    "context_features": list(decision.context.keys()),
                }

            pattern = self.learned_patterns[pattern_key]
            if success:
                pattern["success_count"] += 1
            else:
                pattern["failure_count"] += 1

    def add_tree(self, name: str, tree: DecisionTree):
        """Fügt einen Entscheidungsbaum hinzu"""
        self.trees[name] = tree

    def add_heuristic(self, name: str, heuristic: Heuristic):
        """Fügt eine Heuristik hinzu"""
        self.heuristics[name] = heuristic

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        total_decisions = len(self.decision_history)

        action_counts = {}
        avg_confidence = 0
        avg_time = 0

        for d in self.decision_history:
            action_counts[d.action] = action_counts.get(d.action, 0) + 1
            avg_confidence += d.confidence
            avg_time += d.processing_time_ms

        if total_decisions > 0:
            avg_confidence /= total_decisions
            avg_time /= total_decisions

        return {
            "total_decisions": total_decisions,
            "trees_count": len(self.trees),
            "heuristics_count": len(self.heuristics),
            "learned_patterns": len(self.learned_patterns),
            "action_distribution": action_counts,
            "avg_confidence": round(avg_confidence, 3),
            "avg_processing_time_ms": round(avg_time, 2),
            "heuristic_success_rates": {
                name: round(h.success_rate, 3)
                for name, h in self.heuristics.items()
            }
        }


# Singleton
_decision_engine: Optional[DecisionEngine] = None

def get_decision_engine() -> DecisionEngine:
    """Gibt Singleton-Instanz zurück"""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine()
    return _decision_engine
