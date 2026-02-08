#!/usr/bin/env python3
"""
SCIO - Continuous Learning
Kontinuierliches Lernen aus Feedback und Erfahrungen
"""

import json
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Feedback:
    """Benutzer-Feedback zu einer Aktion"""
    action_id: str
    rating: int  # 1-5
    comment: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningPattern:
    """Ein gelerntes Muster"""
    pattern_id: str
    condition: Dict[str, Any]
    outcome: str
    confidence: float
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None

    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


@dataclass
class PerformanceMetric:
    """Performance-Metrik über Zeit"""
    name: str
    values: List[Tuple[datetime, float]] = field(default_factory=list)
    window_size: int = 100

    def add(self, value: float):
        self.values.append((datetime.now(), value))
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size:]

    def average(self) -> float:
        if not self.values:
            return 0.0
        return sum(v[1] for v in self.values) / len(self.values)

    def trend(self) -> float:
        """Berechnet Trend (positiv = verbessernd)"""
        if len(self.values) < 10:
            return 0.0

        first_half = self.values[:len(self.values)//2]
        second_half = self.values[len(self.values)//2:]

        avg1 = sum(v[1] for v in first_half) / len(first_half)
        avg2 = sum(v[1] for v in second_half) / len(second_half)

        return avg2 - avg1


class ContinuousLearner:
    """
    SCIO Continuous Learning System

    Funktionen:
    - Sammelt und verarbeitet Benutzer-Feedback
    - Erkennt und speichert Muster
    - Überwacht Performance-Metriken
    - Triggert automatisches Retraining
    - A/B Testing für Verbesserungen
    """

    def __init__(self):
        self.feedback_history: List[Feedback] = []
        self._feedback_lock = threading.Lock()
        self.patterns: Dict[str, LearningPattern] = {}
        self._patterns_lock = threading.Lock()
        self.metrics: Dict[str, PerformanceMetric] = {}
        self._metrics_lock = threading.Lock()
        self.ab_tests: Dict[str, Dict] = {}
        self._ab_lock = threading.Lock()

        # Callbacks
        self.on_pattern_learned: List[Callable] = []
        self.on_performance_degradation: List[Callable] = []

        # Thresholds
        self.degradation_threshold = 0.1  # 10% Verschlechterung
        self.pattern_confidence_threshold = 0.7
        self.min_samples_for_pattern = 10

        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert das Continuous Learning System"""
        try:
            self._setup_default_metrics()
            self._load_patterns()
            self._initialized = True
            logger.info("Continuous Learner initialisiert")
            return True
        except Exception as e:
            logger.error(f"Continuous Learner Fehler: {e}")
            return False

    def _setup_default_metrics(self):
        """Erstellt Standard-Metriken"""
        self.metrics["response_time"] = PerformanceMetric("response_time")
        self.metrics["success_rate"] = PerformanceMetric("success_rate")
        self.metrics["user_satisfaction"] = PerformanceMetric("user_satisfaction")
        self.metrics["cost_per_request"] = PerformanceMetric("cost_per_request")
        self.metrics["token_efficiency"] = PerformanceMetric("token_efficiency")

    def _load_patterns(self):
        """Lädt gespeicherte Muster"""
        try:
            from backend.config import Config
            pattern_file = Config.DATA_DIR / "learned_patterns.json"
            if pattern_file.exists():
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                    for p in data.get("patterns", []):
                        pattern = LearningPattern(
                            pattern_id=p["pattern_id"],
                            condition=p["condition"],
                            outcome=p["outcome"],
                            confidence=p["confidence"],
                            success_count=p.get("success_count", 0),
                            failure_count=p.get("failure_count", 0)
                        )
                        self.patterns[pattern.pattern_id] = pattern
                logger.info(f"{len(self.patterns)} Muster geladen")
        except Exception:
            pass

    def save_patterns(self):
        """Speichert Muster"""
        try:
            from backend.config import Config
            pattern_file = Config.DATA_DIR / "learned_patterns.json"
            data = {
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "condition": p.condition,
                        "outcome": p.outcome,
                        "confidence": p.confidence,
                        "success_count": p.success_count,
                        "failure_count": p.failure_count
                    }
                    for p in self.patterns.values()
                ],
                "timestamp": datetime.now().isoformat()
            }
            with open(pattern_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Muster speichern fehlgeschlagen: {e}")

    def record_feedback(self, feedback: Feedback):
        """Zeichnet Benutzer-Feedback auf"""
        with self._feedback_lock:
            self.feedback_history.append(feedback)
            # Buffer begrenzen
            if len(self.feedback_history) > 10000:
                self.feedback_history = self.feedback_history[-5000:]

        # Satisfaction-Metrik aktualisieren
        with self._metrics_lock:
            self.metrics["user_satisfaction"].add(feedback.rating / 5.0)

        # Pattern-Lernen triggern
        self._learn_from_feedback(feedback)

    def record_metric(self, metric_name: str, value: float):
        """Zeichnet eine Metrik auf"""
        with self._metrics_lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = PerformanceMetric(metric_name)
            self.metrics[metric_name].add(value)

        # Degradation prüfen
        self._check_degradation(metric_name)

    def record_outcome(self,
                       context: Dict[str, Any],
                       action: str,
                       success: bool,
                       metrics: Dict[str, float] = None):
        """
        Zeichnet das Ergebnis einer Aktion auf

        Args:
            context: Kontext der Aktion
            action: Die ausgeführte Aktion
            success: Ob erfolgreich
            metrics: Optionale Metriken
        """
        # Metriken aufzeichnen
        if metrics:
            for name, value in metrics.items():
                self.record_metric(name, value)

        # Success Rate
        self.record_metric("success_rate", 1.0 if success else 0.0)

        # Pattern-Matching versuchen
        self._match_and_update_patterns(context, action, success)

    def _learn_from_feedback(self, feedback: Feedback):
        """Lernt aus Feedback"""
        # Extrahiere Kontext-Features
        context = feedback.context

        # Prüfe ob bestehendes Pattern passt
        for pattern in self.patterns.values():
            if self._matches_condition(context, pattern.condition):
                if feedback.rating >= 4:
                    pattern.success_count += 1
                else:
                    pattern.failure_count += 1
                pattern.last_used = datetime.now()
                return

        # Neues Pattern erstellen wenn genug ähnliches Feedback
        similar_feedback = [f for f in self.feedback_history[-100:]
                          if self._similar_context(f.context, context)]

        if len(similar_feedback) >= self.min_samples_for_pattern:
            avg_rating = sum(f.rating for f in similar_feedback) / len(similar_feedback)

            if avg_rating >= 4:
                # Positives Pattern
                pattern_id = f"pattern_{len(self.patterns) + 1}"
                pattern = LearningPattern(
                    pattern_id=pattern_id,
                    condition=self._extract_common_features(similar_feedback),
                    outcome="positive",
                    confidence=avg_rating / 5,
                    success_count=len([f for f in similar_feedback if f.rating >= 4])
                )
                self.patterns[pattern_id] = pattern

                for callback in self.on_pattern_learned:
                    callback(pattern)

    def _match_and_update_patterns(self, context: Dict, action: str, success: bool):
        """Matched und aktualisiert Patterns"""
        for pattern in self.patterns.values():
            if self._matches_condition(context, pattern.condition):
                if success:
                    pattern.success_count += 1
                else:
                    pattern.failure_count += 1

                # Confidence aktualisieren
                pattern.confidence = pattern.success_rate()
                pattern.last_used = datetime.now()

    def _matches_condition(self, context: Dict, condition: Dict) -> bool:
        """Prüft ob Kontext zur Bedingung passt"""
        for key, value in condition.items():
            if key not in context:
                return False
            if context[key] != value:
                return False
        return True

    def _similar_context(self, ctx1: Dict, ctx2: Dict) -> bool:
        """Prüft ob zwei Kontexte ähnlich sind"""
        common_keys = set(ctx1.keys()) & set(ctx2.keys())
        if not common_keys:
            return False

        matches = sum(1 for k in common_keys if ctx1[k] == ctx2[k])
        return matches / len(common_keys) > 0.7

    def _extract_common_features(self, feedbacks: List[Feedback]) -> Dict:
        """Extrahiert gemeinsame Features aus Feedback-Liste"""
        if not feedbacks:
            return {}

        # Zähle Feature-Werte
        feature_counts: Dict[str, Dict] = defaultdict(lambda: defaultdict(int))

        for fb in feedbacks:
            for key, value in fb.context.items():
                feature_counts[key][str(value)] += 1

        # Wähle häufigste Werte
        common = {}
        for key, values in feature_counts.items():
            most_common = max(values.keys(), key=lambda v: values[v])
            if values[most_common] / len(feedbacks) > 0.6:
                common[key] = most_common

        return common

    def _check_degradation(self, metric_name: str):
        """Prüft auf Performance-Degradation"""
        metric = self.metrics.get(metric_name)
        if not metric:
            return

        trend = metric.trend()

        # Negative Trends bei positiven Metriken (success_rate, satisfaction)
        # Positive Trends bei negativen Metriken (response_time, cost)
        is_degrading = False

        if metric_name in ["success_rate", "user_satisfaction", "token_efficiency"]:
            is_degrading = trend < -self.degradation_threshold
        elif metric_name in ["response_time", "cost_per_request"]:
            is_degrading = trend > self.degradation_threshold

        if is_degrading:
            for callback in self.on_performance_degradation:
                callback(metric_name, trend, metric.average())

    def get_recommendation(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Gibt eine Empfehlung basierend auf gelernten Mustern

        Args:
            context: Aktueller Kontext

        Returns:
            Empfohlene Aktion oder None
        """
        best_pattern = None
        best_confidence = 0

        for pattern in self.patterns.values():
            if self._matches_condition(context, pattern.condition):
                if pattern.confidence > best_confidence:
                    best_confidence = pattern.confidence
                    best_pattern = pattern

        if best_pattern and best_confidence >= self.pattern_confidence_threshold:
            return best_pattern.outcome

        return None

    def start_ab_test(self,
                      test_id: str,
                      variants: List[str],
                      metric: str = "success_rate") -> str:
        """
        Startet einen A/B Test

        Args:
            test_id: ID des Tests
            variants: Liste der Varianten
            metric: Zu optimierende Metrik

        Returns:
            Die Variante für diesen Request
        """
        if test_id not in self.ab_tests:
            self.ab_tests[test_id] = {
                "variants": {v: {"count": 0, "metric_sum": 0} for v in variants},
                "metric": metric,
                "started": datetime.now()
            }

        # Wähle Variante mit niedrigster Anzahl (Balancing)
        test = self.ab_tests[test_id]
        variant = min(test["variants"].keys(),
                     key=lambda v: test["variants"][v]["count"])

        test["variants"][variant]["count"] += 1
        return variant

    def record_ab_result(self, test_id: str, variant: str, metric_value: float):
        """Zeichnet A/B Test Ergebnis auf"""
        if test_id in self.ab_tests:
            test = self.ab_tests[test_id]
            if variant in test["variants"]:
                test["variants"][variant]["metric_sum"] += metric_value

    def get_ab_winner(self, test_id: str) -> Optional[Tuple[str, float]]:
        """Gibt den Gewinner eines A/B Tests zurück"""
        if test_id not in self.ab_tests:
            return None

        test = self.ab_tests[test_id]
        best_variant = None
        best_avg = float('-inf')

        for variant, data in test["variants"].items():
            if data["count"] > 0:
                avg = data["metric_sum"] / data["count"]
                if avg > best_avg:
                    best_avg = avg
                    best_variant = variant

        return (best_variant, best_avg) if best_variant else None

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        return {
            "patterns_count": len(self.patterns),
            "high_confidence_patterns": len([p for p in self.patterns.values()
                                            if p.confidence >= self.pattern_confidence_threshold]),
            "feedback_count": len(self.feedback_history),
            "active_ab_tests": len(self.ab_tests),
            "metrics": {
                name: {
                    "average": round(m.average(), 4),
                    "trend": round(m.trend(), 4),
                    "samples": len(m.values)
                }
                for name, m in self.metrics.items()
            }
        }


# Singleton
_continuous_learner: Optional[ContinuousLearner] = None

def get_continuous_learner() -> ContinuousLearner:
    """Gibt Singleton-Instanz zurück"""
    global _continuous_learner
    if _continuous_learner is None:
        _continuous_learner = ContinuousLearner()
    return _continuous_learner
