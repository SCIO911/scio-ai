"""
SCIO Meta-Cognition - Wissen über das eigene Wissen

Erweiterte Meta-kognitive Fähigkeiten:
- Wissen über eigenes Wissen (Epistemic Awareness)
- Erkennen von Wissenslücken (Gap Detection)
- Konfidenz-Kalibrierung (Calibration)
- Unsicherheits-Quantifizierung (Uncertainty)
- Lernstrategie-Optimierung (Learning Strategies)
- Kognitive Ressourcen-Management (Resource Management)
"""

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class KnowledgeState(str, Enum):
    """Zustände des Wissens."""
    KNOWN = "known"                    # Sicher gewusst
    PARTIALLY_KNOWN = "partially_known"  # Teilweise gewusst
    UNCERTAIN = "uncertain"            # Unsicher
    UNKNOWN = "unknown"                # Nicht gewusst
    MISCONCEPTION = "misconception"    # Falsches Wissen


class CognitiveLoad(str, Enum):
    """Kognitive Last-Level."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    OVERLOAD = "overload"


class LearningStrategy(str, Enum):
    """Verfügbare Lernstrategien."""
    RETRIEVAL_PRACTICE = "retrieval_practice"
    SPACED_REPETITION = "spaced_repetition"
    ELABORATION = "elaboration"
    INTERLEAVING = "interleaving"
    DUAL_CODING = "dual_coding"
    SELF_EXPLANATION = "self_explanation"


@dataclass
class KnowledgeAssessment:
    """Bewertung des Wissens zu einem Thema."""

    topic: str
    state: KnowledgeState
    confidence: float  # 0-1
    accuracy: float  # 0-1, historische Genauigkeit
    depth: int  # 1-5, Tiefe des Wissens
    last_accessed: Optional[datetime] = None
    sources: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "state": self.state.value,
            "confidence": self.confidence,
            "accuracy": self.accuracy,
            "depth": self.depth,
            "gaps": self.gaps,
        }


@dataclass
class CalibrationRecord:
    """Aufzeichnung für Konfidenz-Kalibrierung."""

    prediction: str
    stated_confidence: float
    actual_outcome: bool
    timestamp: datetime = field(default_factory=now_utc)


@dataclass
class CognitiveState:
    """Aktueller kognitiver Zustand."""

    load: CognitiveLoad
    attention_focus: Optional[str] = None
    working_memory_usage: float = 0.0  # 0-1
    processing_speed: float = 1.0  # Relative Geschwindigkeit
    fatigue_level: float = 0.0  # 0-1


@dataclass
class LearningPlan:
    """Plan zum Lernen eines Themas."""

    topic: str
    current_state: KnowledgeState
    target_state: KnowledgeState
    strategies: List[LearningStrategy]
    milestones: List[str]
    estimated_sessions: int
    resources: List[str] = field(default_factory=list)


# ============================================================================
# META-COGNITION ENGINE
# ============================================================================

class MetaCognitionEngine:
    """
    Meta-Kognitions-Engine für tiefes Selbstverständnis.

    Ermöglicht SCIO:
    - Eigenes Wissen zu bewerten
    - Wissenslücken zu erkennen
    - Konfidenz zu kalibrieren
    - Lernstrategien zu optimieren
    - Kognitive Ressourcen zu verwalten
    """

    def __init__(self, llm_callback: Optional[Callable] = None):
        self.llm_callback = llm_callback or self._default_llm

        # Knowledge Map
        self._knowledge_map: Dict[str, KnowledgeAssessment] = {}

        # Calibration History
        self._calibration_history: List[CalibrationRecord] = []
        self._calibration_buckets: Dict[int, List[bool]] = defaultdict(list)

        # Cognitive State
        self._cognitive_state = CognitiveState(load=CognitiveLoad.LOW)

        # Learning History
        self._learning_sessions: List[Dict[str, Any]] = []

        logger.info("MetaCognitionEngine initialized")

    async def _default_llm(self, prompt: str) -> str:
        """Standard LLM Callback."""
        return f"Meta-cognitive analysis: {prompt[:50]}..."

    # ========================================================================
    # EPISTEMIC AWARENESS
    # ========================================================================

    async def assess_knowledge(self, topic: str) -> KnowledgeAssessment:
        """
        Bewertet das eigene Wissen zu einem Thema.

        Args:
            topic: Das zu bewertende Thema

        Returns:
            KnowledgeAssessment mit detaillierter Bewertung
        """
        # Check if already assessed recently
        if topic in self._knowledge_map:
            existing = self._knowledge_map[topic]
            if existing.last_accessed:
                hours_since = (now_utc() - existing.last_accessed).total_seconds() / 3600
                if hours_since < 24:
                    return existing

        # Generate assessment prompt
        prompt = f"""Assess your knowledge about: {topic}

Rate the following aspects:
1. How well do you know this topic? (1-5 scale)
   1 = Unknown, 2 = Vaguely familiar, 3 = Basic understanding, 4 = Good knowledge, 5 = Expert
2. What is your confidence in this assessment? (0-100%)
3. What specific areas within this topic do you know well?
4. What are the gaps in your knowledge?
5. What related topics might help understand this better?

Provide a structured assessment:"""

        response = await self.llm_callback(prompt)

        # Parse response (simplified)
        assessment = self._parse_assessment(topic, response)

        # Store in knowledge map
        self._knowledge_map[topic] = assessment

        return assessment

    def _parse_assessment(self, topic: str, response: str) -> KnowledgeAssessment:
        """Parst die Wissensbewertung aus der LLM-Antwort."""
        # Heuristische Analyse
        response_lower = response.lower()

        # Bestimme Wissenszustand
        if "don't know" in response_lower or "unknown" in response_lower:
            state = KnowledgeState.UNKNOWN
            confidence = 0.2
            depth = 1
        elif "not sure" in response_lower or "uncertain" in response_lower:
            state = KnowledgeState.UNCERTAIN
            confidence = 0.4
            depth = 2
        elif "some knowledge" in response_lower or "basic" in response_lower:
            state = KnowledgeState.PARTIALLY_KNOWN
            confidence = 0.6
            depth = 3
        elif "well" in response_lower or "expert" in response_lower:
            state = KnowledgeState.KNOWN
            confidence = 0.8
            depth = 4
        else:
            state = KnowledgeState.PARTIALLY_KNOWN
            confidence = 0.5
            depth = 2

        # Extrahiere Lücken
        gaps = []
        if "gap" in response_lower or "don't know" in response_lower:
            # Vereinfachte Extraktion
            lines = response.split("\n")
            for line in lines:
                if "gap" in line.lower() or "don't know" in line.lower():
                    gaps.append(line.strip()[:100])

        return KnowledgeAssessment(
            topic=topic,
            state=state,
            confidence=confidence,
            accuracy=self._get_historical_accuracy(topic),
            depth=depth,
            last_accessed=now_utc(),
            gaps=gaps[:5],
        )

    def _get_historical_accuracy(self, topic: str) -> float:
        """Berechnet historische Genauigkeit für ein Thema."""
        # Basierend auf Kalibrierungshistorie
        relevant = [
            r for r in self._calibration_history
            if topic.lower() in r.prediction.lower()
        ]

        if not relevant:
            return 0.7  # Default

        correct = sum(1 for r in relevant if r.actual_outcome)
        return correct / len(relevant)

    # ========================================================================
    # GAP DETECTION
    # ========================================================================

    async def identify_gaps(self, topic: str) -> List[str]:
        """
        Identifiziert Wissenslücken zu einem Thema.

        Args:
            topic: Das zu analysierende Thema

        Returns:
            Liste von identifizierten Wissenslücken
        """
        prompt = f"""Analyze your knowledge of: {topic}

Identify specific gaps, weaknesses, or areas where you might make errors.
Consider:
1. Fundamental concepts you might be missing
2. Recent developments you might not know
3. Edge cases or exceptions you might overlook
4. Common misconceptions in this area

List your knowledge gaps:"""

        response = await self.llm_callback(prompt)

        # Extrahiere Lücken
        gaps = []
        lines = response.split("\n")
        for line in lines:
            cleaned = line.strip()
            if cleaned and len(cleaned) > 10 and not cleaned.startswith("#"):
                # Entferne Nummerierung
                if cleaned[0].isdigit() and cleaned[1] in ".):":
                    cleaned = cleaned[2:].strip()
                gaps.append(cleaned[:200])

        return gaps[:10]

    async def detect_misconceptions(self, topic: str, statement: str) -> Dict[str, Any]:
        """
        Erkennt mögliche Fehlvorstellungen.

        Args:
            topic: Das Themengebiet
            statement: Eine Aussage zum Prüfen

        Returns:
            Analyse möglicher Fehlvorstellungen
        """
        prompt = f"""Analyze this statement for potential misconceptions:

Topic: {topic}
Statement: {statement}

1. Is this statement accurate?
2. What aspects might be incorrect or misleading?
3. What is the correct understanding?
4. What common misconceptions are related to this?

Analysis:"""

        response = await self.llm_callback(prompt)

        # Analysiere Antwort
        is_misconception = "incorrect" in response.lower() or "wrong" in response.lower()

        return {
            "statement": statement,
            "topic": topic,
            "is_misconception": is_misconception,
            "analysis": response,
            "confidence": 0.7 if is_misconception else 0.8,
        }

    # ========================================================================
    # CONFIDENCE CALIBRATION
    # ========================================================================

    def record_prediction(
        self,
        prediction: str,
        confidence: float,
        actual_outcome: Optional[bool] = None,
    ) -> CalibrationRecord:
        """
        Zeichnet eine Vorhersage für Kalibrierung auf.

        Args:
            prediction: Die Vorhersage
            confidence: Angegebene Konfidenz (0-1)
            actual_outcome: Tatsächliches Ergebnis (wenn bekannt)

        Returns:
            CalibrationRecord
        """
        record = CalibrationRecord(
            prediction=prediction,
            stated_confidence=confidence,
            actual_outcome=actual_outcome if actual_outcome is not None else True,
        )

        self._calibration_history.append(record)

        # Update calibration buckets (10% intervals)
        bucket = int(confidence * 10)
        if actual_outcome is not None:
            self._calibration_buckets[bucket].append(actual_outcome)

        return record

    def update_prediction_outcome(self, prediction: str, actual_outcome: bool) -> None:
        """Aktualisiert das Ergebnis einer Vorhersage."""
        for record in reversed(self._calibration_history):
            if record.prediction == prediction:
                record.actual_outcome = actual_outcome
                bucket = int(record.stated_confidence * 10)
                self._calibration_buckets[bucket].append(actual_outcome)
                break

    def get_calibration_stats(self) -> Dict[str, Any]:
        """
        Berechnet Kalibrierungsstatistiken.

        Returns:
            Dict mit Kalibrierungsmetriken
        """
        if not self._calibration_history:
            return {"message": "No calibration data available"}

        # Gesamtstatistiken
        total = len(self._calibration_history)
        total_confidence = sum(r.stated_confidence for r in self._calibration_history)
        total_correct = sum(1 for r in self._calibration_history if r.actual_outcome)

        avg_confidence = total_confidence / total
        avg_accuracy = total_correct / total

        # Kalibrierungsfehler
        calibration_error = abs(avg_confidence - avg_accuracy)

        # Brier Score
        brier = sum(
            (r.stated_confidence - (1 if r.actual_outcome else 0)) ** 2
            for r in self._calibration_history
        ) / total

        # Bucket-Analyse
        bucket_stats = {}
        for bucket, outcomes in self._calibration_buckets.items():
            if outcomes:
                expected = bucket / 10
                actual = sum(outcomes) / len(outcomes)
                bucket_stats[f"{bucket*10}-{(bucket+1)*10}%"] = {
                    "expected": expected,
                    "actual": actual,
                    "count": len(outcomes),
                }

        return {
            "total_predictions": total,
            "average_confidence": avg_confidence,
            "average_accuracy": avg_accuracy,
            "calibration_error": calibration_error,
            "brier_score": brier,
            "is_overconfident": avg_confidence > avg_accuracy,
            "is_underconfident": avg_confidence < avg_accuracy,
            "bucket_analysis": bucket_stats,
        }

    def calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Kalibriert eine Konfidenzangabe basierend auf Historie.

        Args:
            raw_confidence: Ursprüngliche Konfidenz

        Returns:
            Kalibrierte Konfidenz
        """
        stats = self.get_calibration_stats()

        if "calibration_error" not in stats:
            return raw_confidence

        # Einfache Kalibrierung: Anpassung basierend auf historischem Bias
        if stats["is_overconfident"]:
            factor = stats["average_accuracy"] / stats["average_confidence"]
            return min(1.0, raw_confidence * factor)
        elif stats["is_underconfident"]:
            factor = stats["average_accuracy"] / stats["average_confidence"]
            return min(1.0, raw_confidence * factor)

        return raw_confidence

    # ========================================================================
    # UNCERTAINTY QUANTIFICATION
    # ========================================================================

    async def quantify_uncertainty(
        self,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Quantifiziert die Unsicherheit einer Antwort.

        Args:
            question: Die gestellte Frage
            answer: Die gegebene Antwort

        Returns:
            Unsicherheitsanalyse
        """
        prompt = f"""Analyze the uncertainty in this answer:

Question: {question}
Answer: {answer}

Rate on a scale of 0-10:
1. Epistemic uncertainty (lack of knowledge): ?
2. Aleatory uncertainty (inherent randomness): ?
3. Model uncertainty (limitations of reasoning): ?

Also identify:
- What assumptions were made?
- What could make this answer wrong?
- How could uncertainty be reduced?

Analysis:"""

        response = await self.llm_callback(prompt)

        # Extrahiere Unsicherheitswerte (vereinfacht)
        epistemic = 0.3
        aleatory = 0.2
        model = 0.2

        # Suche nach Zahlen in der Antwort
        import re
        numbers = re.findall(r'\d+', response[:200])
        if len(numbers) >= 3:
            try:
                epistemic = int(numbers[0]) / 10
                aleatory = int(numbers[1]) / 10
                model = int(numbers[2]) / 10
            except (ValueError, IndexError):
                pass

        total_uncertainty = (epistemic + aleatory + model) / 3

        return {
            "question": question,
            "epistemic_uncertainty": epistemic,
            "aleatory_uncertainty": aleatory,
            "model_uncertainty": model,
            "total_uncertainty": total_uncertainty,
            "confidence": 1 - total_uncertainty,
            "analysis": response,
        }

    # ========================================================================
    # LEARNING STRATEGIES
    # ========================================================================

    async def suggest_learning_strategy(
        self,
        topic: str,
        current_knowledge: Optional[KnowledgeAssessment] = None,
    ) -> LearningPlan:
        """
        Schlägt eine optimale Lernstrategie vor.

        Args:
            topic: Das zu lernende Thema
            current_knowledge: Aktuelle Wissensbewertung

        Returns:
            LearningPlan mit Strategien
        """
        if current_knowledge is None:
            current_knowledge = await self.assess_knowledge(topic)

        # Wähle Strategien basierend auf aktuellem Stand
        strategies = []

        if current_knowledge.state == KnowledgeState.UNKNOWN:
            strategies = [
                LearningStrategy.ELABORATION,
                LearningStrategy.DUAL_CODING,
                LearningStrategy.SELF_EXPLANATION,
            ]
            estimated_sessions = 10
        elif current_knowledge.state == KnowledgeState.PARTIALLY_KNOWN:
            strategies = [
                LearningStrategy.RETRIEVAL_PRACTICE,
                LearningStrategy.INTERLEAVING,
                LearningStrategy.ELABORATION,
            ]
            estimated_sessions = 5
        elif current_knowledge.state == KnowledgeState.UNCERTAIN:
            strategies = [
                LearningStrategy.RETRIEVAL_PRACTICE,
                LearningStrategy.SPACED_REPETITION,
            ]
            estimated_sessions = 3
        else:
            strategies = [
                LearningStrategy.SPACED_REPETITION,
            ]
            estimated_sessions = 1

        # Generiere Meilensteine
        milestones = []
        if current_knowledge.gaps:
            for gap in current_knowledge.gaps[:3]:
                milestones.append(f"Address gap: {gap[:50]}")
        milestones.append(f"Achieve depth level {min(current_knowledge.depth + 1, 5)}")
        milestones.append(f"Pass self-assessment with 80%+ confidence")

        return LearningPlan(
            topic=topic,
            current_state=current_knowledge.state,
            target_state=KnowledgeState.KNOWN,
            strategies=strategies,
            milestones=milestones,
            estimated_sessions=estimated_sessions,
        )

    # ========================================================================
    # COGNITIVE RESOURCE MANAGEMENT
    # ========================================================================

    def update_cognitive_state(
        self,
        attention_focus: Optional[str] = None,
        working_memory_usage: Optional[float] = None,
        fatigue_level: Optional[float] = None,
    ) -> CognitiveState:
        """
        Aktualisiert den kognitiven Zustand.

        Args:
            attention_focus: Aktueller Aufmerksamkeitsfokus
            working_memory_usage: Arbeitsgedächtnis-Nutzung (0-1)
            fatigue_level: Ermüdungsgrad (0-1)

        Returns:
            Aktualisierter CognitiveState
        """
        if attention_focus is not None:
            self._cognitive_state.attention_focus = attention_focus

        if working_memory_usage is not None:
            self._cognitive_state.working_memory_usage = working_memory_usage

        if fatigue_level is not None:
            self._cognitive_state.fatigue_level = fatigue_level

        # Berechne kognitive Last
        load_score = (
            self._cognitive_state.working_memory_usage * 0.5 +
            self._cognitive_state.fatigue_level * 0.5
        )

        if load_score < 0.3:
            self._cognitive_state.load = CognitiveLoad.LOW
        elif load_score < 0.6:
            self._cognitive_state.load = CognitiveLoad.MODERATE
        elif load_score < 0.85:
            self._cognitive_state.load = CognitiveLoad.HIGH
        else:
            self._cognitive_state.load = CognitiveLoad.OVERLOAD

        # Passe Verarbeitungsgeschwindigkeit an
        self._cognitive_state.processing_speed = 1.0 - (load_score * 0.5)

        return self._cognitive_state

    def get_cognitive_state(self) -> CognitiveState:
        """Gibt den aktuellen kognitiven Zustand zurück."""
        return self._cognitive_state

    def should_take_break(self) -> bool:
        """Empfiehlt eine Pause wenn nötig."""
        return (
            self._cognitive_state.load == CognitiveLoad.OVERLOAD or
            self._cognitive_state.fatigue_level > 0.8
        )

    # ========================================================================
    # REFLECTION
    # ========================================================================

    async def reflect_on_process(
        self,
        task: str,
        approach: str,
        result: str,
    ) -> Dict[str, Any]:
        """
        Reflektiert über einen Denkprozess.

        Args:
            task: Die bearbeitete Aufgabe
            approach: Der gewählte Ansatz
            result: Das Ergebnis

        Returns:
            Reflexionsanalyse mit Verbesserungsvorschlägen
        """
        prompt = f"""Reflect on this problem-solving process:

Task: {task}
Approach used: {approach}
Result: {result}

Analyze:
1. What went well in this process?
2. What could have been done better?
3. What alternative approaches could have been used?
4. What did we learn for future tasks?
5. How confident are we in the result?

Reflection:"""

        response = await self.llm_callback(prompt)

        return {
            "task": task,
            "reflection": response,
            "lessons_learned": self._extract_lessons(response),
            "improvement_suggestions": self._extract_suggestions(response),
        }

    def _extract_lessons(self, reflection: str) -> List[str]:
        """Extrahiert Lektionen aus der Reflexion."""
        lessons = []
        lines = reflection.split("\n")
        for line in lines:
            if "learn" in line.lower() or "lesson" in line.lower():
                lessons.append(line.strip()[:200])
        return lessons[:5]

    def _extract_suggestions(self, reflection: str) -> List[str]:
        """Extrahiert Verbesserungsvorschläge."""
        suggestions = []
        lines = reflection.split("\n")
        for line in lines:
            if "better" in line.lower() or "improve" in line.lower() or "could" in line.lower():
                suggestions.append(line.strip()[:200])
        return suggestions[:5]

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Meta-Kognitions-Statistiken zurück."""
        knowledge_states = defaultdict(int)
        for assessment in self._knowledge_map.values():
            knowledge_states[assessment.state.value] += 1

        return {
            "topics_assessed": len(self._knowledge_map),
            "knowledge_states": dict(knowledge_states),
            "calibration": self.get_calibration_stats(),
            "cognitive_state": {
                "load": self._cognitive_state.load.value,
                "fatigue": self._cognitive_state.fatigue_level,
                "working_memory_usage": self._cognitive_state.working_memory_usage,
            },
            "learning_sessions": len(self._learning_sessions),
        }


# ============================================================================
# SINGLETON
# ============================================================================

_metacognition_instance: Optional[MetaCognitionEngine] = None


def get_metacognition(llm_callback: Optional[Callable] = None) -> MetaCognitionEngine:
    """Gibt eine Singleton-Instanz zurück."""
    global _metacognition_instance
    if _metacognition_instance is None:
        _metacognition_instance = MetaCognitionEngine(llm_callback)
    return _metacognition_instance
