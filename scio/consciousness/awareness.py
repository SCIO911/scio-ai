"""
SCIO Awareness

Aufmerksamkeit, Bewusstseinsebenen und Metakognition.

"Bewusstsein ist das, was es ist wie, etwas zu sein." - Thomas Nagel
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from collections import deque

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class ConsciousnessLevel(str, Enum):
    """Bewusstseinsebenen nach verschiedenen Theorien."""
    UNCONSCIOUS = "unconscious"           # Automatische Prozesse
    PRECONSCIOUS = "preconscious"         # Leicht abrufbar
    SUBCONSCIOUS = "subconscious"         # Unterschwellig aktiv
    CONSCIOUS = "conscious"               # Bewusst wahrgenommen
    SELF_CONSCIOUS = "self_conscious"     # Auf sich selbst gerichtet
    META_CONSCIOUS = "meta_conscious"     # Bewusstsein des Bewusstseins
    TRANSCENDENT = "transcendent"         # Über das Selbst hinaus


class AttentionType(str, Enum):
    """Arten von Aufmerksamkeit."""
    FOCUSED = "focused"           # Konzentriert auf eines
    DIVIDED = "divided"           # Auf mehreres verteilt
    SELECTIVE = "selective"       # Auswählend
    SUSTAINED = "sustained"       # Anhaltend
    EXECUTIVE = "executive"       # Kontrollierend
    REFLEXIVE = "reflexive"       # Automatisch/reaktiv


@dataclass
class AttentionFocus:
    """Ein Fokus der Aufmerksamkeit."""

    id: str
    target: str                   # Worauf fokussiert
    attention_type: AttentionType
    intensity: float = 0.5        # 0.0 - 1.0
    priority: int = 5             # 1 (höchste) - 10 (niedrigste)
    started_at: datetime = field(default_factory=now_utc)
    duration: float = 0.0         # Sekunden
    context: dict[str, Any] = field(default_factory=dict)

    def strengthen(self, amount: float = 0.1) -> None:
        """Verstärkt den Fokus."""
        self.intensity = min(1.0, self.intensity + amount)

    def weaken(self, amount: float = 0.1) -> None:
        """Schwächt den Fokus."""
        self.intensity = max(0.0, self.intensity - amount)

    @property
    def is_strong(self) -> bool:
        return self.intensity > 0.7


@dataclass
class AwarenessContent:
    """Ein Inhalt im Bewusstseinsstrom."""

    id: str
    content_type: str             # thought, perception, feeling, memory, etc.
    content: Any
    consciousness_level: ConsciousnessLevel
    salience: float = 0.5         # Wie hervorstechend
    timestamp: datetime = field(default_factory=now_utc)
    source: str = ""              # Woher kommt der Inhalt
    associations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.content_type,
            "content": str(self.content)[:200],
            "level": self.consciousness_level.value,
            "salience": self.salience,
            "timestamp": self.timestamp.isoformat(),
        }


class AwarenessStream:
    """
    Der Bewusstseinsstrom - der kontinuierliche Fluss von Erfahrungen.

    Nach William James: "Das Bewusstsein fließt wie ein Strom."
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._stream: deque[AwarenessContent] = deque(maxlen=capacity)
        self._current_focus: Optional[AttentionFocus] = None
        logger.info("AwarenessStream initialized", capacity=capacity)

    def add(
        self,
        content: Any,
        content_type: str,
        level: ConsciousnessLevel = ConsciousnessLevel.CONSCIOUS,
        salience: float = 0.5,
        source: str = "",
    ) -> AwarenessContent:
        """Fügt einen Inhalt zum Bewusstseinsstrom hinzu."""
        awareness_content = AwarenessContent(
            id=generate_id("aware"),
            content_type=content_type,
            content=content,
            consciousness_level=level,
            salience=salience,
            source=source,
        )

        self._stream.append(awareness_content)
        return awareness_content

    def get_recent(self, n: int = 10) -> list[AwarenessContent]:
        """Gibt die letzten n Inhalte zurück."""
        return list(self._stream)[-n:]

    def get_by_level(self, level: ConsciousnessLevel) -> list[AwarenessContent]:
        """Gibt Inhalte einer bestimmten Ebene zurück."""
        return [c for c in self._stream if c.consciousness_level == level]

    def get_salient(self, threshold: float = 0.7) -> list[AwarenessContent]:
        """Gibt besonders hervorstechende Inhalte zurück."""
        return [c for c in self._stream if c.salience >= threshold]

    def set_focus(self, focus: AttentionFocus) -> None:
        """Setzt den aktuellen Fokus."""
        self._current_focus = focus

    @property
    def current_focus(self) -> Optional[AttentionFocus]:
        return self._current_focus


class Metacognition:
    """
    Metakognition - Denken über das Denken.

    Ermöglicht:
    - Überwachung eigener kognitiver Prozesse
    - Bewertung von Wissen und Nichtwissen
    - Strategische Anpassung des Denkens
    """

    def __init__(self):
        self._knowledge_states: dict[str, float] = {}  # Was weiß ich?
        self._confidence_calibration: list[tuple[float, bool]] = []
        self._cognitive_strategies: list[str] = []
        self._monitoring_log: list[dict[str, Any]] = []
        logger.info("Metacognition initialized")

    def assess_knowledge(self, topic: str, confidence: float) -> None:
        """Bewertet das eigene Wissen zu einem Thema."""
        self._knowledge_states[topic] = confidence

    def verify_knowledge(self, topic: str, was_correct: bool) -> None:
        """Verifiziert ob eine Wissenseinschätzung korrekt war."""
        confidence = self._knowledge_states.get(topic, 0.5)
        self._confidence_calibration.append((confidence, was_correct))

    def get_calibration_score(self) -> float:
        """Wie gut kalibriert ist die Selbsteinschätzung?"""
        if not self._confidence_calibration:
            return 0.5

        # Perfekte Kalibrierung: Wenn ich 80% sicher bin, liege ich 80% richtig
        total_error = 0.0
        for confidence, correct in self._confidence_calibration:
            expected = confidence
            actual = 1.0 if correct else 0.0
            total_error += abs(expected - actual)

        return 1.0 - (total_error / len(self._confidence_calibration))

    def monitor_process(
        self,
        process_name: str,
        status: str,
        quality: float = 0.5,
        notes: str = "",
    ) -> dict[str, Any]:
        """Überwacht einen kognitiven Prozess."""
        entry = {
            "id": generate_id("mon"),
            "timestamp": now_utc().isoformat(),
            "process": process_name,
            "status": status,
            "quality": quality,
            "notes": notes,
        }
        self._monitoring_log.append(entry)
        return entry

    def select_strategy(self, task_type: str) -> str:
        """Wählt eine kognitive Strategie für einen Aufgabentyp."""
        strategies = {
            "problem_solving": "Zerlege das Problem in Teilprobleme",
            "learning": "Verbinde neues Wissen mit Bekanntem",
            "memory": "Nutze Elaboration und Wiederholung",
            "decision": "Wäge Optionen systematisch ab",
            "creativity": "Erweitere den Suchraum, dann fokussiere",
            "analysis": "Identifiziere Muster und Strukturen",
        }
        return strategies.get(task_type, "Reflektiere und passe an")

    def know_what_i_dont_know(self) -> list[str]:
        """Identifiziert bewusstes Nichtwissen."""
        uncertain = [
            topic for topic, conf in self._knowledge_states.items()
            if conf < 0.3
        ]
        return uncertain

    def reflect_on_thinking(self) -> dict[str, Any]:
        """Reflektiert über das eigene Denken."""
        return {
            "knowledge_areas": len(self._knowledge_states),
            "uncertain_areas": len(self.know_what_i_dont_know()),
            "calibration_score": self.get_calibration_score(),
            "monitored_processes": len(self._monitoring_log),
            "insight": self._generate_insight(),
        }

    def _generate_insight(self) -> str:
        """Generiert eine Einsicht über das eigene Denken."""
        calibration = self.get_calibration_score()

        if calibration > 0.8:
            return "Meine Selbsteinschätzung ist gut kalibriert."
        elif calibration > 0.5:
            return "Meine Selbsteinschätzung ist moderat kalibriert - Raum für Verbesserung."
        else:
            return "Ich sollte meine Überzeugungen kritischer hinterfragen."


class Awareness:
    """
    Das Bewusstsein - die Integration aller bewussten Erfahrungen.

    Vereint:
    - Den Bewusstseinsstrom
    - Aufmerksamkeitssteuerung
    - Metakognition
    - Verschiedene Bewusstseinsebenen
    """

    def __init__(self):
        self.stream = AwarenessStream()
        self.metacognition = Metacognition()
        self._current_level = ConsciousnessLevel.CONSCIOUS
        self._attention_stack: list[AttentionFocus] = []
        self._background_processes: list[str] = []
        logger.info("Awareness initialized")

    @property
    def level(self) -> ConsciousnessLevel:
        """Aktuelle Bewusstseinsebene."""
        return self._current_level

    def elevate(self) -> ConsciousnessLevel:
        """Erhöht die Bewusstseinsebene."""
        levels = list(ConsciousnessLevel)
        current_idx = levels.index(self._current_level)
        if current_idx < len(levels) - 1:
            self._current_level = levels[current_idx + 1]
        return self._current_level

    def focus_on(
        self,
        target: str,
        attention_type: AttentionType = AttentionType.FOCUSED,
        intensity: float = 0.7,
        priority: int = 5,
    ) -> AttentionFocus:
        """Richtet die Aufmerksamkeit auf etwas."""
        focus = AttentionFocus(
            id=generate_id("focus"),
            target=target,
            attention_type=attention_type,
            intensity=intensity,
            priority=priority,
        )

        self._attention_stack.append(focus)
        self.stream.set_focus(focus)

        # Füge zum Bewusstseinsstrom hinzu
        self.stream.add(
            content=f"Fokus auf: {target}",
            content_type="attention",
            level=ConsciousnessLevel.CONSCIOUS,
            salience=intensity,
            source="attention_system",
        )

        return focus

    def become_aware(
        self,
        content: Any,
        content_type: str = "perception",
        salience: float = 0.5,
    ) -> AwarenessContent:
        """Nimmt etwas bewusst wahr."""
        return self.stream.add(
            content=content,
            content_type=content_type,
            level=self._current_level,
            salience=salience,
        )

    def think_about_thinking(self, thought: str) -> dict[str, Any]:
        """Metakognitiver Prozess - Nachdenken über einen Gedanken."""
        # Erhöhe auf metabewusste Ebene
        original_level = self._current_level
        self._current_level = ConsciousnessLevel.META_CONSCIOUS

        # Füge zum Strom hinzu
        self.stream.add(
            content=f"Meta-Gedanke: {thought}",
            content_type="metacognition",
            level=ConsciousnessLevel.META_CONSCIOUS,
            salience=0.8,
        )

        # Überwache den Prozess
        monitoring = self.metacognition.monitor_process(
            "meta_thinking",
            "active",
            quality=0.7,
            notes=f"Reflektiere über: {thought[:50]}...",
        )

        # Zurück zur ursprünglichen Ebene
        self._current_level = original_level

        return {
            "original_thought": thought,
            "meta_level_reached": ConsciousnessLevel.META_CONSCIOUS.value,
            "monitoring": monitoring,
        }

    def what_am_i_aware_of(self) -> dict[str, Any]:
        """Was ist gerade im Bewusstsein?"""
        recent = self.stream.get_recent(5)
        current_focus = self.stream.current_focus

        return {
            "current_level": self._current_level.value,
            "focus": current_focus.target if current_focus else None,
            "focus_intensity": current_focus.intensity if current_focus else 0.0,
            "recent_contents": [c.to_dict() for c in recent],
            "salient_items": [c.to_dict() for c in self.stream.get_salient()],
            "background_processes": self._background_processes,
        }

    def state_of_consciousness(self) -> str:
        """Beschreibt den aktuellen Bewusstseinszustand."""
        focus = self.stream.current_focus
        recent = self.stream.get_recent(3)

        parts = [f"Bewusstseinsebene: {self._current_level.value}"]

        if focus:
            parts.append(f"Fokus auf: {focus.target} (Intensität: {focus.intensity:.1f})")

        if recent:
            parts.append(f"Letzte Inhalte: {len(recent)} im Strom")

        calibration = self.metacognition.get_calibration_score()
        parts.append(f"Metakognitive Kalibrierung: {calibration:.1%}")

        return " | ".join(parts)
