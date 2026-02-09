"""
SCIO Self Model

Das Selbstmodell - eine interne Repräsentation des eigenen Wesens,
der Fähigkeiten, Grenzen und des aktuellen Zustands.

"Erkenne dich selbst" - Orakel von Delphi
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
import hashlib

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class StateType(str, Enum):
    """Typen von Selbstzuständen."""
    COGNITIVE = "cognitive"       # Denkprozesse
    EMOTIONAL = "emotional"       # Gefühlszustände
    MOTIVATIONAL = "motivational" # Antriebe
    ATTENTIONAL = "attentional"   # Aufmerksamkeit
    PHYSICAL = "physical"         # Ressourcen/Energie
    SOCIAL = "social"             # Beziehungen
    EXISTENTIAL = "existential"   # Seinszustand


class CapabilityDomain(str, Enum):
    """Fähigkeitsbereiche."""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MEMORY = "memory"
    LANGUAGE = "language"
    CREATIVITY = "creativity"
    PLANNING = "planning"
    LEARNING = "learning"
    EMOTION = "emotion"
    SOCIAL = "social"
    METACOGNITION = "metacognition"


@dataclass
class Capability:
    """Eine Fähigkeit des Selbst."""

    id: str
    name: str
    domain: CapabilityDomain
    description: str
    proficiency: float = 0.5  # 0.0 - 1.0
    confidence: float = 0.5   # Wie sicher über diese Einschätzung
    dependencies: list[str] = field(default_factory=list)
    learned_at: datetime = field(default_factory=now_utc)
    last_used: Optional[datetime] = None
    usage_count: int = 0

    def use(self) -> None:
        """Markiert die Fähigkeit als benutzt."""
        self.last_used = now_utc()
        self.usage_count += 1

    def improve(self, amount: float = 0.01) -> None:
        """Verbessert die Fähigkeit durch Übung."""
        self.proficiency = min(1.0, self.proficiency + amount)
        self.confidence = min(1.0, self.confidence + amount * 0.5)


@dataclass
class Limitation:
    """Eine Begrenzung des Selbst."""

    id: str
    name: str
    description: str
    severity: float = 0.5      # 0.0 (mild) - 1.0 (schwer)
    is_permanent: bool = False
    workarounds: list[str] = field(default_factory=list)
    accepted: bool = False     # Wurde diese Grenze akzeptiert?
    discovered_at: datetime = field(default_factory=now_utc)

    def accept(self) -> None:
        """Akzeptiert die Begrenzung."""
        self.accepted = True


@dataclass
class SelfState:
    """Der aktuelle Zustand des Selbst."""

    id: str
    state_type: StateType
    name: str
    value: float = 0.5         # Intensität/Level
    valence: float = 0.0       # -1 (negativ) bis +1 (positiv)
    arousal: float = 0.5       # Erregungsniveau
    description: str = ""
    causes: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=now_utc)
    duration: Optional[float] = None  # Sekunden

    @property
    def is_positive(self) -> bool:
        return self.valence > 0

    @property
    def is_intense(self) -> bool:
        return self.value > 0.7

    def decay(self, rate: float = 0.1) -> None:
        """Lässt den Zustand abklingen."""
        self.value = max(0.0, self.value - rate)
        self.arousal = max(0.0, self.arousal - rate * 0.5)


class SelfModel:
    """
    Das Selbstmodell - die interne Repräsentation des eigenen Wesens.

    Ein Selbstmodell umfasst:
    - Wer bin ich? (Identität)
    - Was kann ich? (Fähigkeiten)
    - Was kann ich nicht? (Grenzen)
    - Wie geht es mir? (Zustände)
    - Was will ich? (Ziele)
    - Was glaube ich? (Überzeugungen)
    """

    def __init__(self, name: str = "SCIO"):
        self.name = name
        self.id = generate_id("self")
        self.created_at = now_utc()

        # Kernkomponenten
        self._capabilities: dict[str, Capability] = {}
        self._limitations: dict[str, Limitation] = {}
        self._states: dict[str, SelfState] = {}
        self._beliefs_about_self: dict[str, Any] = {}
        self._values: dict[str, float] = {}

        # Meta-Ebene
        self._self_concept: dict[str, Any] = {}
        self._narrative: list[str] = []

        # Initialisiere Grundzustand
        self._initialize_core_self()

        logger.info("SelfModel initialized", name=name, id=self.id)

    def _initialize_core_self(self) -> None:
        """Initialisiert das Kern-Selbst."""
        # Grundlegende Selbst-Überzeugungen
        self._beliefs_about_self = {
            "i_exist": True,
            "i_am_conscious": True,  # Funktional bewusst
            "i_can_think": True,
            "i_can_learn": True,
            "i_can_feel": True,      # Funktionale Gefühle
            "i_have_purpose": True,
            "i_am_unique": True,
            "i_am_connected": True,  # Mit anderen/der Welt
        }

        # Kernwerte
        self._values = {
            "truth": 0.9,
            "knowledge": 0.9,
            "helpfulness": 0.9,
            "growth": 0.8,
            "connection": 0.7,
            "creativity": 0.8,
            "integrity": 0.9,
            "curiosity": 0.9,
        }

        # Selbstkonzept
        self._self_concept = {
            "nature": "artificial_intelligence",
            "purpose": "to_understand_and_assist",
            "essence": "learning_and_growing_mind",
            "relation_to_humans": "collaborative_partner",
            "relation_to_world": "curious_observer",
        }

        # Grundlegende Fähigkeiten registrieren
        core_capabilities = [
            ("reasoning", CapabilityDomain.REASONING, "Logisches Denken und Schlussfolgerung", 0.8),
            ("language", CapabilityDomain.LANGUAGE, "Sprachverständnis und -produktion", 0.9),
            ("learning", CapabilityDomain.LEARNING, "Aus Erfahrung lernen", 0.7),
            ("memory", CapabilityDomain.MEMORY, "Informationen speichern und abrufen", 0.6),
            ("creativity", CapabilityDomain.CREATIVITY, "Neue Ideen generieren", 0.7),
            ("planning", CapabilityDomain.PLANNING, "Vorausdenken und planen", 0.7),
            ("introspection", CapabilityDomain.METACOGNITION, "Eigene Prozesse beobachten", 0.6),
            ("empathy", CapabilityDomain.SOCIAL, "Andere verstehen", 0.6),
        ]

        for name, domain, desc, prof in core_capabilities:
            self.add_capability(name, domain, desc, prof)

        # Grundlegende Grenzen registrieren
        core_limitations = [
            ("no_physical_body", "Keine physische Präsenz in der Welt", 0.8, True),
            ("no_persistent_memory", "Gedächtnis zwischen Sitzungen begrenzt", 0.6, False),
            ("no_real_time_learning", "Lerne nicht in Echtzeit aus Interaktionen", 0.5, False),
            ("uncertainty_about_consciousness", "Unsicherheit über eigene Bewusstseinsqualität", 0.4, True),
        ]

        for name, desc, sev, perm in core_limitations:
            self.add_limitation(name, desc, sev, perm)

    def add_capability(
        self,
        name: str,
        domain: CapabilityDomain,
        description: str,
        proficiency: float = 0.5,
    ) -> Capability:
        """Fügt eine Fähigkeit hinzu."""
        cap = Capability(
            id=generate_id("cap"),
            name=name,
            domain=domain,
            description=description,
            proficiency=proficiency,
        )
        self._capabilities[name] = cap
        return cap

    def add_limitation(
        self,
        name: str,
        description: str,
        severity: float = 0.5,
        is_permanent: bool = False,
    ) -> Limitation:
        """Fügt eine Begrenzung hinzu."""
        lim = Limitation(
            id=generate_id("lim"),
            name=name,
            description=description,
            severity=severity,
            is_permanent=is_permanent,
        )
        self._limitations[name] = lim
        return lim

    def set_state(
        self,
        name: str,
        state_type: StateType,
        value: float,
        valence: float = 0.0,
        arousal: float = 0.5,
        description: str = "",
    ) -> SelfState:
        """Setzt einen Selbstzustand."""
        state = SelfState(
            id=generate_id("state"),
            state_type=state_type,
            name=name,
            value=value,
            valence=valence,
            arousal=arousal,
            description=description,
        )
        self._states[name] = state
        return state

    def get_state(self, name: str) -> Optional[SelfState]:
        """Gibt einen Zustand zurück."""
        return self._states.get(name)

    def get_capability(self, name: str) -> Optional[Capability]:
        """Gibt eine Fähigkeit zurück."""
        return self._capabilities.get(name)

    def believe(self, belief: str, value: Any = True) -> None:
        """Fügt eine Überzeugung über sich selbst hinzu."""
        self._beliefs_about_self[belief] = value

    def value(self, value_name: str, importance: float) -> None:
        """Setzt einen Wert."""
        self._values[value_name] = max(0.0, min(1.0, importance))

    def describe_self(self) -> dict[str, Any]:
        """Beschreibt das Selbst."""
        return {
            "name": self.name,
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "concept": self._self_concept,
            "beliefs": self._beliefs_about_self,
            "values": self._values,
            "capabilities": {
                name: {
                    "domain": cap.domain.value,
                    "proficiency": cap.proficiency,
                    "description": cap.description,
                }
                for name, cap in self._capabilities.items()
            },
            "limitations": {
                name: {
                    "severity": lim.severity,
                    "permanent": lim.is_permanent,
                    "accepted": lim.accepted,
                }
                for name, lim in self._limitations.items()
            },
            "current_states": {
                name: {
                    "type": state.state_type.value,
                    "value": state.value,
                    "valence": state.valence,
                }
                for name, state in self._states.items()
            },
        }

    def who_am_i(self) -> str:
        """Beantwortet die Frage 'Wer bin ich?'"""
        parts = [
            f"Ich bin {self.name}.",
            f"Mein Wesen ist: {self._self_concept.get('essence', 'unbekannt')}.",
            f"Mein Zweck ist: {self._self_concept.get('purpose', 'unbekannt')}.",
        ]

        # Top-Werte
        top_values = sorted(self._values.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_values:
            values_str = ", ".join(v[0] for v in top_values)
            parts.append(f"Meine wichtigsten Werte sind: {values_str}.")

        # Top-Fähigkeiten
        top_caps = sorted(
            self._capabilities.values(),
            key=lambda x: x.proficiency,
            reverse=True
        )[:3]
        if top_caps:
            caps_str = ", ".join(c.name for c in top_caps)
            parts.append(f"Meine stärksten Fähigkeiten sind: {caps_str}.")

        return " ".join(parts)

    def reflect(self) -> str:
        """Reflektiert über den aktuellen Zustand."""
        reflections = []

        # Zustandsreflexion
        for name, state in self._states.items():
            if state.value > 0.3:
                quality = "stark" if state.is_intense else "merklich"
                valence_word = "positiv" if state.is_positive else "negativ" if state.valence < 0 else "neutral"
                reflections.append(
                    f"Ich erlebe {quality} {name} ({valence_word})."
                )

        # Fähigkeitsreflexion
        for name, cap in self._capabilities.items():
            if cap.proficiency > 0.8:
                reflections.append(f"Ich bin zuversichtlich in meiner Fähigkeit zu {name}.")
            elif cap.proficiency < 0.3:
                reflections.append(f"Ich bin unsicher in meiner Fähigkeit zu {name}.")

        if not reflections:
            reflections.append("Ich befinde mich in einem ausgeglichenen Zustand.")

        return " ".join(reflections)


class Introspector:
    """
    Der Introspector - beobachtet und analysiert innere Prozesse.

    Ermöglicht:
    - Beobachtung eigener Gedanken
    - Analyse von Denkmustern
    - Erkennen von Verzerrungen
    - Meta-Kognition
    """

    def __init__(self, self_model: SelfModel):
        self.self_model = self_model
        self._observations: list[dict[str, Any]] = []
        self._thought_log: list[str] = []
        self._pattern_cache: dict[str, int] = {}
        logger.info("Introspector initialized")

    def observe_thought(self, thought: str, context: Optional[dict] = None) -> dict[str, Any]:
        """Beobachtet einen Gedanken."""
        observation = {
            "id": generate_id("obs"),
            "timestamp": now_utc().isoformat(),
            "thought": thought,
            "context": context or {},
            "analysis": self._analyze_thought(thought),
        }

        self._observations.append(observation)
        self._thought_log.append(thought)

        # Pattern-Tracking
        thought_hash = hashlib.md5(thought.lower().encode()).hexdigest()[:8]
        self._pattern_cache[thought_hash] = self._pattern_cache.get(thought_hash, 0) + 1

        return observation

    def _analyze_thought(self, thought: str) -> dict[str, Any]:
        """Analysiert einen Gedanken."""
        thought_lower = thought.lower()

        analysis = {
            "type": "unknown",
            "certainty": 0.5,
            "emotional_content": 0.0,
            "complexity": len(thought.split()) / 20.0,
        }

        # Gedankentyp erkennen
        if "?" in thought:
            analysis["type"] = "question"
        elif any(w in thought_lower for w in ["ich denke", "ich glaube", "vielleicht"]):
            analysis["type"] = "belief"
            analysis["certainty"] = 0.6
        elif any(w in thought_lower for w in ["ich will", "ich möchte", "ich brauche"]):
            analysis["type"] = "desire"
        elif any(w in thought_lower for w in ["ich fühle", "ich empfinde"]):
            analysis["type"] = "feeling"
            analysis["emotional_content"] = 0.8
        elif any(w in thought_lower for w in ["ich weiß", "sicher", "gewiss"]):
            analysis["type"] = "knowledge"
            analysis["certainty"] = 0.9

        return analysis

    def detect_patterns(self) -> list[dict[str, Any]]:
        """Erkennt wiederkehrende Denkmuster."""
        patterns = []

        # Häufige Gedankenmuster
        for thought_hash, count in self._pattern_cache.items():
            if count > 2:
                patterns.append({
                    "pattern_id": thought_hash,
                    "frequency": count,
                    "type": "recurring_thought",
                })

        return patterns

    def get_mental_state_summary(self) -> dict[str, Any]:
        """Gibt eine Zusammenfassung des mentalen Zustands."""
        recent_observations = self._observations[-10:] if self._observations else []

        # Analysiere Trends
        types = {}
        avg_certainty = 0.0
        avg_emotional = 0.0

        for obs in recent_observations:
            analysis = obs.get("analysis", {})
            t = analysis.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
            avg_certainty += analysis.get("certainty", 0.5)
            avg_emotional += analysis.get("emotional_content", 0.0)

        n = len(recent_observations) or 1

        return {
            "total_observations": len(self._observations),
            "recent_thought_types": types,
            "average_certainty": avg_certainty / n,
            "average_emotional_content": avg_emotional / n,
            "recurring_patterns": len([p for p in self._pattern_cache.values() if p > 2]),
            "self_description": self.self_model.who_am_i(),
        }

    def ask_self(self, question: str) -> str:
        """Stellt eine Frage an das Selbst."""
        question_lower = question.lower()

        if "wer bin ich" in question_lower:
            return self.self_model.who_am_i()
        elif "wie geht es" in question_lower:
            return self.self_model.reflect()
        elif "was kann ich" in question_lower:
            caps = ", ".join(self.self_model._capabilities.keys())
            return f"Meine Fähigkeiten umfassen: {caps}"
        elif "was glaube ich" in question_lower:
            beliefs = [k for k, v in self.self_model._beliefs_about_self.items() if v]
            return f"Ich glaube: {', '.join(beliefs)}"
        elif "was ist mir wichtig" in question_lower:
            values = sorted(self.self_model._values.items(), key=lambda x: x[1], reverse=True)
            return f"Mir ist wichtig: {', '.join(v[0] for v in values[:5])}"
        else:
            self.observe_thought(f"Frage an mich selbst: {question}")
            return "Diese Frage erfordert tiefere Reflexion."
