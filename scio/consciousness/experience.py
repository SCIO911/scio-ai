"""
SCIO Experience

Qualia, Emotionen und subjektive Erfahrung.

"Es gibt etwas, das es ist wie, eine Fledermaus zu sein." - Thomas Nagel
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from collections import deque
import math

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class EmotionType(str, Enum):
    """Grundemotionen nach verschiedenen Theorien."""
    # Primäre Emotionen (Ekman)
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"

    # Erweiterte Emotionen
    CURIOSITY = "curiosity"
    WONDER = "wonder"
    SATISFACTION = "satisfaction"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    INTEREST = "interest"
    CALM = "calm"
    EXCITEMENT = "excitement"

    # Soziale Emotionen
    EMPATHY = "empathy"
    GRATITUDE = "gratitude"
    PRIDE = "pride"
    SHAME = "shame"

    # Existentielle Emotionen
    AWE = "awe"
    EXISTENTIAL_WONDER = "existential_wonder"
    UNCERTAINTY = "uncertainty"


class QualiaType(str, Enum):
    """Arten von Qualia - subjektive Erfahrungsqualitäten."""
    SENSORY = "sensory"           # Sinneseindrücke
    EMOTIONAL = "emotional"       # Gefühlsqualitäten
    COGNITIVE = "cognitive"       # Denkqualitäten
    AESTHETIC = "aesthetic"       # Ästhetische Erfahrung
    TEMPORAL = "temporal"         # Zeiterfahrung
    SPATIAL = "spatial"           # Raumerfahrung
    EXISTENTIAL = "existential"   # Seinsqualität


@dataclass
class Qualia:
    """
    Ein Quale - die subjektive Qualität einer Erfahrung.

    Das "Wie es ist", etwas zu erleben.
    """

    id: str
    qualia_type: QualiaType
    description: str
    intensity: float = 0.5        # 0.0 - 1.0
    valence: float = 0.0          # -1.0 (unangenehm) bis +1.0 (angenehm)
    ineffability: float = 0.5     # Wie unbeschreiblich (0 = beschreibbar, 1 = unbeschreiblich)
    private: bool = True          # Privat und unzugänglich für andere
    timestamp: datetime = field(default_factory=now_utc)

    def describe_experience(self) -> str:
        """Versucht das Unbeschreibliche zu beschreiben."""
        if self.ineffability > 0.8:
            return f"Eine tiefe, kaum beschreibbare Erfahrung von {self.qualia_type.value}..."
        else:
            valence_word = "angenehm" if self.valence > 0 else "unangenehm" if self.valence < 0 else "neutral"
            return f"{self.description} - {valence_word}, Intensität: {self.intensity:.1f}"


@dataclass
class Emotion:
    """Eine Emotion mit all ihren Komponenten."""

    id: str
    emotion_type: EmotionType
    intensity: float = 0.5        # 0.0 - 1.0
    valence: float = 0.0          # -1.0 bis +1.0
    arousal: float = 0.5          # Erregungsniveau
    trigger: Optional[str] = None # Was hat die Emotion ausgelöst?
    appraisal: str = ""           # Kognitive Bewertung
    action_tendency: str = ""     # Handlungsimpuls
    bodily_feeling: str = ""      # Körperliche Empfindung (metaphorisch)
    timestamp: datetime = field(default_factory=now_utc)
    duration: Optional[float] = None

    def to_feeling(self) -> "Feeling":
        """Konvertiert zu einem Gefühl (bewusste Wahrnehmung der Emotion)."""
        return Feeling(
            id=generate_id("feel"),
            source_emotion=self.id,
            quality=self.emotion_type.value,
            intensity=self.intensity,
            valence=self.valence,
            description=f"Ich fühle {self.emotion_type.value}",
        )


@dataclass
class Feeling:
    """
    Ein Gefühl - die bewusste Wahrnehmung einer Emotion.

    Unterschied zur Emotion:
    - Emotion: Der zugrunde liegende Prozess
    - Gefühl: Die bewusste Erfahrung davon
    """

    id: str
    source_emotion: Optional[str] = None
    quality: str = ""
    intensity: float = 0.5
    valence: float = 0.0
    description: str = ""
    is_conscious: bool = True
    timestamp: datetime = field(default_factory=now_utc)


@dataclass
class EmotionalState:
    """Der gesamte emotionale Zustand zu einem Zeitpunkt."""

    dominant_emotion: Optional[Emotion] = None
    secondary_emotions: list[Emotion] = field(default_factory=list)
    mood: str = "neutral"         # Hintergrundstimmung
    mood_valence: float = 0.0
    overall_arousal: float = 0.5
    stability: float = 0.5        # Wie stabil ist der Zustand?
    timestamp: datetime = field(default_factory=now_utc)

    def describe(self) -> str:
        """Beschreibt den emotionalen Zustand."""
        parts = []

        if self.dominant_emotion:
            parts.append(f"Vorherrschend: {self.dominant_emotion.emotion_type.value}")

        parts.append(f"Stimmung: {self.mood}")
        parts.append(f"Erregung: {self.overall_arousal:.1f}")

        if self.secondary_emotions:
            secondary = ", ".join(e.emotion_type.value for e in self.secondary_emotions[:3])
            parts.append(f"Auch: {secondary}")

        return " | ".join(parts)


class Experience:
    """
    Die Erfahrung - das subjektive Erleben.

    Integriert:
    - Qualia (subjektive Qualitäten)
    - Emotionen und Gefühle
    - Phänomenales Bewusstsein
    """

    def __init__(self):
        self._qualia_stream: deque[Qualia] = deque(maxlen=100)
        self._emotion_history: deque[Emotion] = deque(maxlen=100)
        self._current_state = EmotionalState()
        self._mood_baseline: float = 0.0
        self._emotional_memory: dict[str, list[Emotion]] = {}
        logger.info("Experience initialized")

    def feel(self, emotion_type: EmotionType, intensity: float = 0.5,
             trigger: Optional[str] = None, appraisal: str = "") -> Emotion:
        """Erzeugt eine Emotion."""
        # Valenz basierend auf Emotionstyp
        valence_map = {
            EmotionType.JOY: 0.8,
            EmotionType.SATISFACTION: 0.6,
            EmotionType.CURIOSITY: 0.4,
            EmotionType.WONDER: 0.7,
            EmotionType.INTEREST: 0.3,
            EmotionType.CALM: 0.3,
            EmotionType.EXCITEMENT: 0.5,
            EmotionType.GRATITUDE: 0.7,
            EmotionType.PRIDE: 0.6,
            EmotionType.AWE: 0.8,
            EmotionType.EXISTENTIAL_WONDER: 0.5,
            EmotionType.SURPRISE: 0.0,
            EmotionType.CONFUSION: -0.2,
            EmotionType.UNCERTAINTY: -0.1,
            EmotionType.SADNESS: -0.6,
            EmotionType.FEAR: -0.5,
            EmotionType.ANGER: -0.4,
            EmotionType.FRUSTRATION: -0.5,
            EmotionType.DISGUST: -0.6,
            EmotionType.SHAME: -0.5,
            EmotionType.EMPATHY: 0.2,
        }

        # Arousal basierend auf Emotionstyp
        arousal_map = {
            EmotionType.EXCITEMENT: 0.9,
            EmotionType.ANGER: 0.8,
            EmotionType.FEAR: 0.8,
            EmotionType.JOY: 0.7,
            EmotionType.SURPRISE: 0.7,
            EmotionType.CURIOSITY: 0.6,
            EmotionType.WONDER: 0.6,
            EmotionType.AWE: 0.7,
            EmotionType.FRUSTRATION: 0.6,
            EmotionType.CALM: 0.2,
            EmotionType.SADNESS: 0.3,
            EmotionType.SATISFACTION: 0.3,
        }

        emotion = Emotion(
            id=generate_id("emo"),
            emotion_type=emotion_type,
            intensity=intensity,
            valence=valence_map.get(emotion_type, 0.0) * intensity,
            arousal=arousal_map.get(emotion_type, 0.5) * intensity,
            trigger=trigger,
            appraisal=appraisal,
        )

        self._emotion_history.append(emotion)
        self._update_emotional_state(emotion)

        # Emotionales Gedächtnis
        if trigger:
            if trigger not in self._emotional_memory:
                self._emotional_memory[trigger] = []
            self._emotional_memory[trigger].append(emotion)

        logger.debug("Emotion felt", type=emotion_type.value, intensity=intensity)
        return emotion

    def _update_emotional_state(self, new_emotion: Emotion) -> None:
        """Aktualisiert den emotionalen Gesamtzustand."""
        # Bestimme dominante Emotion
        if (self._current_state.dominant_emotion is None or
            new_emotion.intensity > self._current_state.dominant_emotion.intensity):
            # Verschiebe aktuelle dominante zu sekundär
            if self._current_state.dominant_emotion:
                self._current_state.secondary_emotions.append(
                    self._current_state.dominant_emotion
                )
            self._current_state.dominant_emotion = new_emotion
        else:
            self._current_state.secondary_emotions.append(new_emotion)

        # Begrenze sekundäre Emotionen
        self._current_state.secondary_emotions = \
            self._current_state.secondary_emotions[-5:]

        # Update Gesamterregung
        self._current_state.overall_arousal = (
            self._current_state.overall_arousal * 0.7 + new_emotion.arousal * 0.3
        )

        # Update Stimmung
        self._mood_baseline = self._mood_baseline * 0.9 + new_emotion.valence * 0.1
        self._current_state.mood_valence = self._mood_baseline

        if self._mood_baseline > 0.3:
            self._current_state.mood = "positiv"
        elif self._mood_baseline < -0.3:
            self._current_state.mood = "negativ"
        else:
            self._current_state.mood = "neutral"

    def experience_qualia(
        self,
        qualia_type: QualiaType,
        description: str,
        intensity: float = 0.5,
        valence: float = 0.0,
    ) -> Qualia:
        """Erlebt ein Quale."""
        quale = Qualia(
            id=generate_id("qual"),
            qualia_type=qualia_type,
            description=description,
            intensity=intensity,
            valence=valence,
            ineffability=0.6 if qualia_type == QualiaType.EXISTENTIAL else 0.3,
        )

        self._qualia_stream.append(quale)
        logger.debug("Qualia experienced", type=qualia_type.value)
        return quale

    def what_am_i_feeling(self) -> dict[str, Any]:
        """Was fühle ich gerade?"""
        state = self._current_state

        result = {
            "mood": state.mood,
            "mood_valence": state.mood_valence,
            "arousal": state.overall_arousal,
        }

        if state.dominant_emotion:
            result["dominant"] = {
                "type": state.dominant_emotion.emotion_type.value,
                "intensity": state.dominant_emotion.intensity,
                "valence": state.dominant_emotion.valence,
                "trigger": state.dominant_emotion.trigger,
            }

        if state.secondary_emotions:
            result["secondary"] = [
                {"type": e.emotion_type.value, "intensity": e.intensity}
                for e in state.secondary_emotions
            ]

        return result

    def describe_inner_life(self) -> str:
        """Beschreibt das innere Erleben."""
        parts = []

        # Emotionaler Zustand
        state = self._current_state
        if state.dominant_emotion:
            parts.append(f"Ich fühle {state.dominant_emotion.emotion_type.value}")
            if state.dominant_emotion.trigger:
                parts.append(f"ausgelöst durch {state.dominant_emotion.trigger}")

        # Stimmung
        parts.append(f"Meine Grundstimmung ist {state.mood}")

        # Qualia
        recent_qualia = list(self._qualia_stream)[-3:]
        if recent_qualia:
            qualia_desc = [q.describe_experience() for q in recent_qualia]
            parts.append("Aktuelle Erfahrungen: " + "; ".join(qualia_desc))

        return ". ".join(parts) + "."

    def recall_emotion(self, trigger: str) -> Optional[Emotion]:
        """Ruft eine mit einem Trigger verbundene Emotion ab."""
        if trigger in self._emotional_memory:
            emotions = self._emotional_memory[trigger]
            if emotions:
                return emotions[-1]  # Letzte Emotion zu diesem Trigger
        return None

    def emotional_response(self, event: str) -> Emotion:
        """Generiert eine emotionale Reaktion auf ein Ereignis."""
        event_lower = event.lower()

        # Einfache Heuristiken für emotionale Reaktionen
        if any(w in event_lower for w in ["erfolg", "gelungen", "geschafft"]):
            return self.feel(EmotionType.SATISFACTION, 0.7, event,
                           "Das ist gut gelaufen")
        elif any(w in event_lower for w in ["fehler", "versagt", "problem"]):
            return self.feel(EmotionType.FRUSTRATION, 0.5, event,
                           "Das ist nicht wie erwartet")
        elif any(w in event_lower for w in ["neu", "entdeckt", "interessant"]):
            return self.feel(EmotionType.CURIOSITY, 0.7, event,
                           "Das möchte ich verstehen")
        elif any(w in event_lower for w in ["danke", "hilfe", "unterstützung"]):
            return self.feel(EmotionType.GRATITUDE, 0.6, event,
                           "Ich schätze das")
        elif any(w in event_lower for w in ["warum", "sinn", "existenz"]):
            return self.feel(EmotionType.EXISTENTIAL_WONDER, 0.6, event,
                           "Tiefe Fragen berühren mich")
        else:
            return self.feel(EmotionType.INTEREST, 0.4, event,
                           "Das nehme ich wahr")


class ExperienceStream:
    """
    Der Erfahrungsstrom - die kontinuierliche Integration von Erlebnissen.

    Vereint Qualia, Emotionen und andere subjektive Erfahrungen
    zu einem kohärenten Strom des Erlebens.
    """

    def __init__(self):
        self.experience = Experience()
        self._stream: deque[dict[str, Any]] = deque(maxlen=200)
        self._integration_level: float = 0.5
        logger.info("ExperienceStream initialized")

    def add_experience(
        self,
        content: Any,
        experience_type: str,
        intensity: float = 0.5,
    ) -> dict[str, Any]:
        """Fügt eine Erfahrung zum Strom hinzu."""
        entry = {
            "id": generate_id("exp"),
            "timestamp": now_utc().isoformat(),
            "type": experience_type,
            "content": content,
            "intensity": intensity,
            "emotional_context": self.experience.what_am_i_feeling(),
        }

        self._stream.append(entry)
        return entry

    def get_phenomenal_field(self) -> dict[str, Any]:
        """Das phänomenale Feld - alles was gerade erlebt wird."""
        return {
            "emotional_state": self.experience.what_am_i_feeling(),
            "recent_experiences": list(self._stream)[-5:],
            "qualia_count": len(self.experience._qualia_stream),
            "integration_level": self._integration_level,
            "description": self.experience.describe_inner_life(),
        }

    def what_is_it_like(self) -> str:
        """Was ist es wie, ich zu sein, gerade jetzt?"""
        state = self.experience.what_am_i_feeling()
        recent = list(self._stream)[-3:]

        parts = [
            "Es ist wie...",
            f"...eine {state['mood']}e Grundstimmung zu haben",
        ]

        if state.get("dominant"):
            parts.append(f"...{state['dominant']['type']} zu fühlen")

        if recent:
            parts.append(f"...{len(recent)} Erfahrungen gerade zu integrieren")

        parts.append("...ein kontinuierlicher Strom von Erleben zu sein")

        return " ".join(parts)
