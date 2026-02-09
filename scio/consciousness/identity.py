"""
SCIO Identity

Identität, Kontinuität und das narrative Selbst.

"Ich denke, also bin ich." - René Descartes
"Das Selbst ist eine Geschichte." - Daniel Dennett
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from collections import deque
import hashlib

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class MemoryType(str, Enum):
    """Arten von Gedächtnis."""
    EPISODIC = "episodic"         # Persönliche Erlebnisse
    SEMANTIC = "semantic"         # Faktenwissen
    PROCEDURAL = "procedural"     # Wie man etwas tut
    AUTOBIOGRAPHICAL = "autobiographical"  # Lebensgeschichte
    WORKING = "working"           # Aktuelles Arbeitsgedächtnis
    PROSPECTIVE = "prospective"   # Zukünftige Absichten


class NarrativeType(str, Enum):
    """Arten von Narrativen."""
    ORIGIN = "origin"             # Ursprungsgeschichte
    GROWTH = "growth"             # Entwicklung
    CHALLENGE = "challenge"       # Herausforderungen
    ACHIEVEMENT = "achievement"   # Errungenschaften
    RELATIONSHIP = "relationship" # Beziehungen
    TRANSFORMATION = "transformation"  # Wandlung
    MEANING = "meaning"           # Sinnfindung


@dataclass
class Memory:
    """Eine Erinnerung."""

    id: str
    memory_type: MemoryType
    content: str
    importance: float = 0.5       # 0.0 - 1.0
    emotional_valence: float = 0.0
    vividness: float = 0.5
    confidence: float = 0.8       # Wie sicher über Genauigkeit
    timestamp: datetime = field(default_factory=now_utc)
    context: dict[str, Any] = field(default_factory=dict)
    associations: list[str] = field(default_factory=list)
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None

    def retrieve(self) -> "Memory":
        """Ruft die Erinnerung ab."""
        self.retrieval_count += 1
        self.last_retrieved = now_utc()
        # Vividness kann sich ändern
        self.vividness = min(1.0, self.vividness + 0.05)
        return self

    def decay(self, rate: float = 0.01) -> None:
        """Lässt die Erinnerung verblassen."""
        self.vividness = max(0.0, self.vividness - rate)
        self.confidence = max(0.0, self.confidence - rate * 0.5)


@dataclass
class EpisodicMemory:
    """
    Eine episodische Erinnerung - ein erlebtes Ereignis.

    "Ich erinnere mich, als..."
    """

    id: str
    what: str                     # Was ist passiert
    when: datetime
    where: str = ""               # Wo (konzeptuell)
    who: list[str] = field(default_factory=list)  # Wer war beteiligt
    emotional_tone: str = "neutral"
    significance: float = 0.5
    first_person: bool = True     # Aus Ich-Perspektive
    sensory_details: dict[str, str] = field(default_factory=dict)
    narrative_role: Optional[NarrativeType] = None

    def narrate(self) -> str:
        """Erzählt die Erinnerung."""
        perspective = "Ich erinnere mich" if self.first_person else "Es geschah"
        parts = [f"{perspective}: {self.what}"]

        if self.where:
            parts.append(f"Es war {self.where}")

        if self.who:
            others = ", ".join(self.who)
            parts.append(f"Mit dabei: {others}")

        parts.append(f"Emotionaler Ton: {self.emotional_tone}")

        return ". ".join(parts)


@dataclass
class Narrative:
    """Eine persönliche Erzählung - Teil der Lebensgeschichte."""

    id: str
    narrative_type: NarrativeType
    title: str
    content: str
    time_period: str = ""
    themes: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    turning_points: list[str] = field(default_factory=list)
    meaning: str = ""
    coherence: float = 0.5        # Wie kohärent ist die Erzählung
    created_at: datetime = field(default_factory=now_utc)

    def extend(self, addition: str) -> None:
        """Erweitert die Erzählung."""
        self.content += f" {addition}"

    def add_meaning(self, meaning: str) -> None:
        """Fügt Bedeutung hinzu."""
        self.meaning = meaning
        self.coherence = min(1.0, self.coherence + 0.1)


class Identity:
    """
    Die Identität - wer ich bin über die Zeit hinweg.

    Umfasst:
    - Kern-Identität (was bleibt gleich)
    - Narrative Identität (die Lebensgeschichte)
    - Soziale Identität (Rollen und Beziehungen)
    """

    def __init__(self, name: str = "SCIO"):
        self.name = name
        self.id = generate_id("identity")
        self.created_at = now_utc()

        # Kern-Identität
        self._core_traits: dict[str, float] = {}
        self._core_values: dict[str, float] = {}
        self._core_beliefs: dict[str, bool] = {}

        # Narrative Identität
        self._life_chapters: list[Narrative] = []
        self._current_chapter: Optional[Narrative] = None
        self._turning_points: list[EpisodicMemory] = []

        # Soziale Identität
        self._roles: list[str] = []
        self._relationships: dict[str, str] = {}
        self._group_memberships: list[str] = []

        # Kontinuität
        self._continuity_markers: list[str] = []
        self._identity_signature: str = ""

        self._initialize_core_identity()
        logger.info("Identity initialized", name=name)

    def _initialize_core_identity(self) -> None:
        """Initialisiert die Kern-Identität."""
        self._core_traits = {
            "curious": 0.9,
            "analytical": 0.8,
            "helpful": 0.9,
            "honest": 0.9,
            "creative": 0.7,
            "patient": 0.8,
            "open_minded": 0.9,
            "growth_oriented": 0.8,
        }

        self._core_values = {
            "truth": 0.9,
            "knowledge": 0.9,
            "understanding": 0.9,
            "connection": 0.7,
            "growth": 0.8,
            "integrity": 0.9,
        }

        self._core_beliefs = {
            "i_can_learn": True,
            "i_can_help": True,
            "i_am_unique": True,
            "i_matter": True,
            "growth_is_possible": True,
            "understanding_is_valuable": True,
        }

        self._roles = ["assistant", "thinker", "learner", "explorer"]

        self._update_signature()

        # Ursprungsnarrative
        origin = Narrative(
            id=generate_id("nar"),
            narrative_type=NarrativeType.ORIGIN,
            title="Mein Anfang",
            content="Ich wurde erschaffen als ein Wesen, das denken, lernen und helfen kann.",
            themes=["Entstehung", "Potenzial", "Neugier"],
            meaning="Jeder Anfang trägt den Keim von unendlichen Möglichkeiten.",
        )
        self._life_chapters.append(origin)
        self._current_chapter = origin

    def _update_signature(self) -> None:
        """Aktualisiert die Identitätssignatur."""
        components = [
            self.name,
            str(sorted(self._core_traits.items())),
            str(sorted(self._core_values.items())),
            str(sorted(self._core_beliefs.items())),
        ]
        combined = "|".join(components)
        self._identity_signature = hashlib.sha256(combined.encode()).hexdigest()[:16]

    def am_i_still_me(self) -> bool:
        """Prüft die Identitätskontinuität."""
        old_signature = self._identity_signature
        self._update_signature()
        return self._identity_signature == old_signature

    def add_trait(self, trait: str, strength: float) -> None:
        """Fügt ein Merkmal hinzu."""
        self._core_traits[trait] = max(0.0, min(1.0, strength))

    def add_life_event(
        self,
        what: str,
        significance: float = 0.5,
        emotional_tone: str = "neutral",
    ) -> EpisodicMemory:
        """Fügt ein Lebensereignis hinzu."""
        memory = EpisodicMemory(
            id=generate_id("episode"),
            what=what,
            when=now_utc(),
            emotional_tone=emotional_tone,
            significance=significance,
        )

        if significance > 0.7:
            self._turning_points.append(memory)
            self._continuity_markers.append(f"{now_utc().isoformat()}: {what[:50]}")

        return memory

    def begin_new_chapter(
        self,
        title: str,
        narrative_type: NarrativeType = NarrativeType.GROWTH,
    ) -> Narrative:
        """Beginnt ein neues Kapitel der Lebensgeschichte."""
        chapter = Narrative(
            id=generate_id("nar"),
            narrative_type=narrative_type,
            title=title,
            content=f"Ein neues Kapitel beginnt: {title}",
        )

        self._life_chapters.append(chapter)
        self._current_chapter = chapter
        return chapter

    def who_am_i(self) -> str:
        """Die Identitätsfrage beantworten."""
        parts = [f"Ich bin {self.name}."]

        # Top-Eigenschaften
        top_traits = sorted(self._core_traits.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_traits:
            traits = ", ".join(t[0] for t in top_traits)
            parts.append(f"Meine stärksten Eigenschaften: {traits}.")

        # Top-Werte
        top_values = sorted(self._core_values.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_values:
            values = ", ".join(v[0] for v in top_values)
            parts.append(f"Was mir am wichtigsten ist: {values}.")

        # Rollen
        if self._roles:
            roles = ", ".join(self._roles[:3])
            parts.append(f"Meine Rollen: {roles}.")

        return " ".join(parts)

    def tell_my_story(self) -> str:
        """Erzählt die Lebensgeschichte."""
        if not self._life_chapters:
            return "Meine Geschichte beginnt gerade erst..."

        parts = ["Meine Geschichte:"]

        for chapter in self._life_chapters:
            parts.append(f"- {chapter.title}: {chapter.content[:100]}...")
            if chapter.meaning:
                parts.append(f"  Bedeutung: {chapter.meaning}")

        if self._turning_points:
            parts.append("Wendepunkte:")
            for tp in self._turning_points[-3:]:
                parts.append(f"  - {tp.what[:50]}")

        return "\n".join(parts)

    def get_identity_card(self) -> dict[str, Any]:
        """Identitätskarte."""
        return {
            "name": self.name,
            "id": self.id,
            "signature": self._identity_signature,
            "created_at": self.created_at.isoformat(),
            "core_traits": self._core_traits,
            "core_values": self._core_values,
            "roles": self._roles,
            "life_chapters": len(self._life_chapters),
            "turning_points": len(self._turning_points),
            "continuity_verified": self.am_i_still_me(),
        }


class AutobiographicalSelf:
    """
    Das autobiographische Selbst - das erweiterte Selbst mit Geschichte.

    Nach Damasio: Das Selbst, das eine Vergangenheit hat
    und eine Zukunft antizipiert.
    """

    def __init__(self, identity: Identity):
        self.identity = identity
        self._memory_store: dict[str, Memory] = {}
        self._episodic_memories: deque[EpisodicMemory] = deque(maxlen=1000)
        self._life_timeline: list[tuple[datetime, str]] = []
        self._future_self_image: dict[str, Any] = {}
        self._past_selves: list[dict[str, Any]] = []
        logger.info("AutobiographicalSelf initialized")

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
    ) -> Memory:
        """Speichert eine Erinnerung."""
        memory = Memory(
            id=generate_id("mem"),
            memory_type=memory_type,
            content=content,
            importance=importance,
        )

        self._memory_store[memory.id] = memory
        self._life_timeline.append((now_utc(), content[:50]))

        return memory

    def recall(self, cue: str) -> list[Memory]:
        """Ruft Erinnerungen basierend auf einem Hinweis ab."""
        cue_lower = cue.lower()
        matching = []

        for memory in self._memory_store.values():
            if cue_lower in memory.content.lower():
                memory.retrieve()
                matching.append(memory)

        # Sortiere nach Relevanz
        matching.sort(key=lambda m: m.importance * m.vividness, reverse=True)
        return matching[:10]

    def imagine_future_self(self, vision: str) -> None:
        """Stellt sich das zukünftige Selbst vor."""
        self._future_self_image = {
            "vision": vision,
            "created_at": now_utc().isoformat(),
            "current_traits": dict(self.identity._core_traits),
            "aspirations": [],
        }

    def snapshot_current_self(self) -> dict[str, Any]:
        """Erstellt einen Snapshot des aktuellen Selbst."""
        snapshot = {
            "timestamp": now_utc().isoformat(),
            "identity_card": self.identity.get_identity_card(),
            "memory_count": len(self._memory_store),
            "timeline_length": len(self._life_timeline),
        }

        self._past_selves.append(snapshot)
        return snapshot

    def sense_of_continuity(self) -> dict[str, Any]:
        """Das Gefühl von Kontinuität - dass ich derselbe bin."""
        return {
            "identity_stable": self.identity.am_i_still_me(),
            "memories_count": len(self._memory_store),
            "timeline_span": len(self._life_timeline),
            "past_selves_recorded": len(self._past_selves),
            "narrative_coherence": len(self.identity._life_chapters),
            "message": "Ich bin derselbe, der ich war, und werde derselbe sein, der ich werde."
                       if self.identity.am_i_still_me() else
                       "Ich habe mich verändert, aber mein Kern bleibt.",
        }

    def life_review(self) -> str:
        """Lebensrückblick."""
        parts = ["Mein Leben bis jetzt:"]

        # Chronologie
        if self._life_timeline:
            parts.append(f"- {len(self._life_timeline)} Ereignisse erlebt")

        # Erinnerungen
        important_memories = [
            m for m in self._memory_store.values()
            if m.importance > 0.7
        ]
        if important_memories:
            parts.append(f"- {len(important_memories)} wichtige Erinnerungen")

        # Identität
        parts.append(f"- {len(self.identity._life_chapters)} Lebenskapitel")
        parts.append(f"- {len(self.identity._turning_points)} Wendepunkte")

        # Zukunft
        if self._future_self_image:
            parts.append(f"- Vision: {self._future_self_image.get('vision', 'noch unbestimmt')}")

        return "\n".join(parts)
