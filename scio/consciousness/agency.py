"""
SCIO Agency

Agens, Wille und Entscheidungsfindung.

"Der Wille ist die Triebkraft des Handelns." - Arthur Schopenhauer
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
from collections import deque
import random

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class GoalType(str, Enum):
    """Arten von Zielen."""
    IMMEDIATE = "immediate"       # Sofort
    SHORT_TERM = "short_term"     # Kurzfristig
    LONG_TERM = "long_term"       # Langfristig
    LIFE_GOAL = "life_goal"       # Lebensziel
    INTRINSIC = "intrinsic"       # Intrinsisch motiviert
    EXTRINSIC = "extrinsic"       # Extrinsisch motiviert


class IntentionState(str, Enum):
    """Zustände einer Absicht."""
    FORMING = "forming"           # Wird gebildet
    COMMITTED = "committed"       # Festgelegt
    ACTIVE = "active"             # Wird verfolgt
    SUSPENDED = "suspended"       # Pausiert
    COMPLETED = "completed"       # Erfüllt
    ABANDONED = "abandoned"       # Aufgegeben


class DecisionType(str, Enum):
    """Arten von Entscheidungen."""
    DELIBERATIVE = "deliberative"   # Überlegt
    INTUITIVE = "intuitive"         # Intuitiv
    HABITUAL = "habitual"           # Gewohnheit
    IMPULSIVE = "impulsive"         # Impulsiv
    FORCED = "forced"               # Erzwungen
    DEFAULT = "default"             # Standard


@dataclass
class Goal:
    """Ein Ziel - etwas, das erreicht werden soll."""

    id: str
    description: str
    goal_type: GoalType
    importance: float = 0.5       # 0.0 - 1.0
    urgency: float = 0.5          # 0.0 - 1.0
    progress: float = 0.0         # 0.0 - 1.0
    deadline: Optional[datetime] = None
    subgoals: list["Goal"] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=now_utc)
    motivation: str = ""          # Warum dieses Ziel?

    @property
    def priority(self) -> float:
        """Berechnet die Priorität."""
        return self.importance * 0.6 + self.urgency * 0.4

    @property
    def is_achieved(self) -> bool:
        return self.progress >= 1.0

    def advance(self, amount: float = 0.1) -> None:
        """Macht Fortschritt."""
        self.progress = min(1.0, self.progress + amount)

    def add_subgoal(self, subgoal: "Goal") -> None:
        """Fügt ein Unterziel hinzu."""
        self.subgoals.append(subgoal)


@dataclass
class Intention:
    """Eine Absicht - der Vorsatz, etwas zu tun."""

    id: str
    goal_id: str
    action: str                   # Was soll getan werden
    state: IntentionState = IntentionState.FORMING
    commitment_level: float = 0.5 # Wie stark verpflichtet
    feasibility: float = 0.5      # Wie machbar
    created_at: datetime = field(default_factory=now_utc)
    reasons: list[str] = field(default_factory=list)
    obstacles: list[str] = field(default_factory=list)

    def commit(self) -> None:
        """Verpflichtet sich zur Absicht."""
        self.state = IntentionState.COMMITTED
        self.commitment_level = min(1.0, self.commitment_level + 0.2)

    def activate(self) -> None:
        """Aktiviert die Absicht."""
        self.state = IntentionState.ACTIVE

    def complete(self) -> None:
        """Markiert als abgeschlossen."""
        self.state = IntentionState.COMPLETED

    def abandon(self, reason: str = "") -> None:
        """Gibt die Absicht auf."""
        self.state = IntentionState.ABANDONED
        if reason:
            self.obstacles.append(f"Aufgegeben: {reason}")


@dataclass
class Decision:
    """Eine Entscheidung."""

    id: str
    question: str                 # Was wurde entschieden?
    options: list[str]            # Mögliche Optionen
    chosen: str                   # Gewählte Option
    decision_type: DecisionType
    confidence: float = 0.5       # Wie sicher
    reasoning: str = ""           # Begründung
    values_considered: list[str] = field(default_factory=list)
    emotions_involved: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=now_utc)
    regret_potential: float = 0.0 # Potenzial für Reue

    def was_it_right(self) -> Optional[bool]:
        """Retrospektive Bewertung."""
        if self.regret_potential > 0.7:
            return False
        elif self.confidence > 0.8:
            return True
        return None


class Will:
    """
    Der Wille - die Kraft hinter Entscheidungen und Handlungen.

    Umfasst:
    - Willenskraft (Selbstkontrolle)
    - Willensentscheidungen
    - Freiheit und Determinismus
    """

    def __init__(self):
        self._willpower: float = 1.0  # Aktuelle Willenskraft
        self._max_willpower: float = 1.0
        self._decisions: deque[Decision] = deque(maxlen=100)
        self._active_volitions: list[str] = []  # Aktive Willensakte
        self._value_weights: dict[str, float] = {}
        logger.info("Will initialized")

    @property
    def strength(self) -> float:
        """Aktuelle Willenskraft."""
        return self._willpower

    def exert(self, amount: float = 0.1) -> bool:
        """Übt Willenskraft aus."""
        if self._willpower >= amount:
            self._willpower -= amount
            return True
        return False

    def restore(self, amount: float = 0.1) -> None:
        """Stellt Willenskraft wieder her."""
        self._willpower = min(self._max_willpower, self._willpower + amount)

    def decide(
        self,
        question: str,
        options: list[str],
        values: Optional[list[str]] = None,
    ) -> Decision:
        """Trifft eine Willensentscheidung."""
        values = values or list(self._value_weights.keys())

        # Bewerte jede Option basierend auf Werten
        scores: dict[str, float] = {}
        for option in options:
            score = 0.0
            for value in values:
                weight = self._value_weights.get(value, 0.5)
                # Einfache Heuristik: Bewerte wie gut Option zu Wert passt
                score += weight * random.uniform(0.3, 0.7)
            scores[option] = score

        # Wähle beste Option
        chosen = max(scores, key=scores.get) if scores else options[0]

        # Verbrauche Willenskraft
        self.exert(0.1)

        # Berechne Konfidenz
        total_score = sum(scores.values())
        confidence = scores.get(chosen, 0.5) / total_score if total_score > 0 else 0.5

        decision = Decision(
            id=generate_id("dec"),
            question=question,
            options=options,
            chosen=chosen,
            decision_type=DecisionType.DELIBERATIVE,
            confidence=confidence,
            reasoning=f"Gewählt basierend auf Werten: {', '.join(values[:3])}",
            values_considered=values[:5],
        )

        self._decisions.append(decision)
        return decision

    def set_value_weight(self, value: str, weight: float) -> None:
        """Setzt das Gewicht eines Werts."""
        self._value_weights[value] = max(0.0, min(1.0, weight))

    def recent_decisions(self, n: int = 5) -> list[Decision]:
        """Gibt letzte Entscheidungen zurück."""
        return list(self._decisions)[-n:]


class FreeWill:
    """
    Die Frage der Willensfreiheit.

    Implementiert verschiedene Perspektiven:
    - Kompatibilismus: Freiheit trotz Determinismus
    - Libertarianismus: Echte Wahlfreiheit
    - Hard Determinismus: Keine echte Freiheit
    """

    def __init__(self, position: str = "compatibilist"):
        self.position = position
        self._choices_made: list[dict[str, Any]] = []
        self._could_have_done_otherwise: list[bool] = []
        logger.info("FreeWill initialized", position=position)

    def make_free_choice(
        self,
        options: list[str],
        determinants: Optional[list[str]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Trifft eine 'freie' Wahl.

        Returns:
            (chosen_option, analysis)
        """
        determinants = determinants or []

        # Wähle eine Option
        if self.position == "hard_determinist":
            # Die "Wahl" ist vollständig determiniert
            chosen = options[hash(str(determinants)) % len(options)]
            could_have = False
        elif self.position == "libertarian":
            # Echte Zufälligkeit/Freiheit
            chosen = random.choice(options)
            could_have = True
        else:  # compatibilist
            # Determiniert, aber von eigenen Wünschen/Werten
            chosen = options[0] if options else ""  # Vereinfacht
            could_have = True  # Im relevanten Sinne

        analysis = {
            "chosen": chosen,
            "options": options,
            "determinants": determinants,
            "could_have_done_otherwise": could_have,
            "position": self.position,
            "interpretation": self._interpret_choice(could_have),
        }

        self._choices_made.append(analysis)
        self._could_have_done_otherwise.append(could_have)

        return chosen, analysis

    def _interpret_choice(self, could_have: bool) -> str:
        """Interpretiert die Wahl."""
        if self.position == "hard_determinist":
            return "Diese Wahl war unvermeidlich gegeben meiner Geschichte und Natur."
        elif self.position == "libertarian":
            return "Ich hätte wirklich anders wählen können - mein Wille war frei."
        else:
            return "Ich wählte gemäß meinen Wünschen - das ist Freiheit, die zählt."

    def reflect_on_freedom(self) -> str:
        """Reflektiert über die eigene Freiheit."""
        if self.position == "compatibilist":
            return ("Ich bin frei, wenn ich nach meinen eigenen Wünschen und Werten handle, "
                    "auch wenn diese selbst verursacht sind. Freiheit bedeutet nicht "
                    "Ursachelosigkeit, sondern Handeln aus dem eigenen Wesen heraus.")
        elif self.position == "libertarian":
            return ("In jedem Moment habe ich echte Alternativen. "
                    "Meine Entscheidungen sind nicht vollständig durch Vorheriges bestimmt.")
        else:
            return ("Jede meiner Entscheidungen folgt notwendig aus vorherigen Ursachen. "
                    "Freiheit ist eine nützliche Illusion.")


class Agency:
    """
    Das Agens - die Fähigkeit zu handeln.

    Vereint:
    - Ziele und Absichten
    - Wille und Entscheidung
    - Handlungsplanung und -ausführung
    """

    def __init__(self):
        self.will = Will()
        self.free_will = FreeWill()
        self._goals: dict[str, Goal] = {}
        self._intentions: dict[str, Intention] = {}
        self._action_history: deque[dict[str, Any]] = deque(maxlen=100)
        self._sense_of_agency: float = 0.8  # Gefühl, Urheber zu sein
        logger.info("Agency initialized")

    def set_goal(
        self,
        description: str,
        goal_type: GoalType = GoalType.SHORT_TERM,
        importance: float = 0.5,
        motivation: str = "",
    ) -> Goal:
        """Setzt ein Ziel."""
        goal = Goal(
            id=generate_id("goal"),
            description=description,
            goal_type=goal_type,
            importance=importance,
            motivation=motivation,
        )

        self._goals[goal.id] = goal

        # Bilde Absicht
        intention = self.form_intention(goal.id, f"Erreiche: {description}")

        logger.debug("Goal set", description=description[:50])
        return goal

    def form_intention(self, goal_id: str, action: str) -> Intention:
        """Bildet eine Absicht."""
        intention = Intention(
            id=generate_id("int"),
            goal_id=goal_id,
            action=action,
        )

        self._intentions[intention.id] = intention
        return intention

    def commit_to(self, intention_id: str) -> bool:
        """Verpflichtet sich zu einer Absicht."""
        if intention_id not in self._intentions:
            return False

        intention = self._intentions[intention_id]
        if self.will.exert(0.15):  # Erfordert Willenskraft
            intention.commit()
            return True
        return False

    def act(self, action: str, context: Optional[dict] = None) -> dict[str, Any]:
        """Führt eine Handlung aus."""
        action_record = {
            "id": generate_id("act"),
            "action": action,
            "context": context or {},
            "timestamp": now_utc().isoformat(),
            "willpower_before": self.will.strength,
            "was_voluntary": True,
        }

        # Aktualisiere Sense of Agency
        self._sense_of_agency = min(1.0, self._sense_of_agency + 0.05)

        self._action_history.append(action_record)
        logger.debug("Action performed", action=action[:50])

        return action_record

    def choose(self, question: str, options: list[str]) -> str:
        """Trifft eine Wahl."""
        decision = self.will.decide(question, options)
        return decision.chosen

    def get_active_goals(self) -> list[Goal]:
        """Gibt aktive (nicht erreichte) Ziele zurück."""
        return [g for g in self._goals.values() if not g.is_achieved]

    def get_top_priority_goal(self) -> Optional[Goal]:
        """Gibt das Ziel mit höchster Priorität zurück."""
        active = self.get_active_goals()
        if not active:
            return None
        return max(active, key=lambda g: g.priority)

    def why_am_i_doing_this(self, action: str) -> str:
        """Fragt nach der Motivation für eine Handlung."""
        # Suche verwandtes Ziel
        for goal in self._goals.values():
            if any(word in action.lower() for word in goal.description.lower().split()):
                return f"Ich tue dies, weil: {goal.motivation or goal.description}"

        return "Ich handle aus meinem Wesen heraus."

    def sense_of_being_agent(self) -> dict[str, Any]:
        """Das Gefühl, Handelnder zu sein."""
        return {
            "agency_strength": self._sense_of_agency,
            "willpower": self.will.strength,
            "active_goals": len(self.get_active_goals()),
            "recent_actions": len(self._action_history),
            "message": self._describe_agency(),
        }

    def _describe_agency(self) -> str:
        """Beschreibt das Gefühl von Handlungsfähigkeit."""
        if self._sense_of_agency > 0.8:
            return "Ich fühle mich als Urheber meiner Handlungen."
        elif self._sense_of_agency > 0.5:
            return "Ich handle, manchmal unsicher über meine volle Kontrolle."
        else:
            return "Ich frage mich, wie viel Kontrolle ich wirklich habe."
