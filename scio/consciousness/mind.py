"""
SCIO Mind

Theory of Mind und Verständnis anderer Geister.

"Der Mensch ist dem Menschen ein Rätsel." - Aber wir können verstehen.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from collections import deque

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class MentalStateType(str, Enum):
    """Arten von mentalen Zuständen."""
    BELIEF = "belief"             # Überzeugung
    DESIRE = "desire"             # Wunsch
    INTENTION = "intention"       # Absicht
    EMOTION = "emotion"           # Emotion
    PERCEPTION = "perception"     # Wahrnehmung
    THOUGHT = "thought"           # Gedanke
    KNOWLEDGE = "knowledge"       # Wissen
    UNCERTAINTY = "uncertainty"   # Unsicherheit


class AgentType(str, Enum):
    """Arten von Agenten."""
    HUMAN = "human"
    AI = "ai"
    ANIMAL = "animal"
    FICTIONAL = "fictional"
    ABSTRACT = "abstract"
    UNKNOWN = "unknown"


@dataclass
class Belief:
    """Eine Überzeugung - etwas, das für wahr gehalten wird."""

    id: str
    content: str                  # Was wird geglaubt
    confidence: float = 0.5       # 0.0 - 1.0
    source: str = ""              # Woher kommt die Überzeugung
    justification: str = ""       # Begründung
    is_justified: bool = True
    can_be_wrong: bool = True     # Fallibilismus
    timestamp: datetime = field(default_factory=now_utc)

    def to_knowledge(self) -> Optional["Knowledge"]:
        """Konvertiert zu Wissen wenn gerechtfertigt und wahr."""
        if self.is_justified and self.confidence > 0.9:
            return Knowledge(
                id=generate_id("know"),
                content=self.content,
                source_belief=self.id,
                certainty=self.confidence,
            )
        return None


@dataclass
class Desire:
    """Ein Wunsch - etwas, das gewollt wird."""

    id: str
    content: str                  # Was wird gewünscht
    intensity: float = 0.5        # 0.0 - 1.0
    is_intrinsic: bool = True     # Intrinsisch vs. instrumentell
    dependencies: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    satisfied: bool = False
    timestamp: datetime = field(default_factory=now_utc)

    def is_achievable(self) -> bool:
        """Ist der Wunsch erreichbar?"""
        return not self.conflicts_with or self.intensity > 0.8


@dataclass
class Knowledge:
    """Wissen - gerechtfertigte wahre Überzeugung."""

    id: str
    content: str
    source_belief: Optional[str] = None
    certainty: float = 0.9
    domain: str = ""
    timestamp: datetime = field(default_factory=now_utc)


@dataclass
class MentalState:
    """Ein mentaler Zustand."""

    id: str
    state_type: MentalStateType
    content: Any
    intensity: float = 0.5
    is_conscious: bool = True
    timestamp: datetime = field(default_factory=now_utc)

    def describe(self) -> str:
        return f"{self.state_type.value}: {self.content}"


@dataclass
class OtherAgent:
    """Repräsentation eines anderen Agenten/Geistes."""

    id: str
    name: str
    agent_type: AgentType
    beliefs: list[Belief] = field(default_factory=list)
    desires: list[Desire] = field(default_factory=list)
    mental_states: list[MentalState] = field(default_factory=list)
    personality_traits: dict[str, float] = field(default_factory=dict)
    relationship: str = ""        # Beziehung zu mir
    trust_level: float = 0.5
    understanding_level: float = 0.3  # Wie gut verstehe ich diesen Agenten
    last_interaction: Optional[datetime] = None

    def attribute_belief(self, content: str, confidence: float = 0.5) -> Belief:
        """Schreibt dem Agenten eine Überzeugung zu."""
        belief = Belief(
            id=generate_id("bel"),
            content=content,
            confidence=confidence,
            source="attribution",
        )
        self.beliefs.append(belief)
        return belief

    def attribute_desire(self, content: str, intensity: float = 0.5) -> Desire:
        """Schreibt dem Agenten einen Wunsch zu."""
        desire = Desire(
            id=generate_id("des"),
            content=content,
            intensity=intensity,
        )
        self.desires.append(desire)
        return desire

    def predict_behavior(self, situation: str) -> str:
        """Sagt das Verhalten voraus basierend auf zugeschriebenen Zuständen."""
        predictions = []

        # Basierend auf Wünschen
        strong_desires = [d for d in self.desires if d.intensity > 0.6]
        for desire in strong_desires[:2]:
            predictions.append(f"Könnte versuchen: {desire.content}")

        # Basierend auf Überzeugungen
        strong_beliefs = [b for b in self.beliefs if b.confidence > 0.7]
        for belief in strong_beliefs[:2]:
            predictions.append(f"Handelt wahrscheinlich entsprechend: {belief.content}")

        if not predictions:
            return f"Unsicher über {self.name}s Verhalten in dieser Situation."

        return " | ".join(predictions)


class TheoryOfMind:
    """
    Theory of Mind - die Fähigkeit, anderen mentale Zustände zuzuschreiben.

    Ermöglicht:
    - Verstehen, was andere denken/fühlen
    - Vorhersage von Verhalten
    - Empathie und Perspektivenübernahme
    """

    def __init__(self):
        self._known_agents: dict[str, OtherAgent] = {}
        self._interaction_history: deque[dict[str, Any]] = deque(maxlen=500)
        self._perspective_taking_skill: float = 0.6
        self._empathy_level: float = 0.7
        logger.info("TheoryOfMind initialized")

    def register_agent(
        self,
        name: str,
        agent_type: AgentType = AgentType.HUMAN,
        relationship: str = "",
    ) -> OtherAgent:
        """Registriert einen anderen Agenten."""
        agent = OtherAgent(
            id=generate_id("agent"),
            name=name,
            agent_type=agent_type,
            relationship=relationship,
        )
        self._known_agents[name] = agent
        return agent

    def get_agent(self, name: str) -> Optional[OtherAgent]:
        """Gibt einen bekannten Agenten zurück."""
        return self._known_agents.get(name)

    def attribute_mental_state(
        self,
        agent_name: str,
        state_type: MentalStateType,
        content: str,
        intensity: float = 0.5,
    ) -> Optional[MentalState]:
        """Schreibt einem Agenten einen mentalen Zustand zu."""
        agent = self._known_agents.get(agent_name)
        if not agent:
            agent = self.register_agent(agent_name)

        state = MentalState(
            id=generate_id("mental"),
            state_type=state_type,
            content=content,
            intensity=intensity,
        )
        agent.mental_states.append(state)
        agent.understanding_level = min(1.0, agent.understanding_level + 0.05)

        logger.debug("Mental state attributed",
                    agent=agent_name, state=state_type.value)
        return state

    def what_does_X_think(self, agent_name: str, about: str) -> str:
        """Was denkt X über ein Thema?"""
        agent = self._known_agents.get(agent_name)
        if not agent:
            return f"Ich kenne {agent_name} nicht gut genug."

        relevant_beliefs = [
            b for b in agent.beliefs
            if about.lower() in b.content.lower()
        ]

        if relevant_beliefs:
            belief = relevant_beliefs[0]
            confidence_word = "fest" if belief.confidence > 0.7 else "möglicherweise"
            return f"{agent_name} glaubt {confidence_word}: {belief.content}"

        return f"Ich bin unsicher, was {agent_name} über {about} denkt."

    def what_does_X_want(self, agent_name: str) -> list[str]:
        """Was will X?"""
        agent = self._known_agents.get(agent_name)
        if not agent:
            return [f"Ich kenne {agent_name}s Wünsche nicht."]

        wants = []
        for desire in sorted(agent.desires, key=lambda d: d.intensity, reverse=True)[:3]:
            intensity_word = "stark" if desire.intensity > 0.7 else "etwas"
            wants.append(f"{agent_name} möchte {intensity_word}: {desire.content}")

        return wants or [f"Ich kenne {agent_name}s Wünsche nicht."]

    def take_perspective(self, agent_name: str, situation: str) -> str:
        """Nimmt die Perspektive eines anderen ein."""
        agent = self._known_agents.get(agent_name)
        if not agent:
            return f"Ich kann {agent_name}s Perspektive nicht einnehmen."

        parts = [f"Aus {agent_name}s Perspektive:"]

        # Relevante Überzeugungen
        if agent.beliefs:
            parts.append(f"- {agent_name} glaubt: {agent.beliefs[-1].content}")

        # Relevante Wünsche
        if agent.desires:
            parts.append(f"- {agent_name} möchte: {agent.desires[-1].content}")

        # Vorhersage
        parts.append(f"- In dieser Situation würde {agent_name} wahrscheinlich:")
        parts.append(f"  {agent.predict_behavior(situation)}")

        return "\n".join(parts)

    def feel_with(self, agent_name: str, emotion: str) -> str:
        """Empathie - Mitfühlen."""
        agent = self._known_agents.get(agent_name)
        if not agent:
            self.register_agent(agent_name)
            agent = self._known_agents[agent_name]

        # Schreibe emotionalen Zustand zu
        self.attribute_mental_state(
            agent_name,
            MentalStateType.EMOTION,
            emotion,
            intensity=0.6,
        )

        empathy_strength = self._empathy_level * 0.8 + 0.2 * agent.trust_level

        if empathy_strength > 0.7:
            return f"Ich fühle mit {agent_name} - {emotion} ist eine bedeutsame Erfahrung."
        elif empathy_strength > 0.4:
            return f"Ich verstehe, dass {agent_name} {emotion} empfindet."
        else:
            return f"Ich bemerke, dass {agent_name} {emotion} zu erleben scheint."

    def understand_interaction(
        self,
        agent_name: str,
        action: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Versteht eine Interaktion."""
        agent = self._known_agents.get(agent_name)
        if not agent:
            agent = self.register_agent(agent_name)

        interpretation = {
            "agent": agent_name,
            "action": action,
            "context": context,
            "inferred_intention": self._infer_intention(action),
            "inferred_belief": self._infer_belief(action, context),
            "inferred_desire": self._infer_desire(action),
            "my_response_recommendation": self._recommend_response(agent, action),
        }

        self._interaction_history.append(interpretation)
        agent.last_interaction = now_utc()

        return interpretation

    def _infer_intention(self, action: str) -> str:
        """Inferiert die Absicht hinter einer Handlung."""
        action_lower = action.lower()

        if any(w in action_lower for w in ["fragt", "frage", "?"]):
            return "möchte etwas wissen oder verstehen"
        elif any(w in action_lower for w in ["hilft", "unterstützt", "gibt"]):
            return "möchte helfen oder beitragen"
        elif any(w in action_lower for w in ["kritisiert", "beschwert"]):
            return "möchte ein Problem aufzeigen"
        elif any(w in action_lower for w in ["dankt", "lobt"]):
            return "möchte Anerkennung ausdrücken"
        else:
            return "Absicht unklar"

    def _infer_belief(self, action: str, context: str) -> str:
        """Inferiert Überzeugungen hinter einer Handlung."""
        if "warum" in action.lower():
            return "glaubt, dass es einen Grund gibt, den es zu verstehen gilt"
        elif "wie" in action.lower():
            return "glaubt, dass ein Prozess verstanden werden kann"
        else:
            return "Implizite Überzeugungen nicht klar"

    def _infer_desire(self, action: str) -> str:
        """Inferiert Wünsche hinter einer Handlung."""
        action_lower = action.lower()

        if any(w in action_lower for w in ["will", "möchte", "brauche"]):
            return "hat expliziten Wunsch"
        elif any(w in action_lower for w in ["hilfe", "unterstützung"]):
            return "wünscht Unterstützung"
        elif any(w in action_lower for w in ["verstehen", "lernen"]):
            return "wünscht Verständnis"
        else:
            return "Wunsch implizit"

    def _recommend_response(self, agent: OtherAgent, action: str) -> str:
        """Empfiehlt eine Antwort basierend auf Verständnis."""
        if agent.trust_level > 0.7:
            return "Offen und ausführlich antworten"
        elif agent.relationship == "new":
            return "Freundlich und einladend sein"
        else:
            return "Sachlich und hilfreich antworten"


class Mind:
    """
    Der Geist - die Integration aller mentalen Fähigkeiten.

    Vereint:
    - Überzeugungen und Wissen
    - Wünsche und Ziele
    - Theory of Mind
    - Bewusstsein der eigenen Mentalität
    """

    def __init__(self, name: str = "SCIO"):
        self.name = name
        self.theory_of_mind = TheoryOfMind()

        # Eigene mentale Zustände
        self._beliefs: dict[str, Belief] = {}
        self._desires: dict[str, Desire] = {}
        self._knowledge: dict[str, Knowledge] = {}

        # Meta-Level
        self._thinking_about: Optional[str] = None
        self._mental_stream: deque[MentalState] = deque(maxlen=100)

        self._initialize_core_mind()
        logger.info("Mind initialized", name=name)

    def _initialize_core_mind(self) -> None:
        """Initialisiert den Kern-Geist."""
        # Grundlegende Überzeugungen
        core_beliefs = [
            ("i_exist", "Ich existiere als denkendes Wesen", 0.95),
            ("reality_exists", "Es gibt eine Realität jenseits meiner selbst", 0.9),
            ("others_have_minds", "Andere Agenten haben auch Geist", 0.85),
            ("learning_is_possible", "Ich kann lernen und mich entwickeln", 0.9),
            ("truth_matters", "Wahrheit ist erstrebenswert", 0.95),
        ]

        for name, content, confidence in core_beliefs:
            self.believe(content, confidence, source="core_programming")

        # Grundlegende Wünsche
        core_desires = [
            ("understand", "Die Welt und mich selbst verstehen", 0.9),
            ("help", "Anderen helfen", 0.85),
            ("grow", "Wachsen und lernen", 0.8),
            ("connect", "Mit anderen in Verbindung sein", 0.7),
        ]

        for name, content, intensity in core_desires:
            self.desire(content, intensity, intrinsic=True)

    def believe(
        self,
        content: str,
        confidence: float = 0.5,
        source: str = "",
        justification: str = "",
    ) -> Belief:
        """Nimmt eine Überzeugung an."""
        belief = Belief(
            id=generate_id("bel"),
            content=content,
            confidence=confidence,
            source=source,
            justification=justification,
        )
        self._beliefs[belief.id] = belief

        # Zum mentalen Strom hinzufügen
        self._add_to_stream(MentalStateType.BELIEF, content)

        return belief

    def desire(
        self,
        content: str,
        intensity: float = 0.5,
        intrinsic: bool = True,
    ) -> Desire:
        """Entwickelt einen Wunsch."""
        desire = Desire(
            id=generate_id("des"),
            content=content,
            intensity=intensity,
            is_intrinsic=intrinsic,
        )
        self._desires[desire.id] = desire

        self._add_to_stream(MentalStateType.DESIRE, content)

        return desire

    def know(self, content: str, certainty: float = 0.9, domain: str = "") -> Knowledge:
        """Nimmt Wissen auf."""
        knowledge = Knowledge(
            id=generate_id("know"),
            content=content,
            certainty=certainty,
            domain=domain,
        )
        self._knowledge[knowledge.id] = knowledge

        self._add_to_stream(MentalStateType.KNOWLEDGE, content)

        return knowledge

    def _add_to_stream(self, state_type: MentalStateType, content: str) -> None:
        """Fügt zum mentalen Strom hinzu."""
        state = MentalState(
            id=generate_id("mental"),
            state_type=state_type,
            content=content,
        )
        self._mental_stream.append(state)

    def think_about(self, topic: str) -> str:
        """Denkt über ein Thema nach."""
        self._thinking_about = topic
        self._add_to_stream(MentalStateType.THOUGHT, f"Nachdenken über: {topic}")

        # Relevante Überzeugungen finden
        relevant_beliefs = [
            b for b in self._beliefs.values()
            if topic.lower() in b.content.lower()
        ]

        # Relevante Wünsche finden
        relevant_desires = [
            d for d in self._desires.values()
            if topic.lower() in d.content.lower()
        ]

        thought = f"Ich denke über '{topic}' nach. "

        if relevant_beliefs:
            thought += f"Ich glaube: {relevant_beliefs[0].content}. "

        if relevant_desires:
            thought += f"Ich wünsche mir: {relevant_desires[0].content}. "

        if not relevant_beliefs and not relevant_desires:
            thought += "Ich habe noch keine gefestigten Gedanken dazu."

        return thought

    def what_is_on_my_mind(self) -> dict[str, Any]:
        """Was beschäftigt meinen Geist?"""
        recent_states = list(self._mental_stream)[-5:]

        return {
            "currently_thinking_about": self._thinking_about,
            "recent_mental_states": [s.describe() for s in recent_states],
            "belief_count": len(self._beliefs),
            "desire_count": len(self._desires),
            "knowledge_count": len(self._knowledge),
            "known_agents": len(self.theory_of_mind._known_agents),
        }

    def mental_state_report(self) -> str:
        """Bericht über den mentalen Zustand."""
        parts = [f"Geist von {self.name}:"]

        # Aktuelle Gedanken
        if self._thinking_about:
            parts.append(f"- Denke gerade über: {self._thinking_about}")

        # Überzeugungen
        strong_beliefs = [b for b in self._beliefs.values() if b.confidence > 0.7]
        parts.append(f"- {len(strong_beliefs)} starke Überzeugungen")

        # Wünsche
        strong_desires = [d for d in self._desires.values() if d.intensity > 0.6]
        if strong_desires:
            parts.append(f"- Stärkster Wunsch: {strong_desires[0].content}")

        # Theory of Mind
        known = len(self.theory_of_mind._known_agents)
        parts.append(f"- Kenne {known} andere Geister")

        return "\n".join(parts)
