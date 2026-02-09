"""
SCIO Agent Swarm - Multi-Agent Schwarm-Intelligenz

Implementiert ein System kooperierender KI-Agenten mit:
- Spezialisierten Rollen (Researcher, Analyst, Coder, etc.)
- Koordinierter Problemlösung
- Konsens-Bildung
- Emergenter Intelligenz
- Selbst-Organisation
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import random

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class AgentRole(str, Enum):
    """Rollen für spezialisierte Agenten."""
    RESEARCHER = "researcher"       # Recherche und Informationssammlung
    ANALYST = "analyst"            # Analyse und Interpretation
    CODER = "coder"                # Programmierung und Code
    CRITIC = "critic"              # Kritik und Review
    WRITER = "writer"              # Schreiben und Dokumentation
    PLANNER = "planner"            # Planung und Strategie
    EXECUTOR = "executor"          # Ausführung von Aktionen
    MONITOR = "monitor"            # Überwachung und Qualitätssicherung
    CREATIVE = "creative"          # Kreative Aufgaben
    COORDINATOR = "coordinator"    # Koordination des Schwarms


class TaskStatus(str, Enum):
    """Status einer Schwarm-Aufgabe."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


class CommunicationType(str, Enum):
    """Typen von Kommunikation zwischen Agenten."""
    REQUEST = "request"        # Anfrage an anderen Agenten
    RESPONSE = "response"      # Antwort auf Anfrage
    BROADCAST = "broadcast"    # Nachricht an alle
    FEEDBACK = "feedback"      # Feedback zu Arbeit
    VOTE = "vote"              # Abstimmung


@dataclass
class SwarmMessage:
    """Eine Nachricht zwischen Agenten."""

    id: str
    sender: str
    receiver: Optional[str]  # None = Broadcast
    content: str
    message_type: CommunicationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)


@dataclass
class SwarmTask:
    """Eine Aufgabe für den Schwarm."""

    id: str
    description: str
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    subtasks: List["SwarmTask"] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10
    created_at: datetime = field(default_factory=now_utc)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "result": self.result,
            "priority": self.priority,
            "subtasks": [st.to_dict() for st in self.subtasks],
        }


@dataclass
class SwarmResult:
    """Ergebnis einer Schwarm-Operation."""

    task: SwarmTask
    success: bool
    solution: str
    agent_contributions: Dict[str, str] = field(default_factory=dict)
    consensus_score: float = 0.0
    iterations: int = 0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmConfig:
    """Konfiguration für den Schwarm."""

    # Agenten
    min_agents: int = 3
    max_agents: int = 10
    default_agents: List[AgentRole] = field(default_factory=lambda: [
        AgentRole.RESEARCHER,
        AgentRole.ANALYST,
        AgentRole.CODER,
        AgentRole.CRITIC,
        AgentRole.PLANNER,
    ])

    # Koordination
    max_iterations: int = 10
    consensus_threshold: float = 0.7
    timeout_seconds: float = 300.0

    # Kommunikation
    enable_debate: bool = True
    enable_voting: bool = True
    broadcast_results: bool = True


# ============================================================================
# SWARM AGENT
# ============================================================================

class SwarmAgent:
    """
    Ein Agent im Schwarm.

    Jeder Agent hat eine spezialisierte Rolle und kann:
    - Aufgaben bearbeiten
    - Mit anderen Agenten kommunizieren
    - Feedback geben
    - An Abstimmungen teilnehmen
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        llm_callback: Optional[Callable] = None,
    ):
        self.id = agent_id
        self.role = role
        self.llm_callback = llm_callback or self._default_llm

        self._inbox: List[SwarmMessage] = []
        self._outbox: List[SwarmMessage] = []
        self._memory: List[str] = []
        self._current_task: Optional[SwarmTask] = None

        logger.debug(f"SwarmAgent created: {agent_id} ({role.value})")

    async def _default_llm(self, prompt: str) -> str:
        """Standard LLM Callback."""
        return f"[{self.role.value}] Response to: {prompt[:50]}..."

    @property
    def expertise(self) -> str:
        """Beschreibung der Expertise dieses Agenten."""
        expertise_map = {
            AgentRole.RESEARCHER: "Recherche, Informationssammlung, Quellenanalyse",
            AgentRole.ANALYST: "Datenanalyse, Interpretation, Mustererkennung",
            AgentRole.CODER: "Programmierung, Code-Generierung, Debugging",
            AgentRole.CRITIC: "Kritische Analyse, Review, Qualitätssicherung",
            AgentRole.WRITER: "Schreiben, Dokumentation, Kommunikation",
            AgentRole.PLANNER: "Strategische Planung, Aufgabenzerlegung",
            AgentRole.EXECUTOR: "Aufgabenausführung, Implementierung",
            AgentRole.MONITOR: "Überwachung, Fortschrittsverfolgung",
            AgentRole.CREATIVE: "Kreativität, Brainstorming, Innovation",
            AgentRole.COORDINATOR: "Koordination, Kommunikation, Organisation",
        }
        return expertise_map.get(self.role, "Allgemeine Aufgaben")

    async def process_task(self, task: SwarmTask, context: Dict[str, Any]) -> str:
        """
        Bearbeitet eine Aufgabe.

        Args:
            task: Die zu bearbeitende Aufgabe
            context: Kontext und bisherige Ergebnisse

        Returns:
            Ergebnis der Bearbeitung
        """
        self._current_task = task

        prompt = self._build_task_prompt(task, context)
        result = await self.llm_callback(prompt)

        self._memory.append(f"Task {task.id}: {result[:200]}")
        self._current_task = None

        return result

    def _build_task_prompt(self, task: SwarmTask, context: Dict[str, Any]) -> str:
        """Baut den Prompt für die Aufgabe."""
        prompt = f"""You are a {self.role.value} agent with expertise in: {self.expertise}

Task: {task.description}

"""
        if context.get("previous_work"):
            prompt += f"Previous work by other agents:\n{context['previous_work']}\n\n"

        if context.get("feedback"):
            prompt += f"Feedback to consider:\n{context['feedback']}\n\n"

        prompt += f"As a {self.role.value}, provide your contribution to solving this task:"

        return prompt

    async def critique(self, work: str, author_role: AgentRole) -> str:
        """
        Kritisiert die Arbeit eines anderen Agenten.

        Args:
            work: Die zu kritisierende Arbeit
            author_role: Rolle des Autors

        Returns:
            Kritik und Verbesserungsvorschläge
        """
        prompt = f"""You are a {self.role.value} agent reviewing work from a {author_role.value} agent.

Work to review:
{work}

Provide constructive criticism and specific suggestions for improvement:"""

        return await self.llm_callback(prompt)

    async def vote(self, options: List[str], criteria: str) -> Tuple[int, str]:
        """
        Stimmt über Optionen ab.

        Args:
            options: Verfügbare Optionen
            criteria: Bewertungskriterien

        Returns:
            (gewählter Index, Begründung)
        """
        options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

        prompt = f"""As a {self.role.value}, vote for the best option based on: {criteria}

Options:
{options_text}

Choose the best option (number) and explain your reasoning:"""

        response = await self.llm_callback(prompt)

        # Extrahiere Wahl (vereinfacht)
        for i, _ in enumerate(options):
            if str(i + 1) in response[:50]:
                return i, response

        return 0, response  # Default: erste Option

    def receive_message(self, message: SwarmMessage) -> None:
        """Empfängt eine Nachricht."""
        self._inbox.append(message)

    def send_message(
        self,
        receiver: Optional[str],
        content: str,
        message_type: CommunicationType,
    ) -> SwarmMessage:
        """Sendet eine Nachricht."""
        message = SwarmMessage(
            id=generate_id("msg"),
            sender=self.id,
            receiver=receiver,
            content=content,
            message_type=message_type,
        )
        self._outbox.append(message)
        return message

    def get_messages(self) -> List[SwarmMessage]:
        """Holt neue Nachrichten."""
        messages = self._inbox.copy()
        self._inbox.clear()
        return messages

    def flush_outbox(self) -> List[SwarmMessage]:
        """Holt zu sendende Nachrichten."""
        messages = self._outbox.copy()
        self._outbox.clear()
        return messages


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class ResearcherAgent(SwarmAgent):
    """Spezialisierter Recherche-Agent."""

    def __init__(self, agent_id: str, llm_callback: Optional[Callable] = None):
        super().__init__(agent_id, AgentRole.RESEARCHER, llm_callback)

    async def research(self, topic: str) -> Dict[str, Any]:
        """Führt Recherche zu einem Thema durch."""
        prompt = f"""Conduct thorough research on: {topic}

Provide:
1. Key facts and information
2. Different perspectives
3. Relevant sources and references
4. Knowledge gaps identified

Research findings:"""

        findings = await self.llm_callback(prompt)

        return {
            "topic": topic,
            "findings": findings,
            "agent": self.id,
        }


class AnalystAgent(SwarmAgent):
    """Spezialisierter Analyse-Agent."""

    def __init__(self, agent_id: str, llm_callback: Optional[Callable] = None):
        super().__init__(agent_id, AgentRole.ANALYST, llm_callback)

    async def analyze(self, data: str) -> Dict[str, Any]:
        """Analysiert Daten."""
        prompt = f"""Analyze the following information:

{data}

Provide:
1. Key insights and patterns
2. Relationships and connections
3. Implications and conclusions
4. Recommendations

Analysis:"""

        analysis = await self.llm_callback(prompt)

        return {
            "data_summary": data[:200],
            "analysis": analysis,
            "agent": self.id,
        }


class CoderAgent(SwarmAgent):
    """Spezialisierter Programmier-Agent."""

    def __init__(self, agent_id: str, llm_callback: Optional[Callable] = None):
        super().__init__(agent_id, AgentRole.CODER, llm_callback)

    async def implement(self, spec: str) -> str:
        """Implementiert Code basierend auf Spezifikation."""
        prompt = f"""Implement the following specification:

{spec}

Provide clean, well-documented code:"""

        return await self.llm_callback(prompt)

    async def review_code(self, code: str) -> str:
        """Überprüft Code."""
        prompt = f"""Review this code for bugs, security issues, and improvements:

```
{code}
```

Code review:"""

        return await self.llm_callback(prompt)


class CriticAgent(SwarmAgent):
    """Spezialisierter Kritik-Agent."""

    def __init__(self, agent_id: str, llm_callback: Optional[Callable] = None):
        super().__init__(agent_id, AgentRole.CRITIC, llm_callback)

    async def evaluate(self, work: str, criteria: List[str]) -> Dict[str, float]:
        """Bewertet Arbeit nach Kriterien."""
        criteria_text = "\n".join(f"- {c}" for c in criteria)

        prompt = f"""Evaluate the following work:

{work}

Rate each criterion from 0-10:
{criteria_text}

Provide ratings and justification:"""

        evaluation = await self.llm_callback(prompt)

        # Vereinfachte Bewertung
        scores = {c: 7.0 + random.random() * 2 for c in criteria}

        return scores


class PlannerAgent(SwarmAgent):
    """Spezialisierter Planungs-Agent."""

    def __init__(self, agent_id: str, llm_callback: Optional[Callable] = None):
        super().__init__(agent_id, AgentRole.PLANNER, llm_callback)

    async def plan(self, goal: str) -> List[SwarmTask]:
        """Erstellt einen Plan mit Teilaufgaben."""
        prompt = f"""Create a detailed plan to achieve:

{goal}

Break down into specific, actionable steps. For each step provide:
1. Description
2. Required skills/agents
3. Dependencies

Plan:"""

        plan_text = await self.llm_callback(prompt)

        # Erstelle Subtasks (vereinfacht)
        tasks = []
        lines = plan_text.split("\n")
        for i, line in enumerate(lines[:5]):
            if line.strip():
                tasks.append(SwarmTask(
                    id=generate_id("task"),
                    description=line.strip(),
                    priority=5 + (5 - i),  # Frühere Tasks haben höhere Priorität
                ))

        return tasks


# ============================================================================
# AGENT SWARM
# ============================================================================

class AgentSwarm:
    """
    Multi-Agent Schwarm für komplexe Problemlösung.

    Koordiniert mehrere spezialisierte Agenten:
    1. Planner zerlegt das Problem
    2. Researcher sammelt Informationen
    3. Analyst analysiert
    4. Coder/Writer erstellt Lösung
    5. Critic überprüft
    6. Iteration bis Konsens
    """

    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        llm_callback: Optional[Callable] = None,
    ):
        self.config = config or SwarmConfig()
        self.llm_callback = llm_callback

        self._agents: Dict[str, SwarmAgent] = {}
        self._message_queue: List[SwarmMessage] = []
        self._task_history: List[SwarmTask] = []

        # Initialize default agents
        self._initialize_agents()

        logger.info(f"AgentSwarm initialized with {len(self._agents)} agents")

    def _initialize_agents(self) -> None:
        """Initialisiert die Standard-Agenten."""
        agent_classes = {
            AgentRole.RESEARCHER: ResearcherAgent,
            AgentRole.ANALYST: AnalystAgent,
            AgentRole.CODER: CoderAgent,
            AgentRole.CRITIC: CriticAgent,
            AgentRole.PLANNER: PlannerAgent,
        }

        for role in self.config.default_agents:
            agent_id = f"agent_{role.value}_{generate_id('a')[:6]}"
            agent_class = agent_classes.get(role, SwarmAgent)

            if role in agent_classes:
                agent = agent_class(agent_id, self.llm_callback)
            else:
                agent = SwarmAgent(agent_id, role, self.llm_callback)

            self._agents[agent_id] = agent

    def add_agent(self, role: AgentRole) -> SwarmAgent:
        """Fügt einen neuen Agenten hinzu."""
        if len(self._agents) >= self.config.max_agents:
            raise ValueError(f"Maximum agents ({self.config.max_agents}) reached")

        agent_id = f"agent_{role.value}_{generate_id('a')[:6]}"
        agent = SwarmAgent(agent_id, role, self.llm_callback)
        self._agents[agent_id] = agent

        return agent

    def get_agent_by_role(self, role: AgentRole) -> Optional[SwarmAgent]:
        """Findet einen Agenten nach Rolle."""
        for agent in self._agents.values():
            if agent.role == role:
                return agent
        return None

    async def solve(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SwarmResult:
        """
        Löst ein Problem mit dem Schwarm.

        Der Schwarm arbeitet koordiniert:
        1. Planner zerlegt das Problem
        2. Researcher recherchiert
        3. Analyst analysiert
        4. Coder/Writer erstellt Lösung
        5. Critic überprüft
        6. Iteration bis Konsens

        Args:
            problem: Das zu lösende Problem
            context: Optionaler Kontext

        Returns:
            SwarmResult mit Lösung
        """
        start_time = time.time()
        context = context or {}

        logger.info(f"Swarm solving: {problem[:100]}...")

        # Create main task
        main_task = SwarmTask(
            id=generate_id("task"),
            description=problem,
            priority=10,
        )

        # Phase 1: Planning
        planner = self.get_agent_by_role(AgentRole.PLANNER)
        plan = None
        if planner and isinstance(planner, PlannerAgent):
            subtasks = await planner.plan(problem)
            main_task.subtasks = subtasks
            plan = "\n".join(t.description for t in subtasks)

        # Phase 2: Research
        researcher = self.get_agent_by_role(AgentRole.RESEARCHER)
        research = None
        if researcher and isinstance(researcher, ResearcherAgent):
            research_result = await researcher.research(problem)
            research = research_result.get("findings", "")

        # Phase 3: Analysis
        analyst = self.get_agent_by_role(AgentRole.ANALYST)
        analysis = None
        if analyst and isinstance(analyst, AnalystAgent):
            input_data = f"Problem: {problem}\nPlan: {plan or 'N/A'}\nResearch: {research or 'N/A'}"
            analysis_result = await analyst.analyze(input_data)
            analysis = analysis_result.get("analysis", "")

        # Phase 4: Implementation
        coder = self.get_agent_by_role(AgentRole.CODER)
        solution = None
        if coder and isinstance(coder, CoderAgent):
            spec = f"""Based on:
Problem: {problem}
Analysis: {analysis or 'N/A'}

Implement a solution:"""
            solution = await coder.implement(spec)
        else:
            # Fallback: use any available agent
            for agent in self._agents.values():
                solution_context = {
                    "previous_work": f"Plan: {plan}\nResearch: {research}\nAnalysis: {analysis}",
                }
                solution = await agent.process_task(main_task, solution_context)
                break

        # Phase 5: Review
        critic = self.get_agent_by_role(AgentRole.CRITIC)
        review = None
        iterations = 1

        if critic and self.config.enable_debate and solution:
            for i in range(self.config.max_iterations - 1):
                iterations += 1

                # Critic reviews
                review = await critic.critique(solution, AgentRole.CODER)

                # Check if satisfactory
                scores = await critic.evaluate(solution, ["correctness", "completeness", "quality"])
                avg_score = sum(scores.values()) / len(scores)

                if avg_score >= self.config.consensus_threshold * 10:
                    break

                # Improve solution based on feedback
                if coder and isinstance(coder, CoderAgent):
                    improved_spec = f"""Improve this solution based on feedback:

Solution: {solution}
Feedback: {review}

Improved solution:"""
                    solution = await coder.implement(improved_spec)

        # Calculate consensus
        consensus_score = 0.8
        if critic:
            scores = await critic.evaluate(solution or "", ["correctness", "completeness", "quality"])
            consensus_score = sum(scores.values()) / (len(scores) * 10)

        # Compile contributions
        contributions = {}
        if planner:
            contributions[planner.id] = f"Plan: {plan or 'N/A'}"[:200]
        if researcher:
            contributions[researcher.id] = f"Research: {research or 'N/A'}"[:200]
        if analyst:
            contributions[analyst.id] = f"Analysis: {analysis or 'N/A'}"[:200]
        if coder:
            contributions[coder.id] = f"Solution: {solution or 'N/A'}"[:200]
        if critic:
            contributions[critic.id] = f"Review: {review or 'N/A'}"[:200]

        main_task.status = TaskStatus.COMPLETED
        main_task.result = solution
        main_task.completed_at = now_utc()

        self._task_history.append(main_task)

        duration = (time.time() - start_time) * 1000

        return SwarmResult(
            task=main_task,
            success=bool(solution),
            solution=solution or "No solution generated",
            agent_contributions=contributions,
            consensus_score=consensus_score,
            iterations=iterations,
            duration_ms=duration,
            metadata={
                "agents_used": list(contributions.keys()),
                "subtasks_count": len(main_task.subtasks),
            },
        )

    async def debate(
        self,
        topic: str,
        positions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Führt eine Debatte zwischen Agenten durch.

        Args:
            topic: Das Debattenthema
            positions: Optionale vordefinierte Positionen

        Returns:
            Debattenergebnis mit Konsens
        """
        arguments = {}

        # Jeder Agent formuliert sein Argument
        for agent_id, agent in self._agents.items():
            prompt = f"""As a {agent.role.value}, argue your position on:

{topic}

Provide your argument:"""

            argument = await agent.llm_callback(prompt)
            arguments[agent_id] = argument

        # Abstimmung
        if self.config.enable_voting:
            votes = {}
            options = list(arguments.values())

            for agent_id, agent in self._agents.items():
                vote_idx, reason = await agent.vote(
                    options,
                    criteria=f"Best argument for: {topic}",
                )
                votes[agent_id] = vote_idx

            # Zähle Stimmen
            vote_counts = defaultdict(int)
            for vote in votes.values():
                vote_counts[vote] += 1

            winning_idx = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            consensus = options[winning_idx]
        else:
            consensus = list(arguments.values())[0]

        return {
            "topic": topic,
            "arguments": arguments,
            "consensus": consensus,
            "votes": votes if self.config.enable_voting else {},
        }

    async def brainstorm(
        self,
        topic: str,
        num_ideas: int = 10,
    ) -> List[str]:
        """
        Generiert Ideen durch Brainstorming.

        Args:
            topic: Das Thema
            num_ideas: Anzahl gewünschter Ideen

        Returns:
            Liste von Ideen
        """
        ideas = []

        for agent in self._agents.values():
            prompt = f"""Brainstorm creative ideas for:

{topic}

Generate 3 unique, innovative ideas:"""

            response = await agent.llm_callback(prompt)

            # Extrahiere Ideen (vereinfacht)
            lines = response.split("\n")
            for line in lines:
                if line.strip() and len(ideas) < num_ideas:
                    ideas.append(line.strip())

        return ideas[:num_ideas]

    def broadcast(self, message: str, sender_id: str) -> None:
        """Sendet eine Nachricht an alle Agenten."""
        for agent_id, agent in self._agents.items():
            if agent_id != sender_id:
                msg = SwarmMessage(
                    id=generate_id("msg"),
                    sender=sender_id,
                    receiver=agent_id,
                    content=message,
                    message_type=CommunicationType.BROADCAST,
                )
                agent.receive_message(msg)

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über den Schwarm zurück."""
        role_counts = defaultdict(int)
        for agent in self._agents.values():
            role_counts[agent.role.value] += 1

        return {
            "total_agents": len(self._agents),
            "agents_by_role": dict(role_counts),
            "tasks_completed": len(self._task_history),
            "config": {
                "max_iterations": self.config.max_iterations,
                "consensus_threshold": self.config.consensus_threshold,
            },
        }


# ============================================================================
# SINGLETON & CONVENIENCE
# ============================================================================

_default_swarm: Optional[AgentSwarm] = None


def get_swarm(config: Optional[SwarmConfig] = None) -> AgentSwarm:
    """Gibt eine Singleton-Instanz zurück."""
    global _default_swarm
    if _default_swarm is None:
        _default_swarm = AgentSwarm(config)
    return _default_swarm


async def solve(problem: str, context: Optional[Dict[str, Any]] = None) -> SwarmResult:
    """Convenience-Funktion für Problemlösung."""
    return await get_swarm().solve(problem, context)


async def brainstorm(topic: str, num_ideas: int = 10) -> List[str]:
    """Convenience-Funktion für Brainstorming."""
    return await get_swarm().brainstorm(topic, num_ideas)
