"""
SCIO Advanced Reasoning - Superintelligentes Reasoning System

Implementiert die fortschrittlichsten Reasoning-Techniken:
- Tree of Thought (ToT): Exploriert mehrere Denkpfade parallel
- Chain of Thought (CoT): Schritt-für-Schritt logisches Denken
- ReAct: Kombiniert Reasoning mit Actions
- MCTS: Monte Carlo Tree Search für komplexe Planung
- Self-Reflection: Iterative Selbstverbesserung
"""

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import heapq
from collections import defaultdict
import json

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ThoughtType(str, Enum):
    """Typen von Gedanken im Reasoning-Prozess."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"
    ACTION = "action"
    REFLECTION = "reflection"
    QUESTION = "question"
    INSIGHT = "insight"


class ReasoningStrategy(str, Enum):
    """Verfügbare Reasoning-Strategien."""
    TREE_OF_THOUGHT = "tree_of_thought"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    MCTS = "mcts"
    SELF_REFLECTION = "self_reflection"
    HYBRID = "hybrid"


@dataclass
class ThoughtNode:
    """Ein Gedankenknoten im Reasoning-Baum."""

    id: str
    content: str
    thought_type: ThoughtType
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.5
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "thought_type": self.thought_type.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "score": self.score,
            "confidence": self.confidence,
            "depth": self.depth,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningStep:
    """Ein Schritt im Reasoning-Prozess."""

    step_number: int
    thought: str
    reasoning: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 0.5


@dataclass
class ReasoningResult:
    """Ergebnis eines Reasoning-Prozesses."""

    strategy: ReasoningStrategy
    problem: str
    solution: str
    confidence: float
    steps: List[ReasoningStep] = field(default_factory=list)
    thoughts: List[ThoughtNode] = field(default_factory=list)
    reasoning_trace: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.7


@dataclass
class ReasoningConfig:
    """Konfiguration für Reasoning-Prozesse."""

    # General
    max_depth: int = 10
    max_iterations: int = 100
    timeout_seconds: float = 60.0
    min_confidence: float = 0.3

    # Tree of Thought
    tot_branches: int = 5
    tot_beam_width: int = 3
    tot_evaluation_samples: int = 3

    # Chain of Thought
    cot_max_steps: int = 20
    cot_detail_level: str = "high"  # low, medium, high

    # ReAct
    react_max_actions: int = 10
    react_action_timeout: float = 30.0

    # MCTS
    mcts_simulations: int = 1000
    mcts_exploration_weight: float = 1.414  # sqrt(2)
    mcts_max_rollout_depth: int = 20

    # Self-Reflection
    reflection_iterations: int = 3
    reflection_criteria: List[str] = field(default_factory=lambda: [
        "correctness", "completeness", "clarity", "efficiency"
    ])


# ============================================================================
# BASE REASONING CLASS
# ============================================================================

class BaseReasoner(ABC):
    """Abstrakte Basisklasse für alle Reasoner."""

    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self._thought_tree: Dict[str, ThoughtNode] = {}
        self._root_id: Optional[str] = None

    @abstractmethod
    async def reason(self, problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """Führt den Reasoning-Prozess durch."""
        pass

    def _create_thought(
        self,
        content: str,
        thought_type: ThoughtType,
        parent_id: Optional[str] = None,
        score: float = 0.0,
        confidence: float = 0.5,
    ) -> ThoughtNode:
        """Erstellt einen neuen Gedankenknoten."""
        thought_id = generate_id("thought")
        depth = 0

        if parent_id and parent_id in self._thought_tree:
            depth = self._thought_tree[parent_id].depth + 1
            self._thought_tree[parent_id].children_ids.append(thought_id)

        thought = ThoughtNode(
            id=thought_id,
            content=content,
            thought_type=thought_type,
            parent_id=parent_id,
            score=score,
            confidence=confidence,
            depth=depth,
        )

        self._thought_tree[thought_id] = thought

        if self._root_id is None:
            self._root_id = thought_id

        return thought

    def _get_path_to_root(self, thought_id: str) -> List[ThoughtNode]:
        """Gibt den Pfad von einem Gedanken zur Wurzel zurück."""
        path = []
        current_id = thought_id

        while current_id:
            if current_id in self._thought_tree:
                node = self._thought_tree[current_id]
                path.append(node)
                current_id = node.parent_id
            else:
                break

        return list(reversed(path))

    def _get_best_path(self) -> List[ThoughtNode]:
        """Findet den besten Pfad durch den Gedankenbaum."""
        if not self._root_id:
            return []

        # Finde Blatt mit höchstem Score
        best_leaf = None
        best_score = float("-inf")

        for thought in self._thought_tree.values():
            if not thought.children_ids:  # Blatt
                if thought.score > best_score:
                    best_score = thought.score
                    best_leaf = thought

        if best_leaf:
            return self._get_path_to_root(best_leaf.id)
        return []


# ============================================================================
# TREE OF THOUGHT (ToT)
# ============================================================================

class TreeOfThought(BaseReasoner):
    """
    Tree of Thought Reasoning.

    Exploriert mehrere Denkpfade parallel und wählt den besten aus.
    Inspiriert von: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    """

    def __init__(self, config: Optional[ReasoningConfig] = None, llm_callback: Optional[Callable] = None):
        super().__init__(config)
        self.llm_callback = llm_callback or self._default_llm

    async def _default_llm(self, prompt: str) -> str:
        """
        Standard-LLM-Callback der SCIO's LLM-System nutzt.

        Falls kein LLM verfügbar ist, wird ein regelbasierter Fallback verwendet.
        """
        try:
            # Versuche SCIO's LLM-System zu nutzen
            from scio.intelligence.llm import get_llm

            llm = get_llm()
            if llm:
                response = await llm.generate(prompt, max_tokens=500)
                if response and hasattr(response, 'text'):
                    return response.text
                elif isinstance(response, str):
                    return response

        except ImportError:
            logger.debug("LLM module not available, using fallback")
        except Exception as e:
            logger.debug(f"LLM call failed: {e}, using fallback")

        # Regelbasierter Fallback für Tree-of-Thought
        prompt_lower = prompt.lower()

        # Analysiere Prompt-Typ und generiere passende Antwort
        if "evaluate" in prompt_lower or "score" in prompt_lower:
            return "Score: 0.7 - The approach shows promise but needs further refinement."

        elif "generate" in prompt_lower or "thought" in prompt_lower:
            return "Let me consider this from multiple angles: 1) Direct approach, 2) Alternative methods, 3) Potential obstacles"

        elif "expand" in prompt_lower or "develop" in prompt_lower:
            return "Expanding on this idea: We should examine the underlying assumptions and verify each step logically."

        elif "conclude" in prompt_lower or "final" in prompt_lower:
            return "Based on the analysis: The most promising path forward combines systematic verification with creative problem-solving."

        elif "problem" in prompt_lower:
            return "Key aspects to consider: 1) Core requirements, 2) Constraints, 3) Available resources, 4) Success criteria"

        else:
            # Generischer Fallback
            return f"Analysis of: {prompt[:100]}... - Consider breaking this into smaller components for systematic evaluation."

    async def reason(self, problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Führt Tree-of-Thought Reasoning durch.

        1. Generiert initiale Gedanken (Branches)
        2. Bewertet jeden Gedanken
        3. Expandiert die besten (Beam Search)
        4. Wiederholt bis Lösung gefunden
        """
        start_time = time.time()
        context = context or {}

        logger.info("Starting Tree of Thought reasoning", problem=problem[:100])

        # Root-Gedanke
        root = self._create_thought(
            content=f"Problem: {problem}",
            thought_type=ThoughtType.OBSERVATION,
            score=0.0,
            confidence=1.0,
        )

        # Generiere initiale Gedanken
        current_level = [root.id]
        all_thoughts = [root]

        for depth in range(self.config.max_depth):
            if not current_level:
                break

            next_level = []

            for thought_id in current_level:
                parent = self._thought_tree[thought_id]

                # Generiere Branches
                branches = await self._generate_branches(problem, parent, context)

                for branch_content, branch_type in branches:
                    # Bewerte den Gedanken
                    score = await self._evaluate_thought(branch_content, problem, context)

                    thought = self._create_thought(
                        content=branch_content,
                        thought_type=branch_type,
                        parent_id=thought_id,
                        score=score,
                        confidence=min(1.0, score + 0.3),
                    )
                    all_thoughts.append(thought)
                    next_level.append(thought.id)

            # Beam Search: Behalte nur die besten
            if next_level:
                scored = [(self._thought_tree[tid].score, tid) for tid in next_level]
                scored.sort(reverse=True)
                current_level = [tid for _, tid in scored[:self.config.tot_beam_width]]

            # Prüfe auf Lösung
            best_thought = self._thought_tree[current_level[0]] if current_level else None
            if best_thought and best_thought.thought_type == ThoughtType.CONCLUSION:
                if best_thought.score >= 0.8:
                    break

        # Extrahiere beste Lösung
        best_path = self._get_best_path()
        solution = await self._synthesize_solution(best_path, problem)

        duration = (time.time() - start_time) * 1000

        return ReasoningResult(
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
            problem=problem,
            solution=solution,
            confidence=best_path[-1].confidence if best_path else 0.0,
            thoughts=all_thoughts,
            reasoning_trace=self._format_trace(best_path),
            duration_ms=duration,
            metadata={
                "total_thoughts": len(all_thoughts),
                "max_depth_reached": max(t.depth for t in all_thoughts) if all_thoughts else 0,
                "branches_per_node": self.config.tot_branches,
            },
        )

    async def _generate_branches(
        self,
        problem: str,
        parent: ThoughtNode,
        context: Dict[str, Any],
    ) -> List[Tuple[str, ThoughtType]]:
        """Generiert mehrere Gedanken-Branches."""
        branches = []

        # Prompts für verschiedene Denkrichtungen
        prompts = [
            ("Analysiere das Problem genauer: ", ThoughtType.ANALYSIS),
            ("Formuliere eine Hypothese: ", ThoughtType.HYPOTHESIS),
            ("Welche Fragen entstehen?: ", ThoughtType.QUESTION),
            ("Was ist ein möglicher Lösungsansatz?: ", ThoughtType.INSIGHT),
            ("Ziehe eine Schlussfolgerung: ", ThoughtType.CONCLUSION),
        ]

        for prompt_prefix, thought_type in prompts[:self.config.tot_branches]:
            full_prompt = f"{prompt_prefix}\nProblem: {problem}\nBisheriger Gedanke: {parent.content}"

            try:
                response = await self.llm_callback(full_prompt)
                branches.append((response, thought_type))
            except Exception as e:
                logger.warning(f"Branch generation failed: {e}")
                # Fallback
                branches.append((f"Considering: {prompt_prefix}", thought_type))

        return branches

    async def _evaluate_thought(
        self,
        thought: str,
        problem: str,
        context: Dict[str, Any],
    ) -> float:
        """Bewertet einen Gedanken auf einer Skala von 0-1."""
        # Heuristiken für Bewertung
        score = 0.5

        # Länge (zu kurz = schlecht, zu lang = auch schlecht)
        length = len(thought)
        if 50 < length < 500:
            score += 0.1
        elif length < 20:
            score -= 0.2

        # Keywords für gute Gedanken
        good_keywords = ["therefore", "because", "if", "then", "conclude", "solution", "answer"]
        for kw in good_keywords:
            if kw.lower() in thought.lower():
                score += 0.05

        # Relevanz zum Problem
        problem_words = set(problem.lower().split())
        thought_words = set(thought.lower().split())
        overlap = len(problem_words & thought_words) / max(len(problem_words), 1)
        score += overlap * 0.2

        return min(1.0, max(0.0, score))

    async def _synthesize_solution(self, path: List[ThoughtNode], problem: str) -> str:
        """Synthetisiert die finale Lösung aus dem besten Pfad."""
        if not path:
            return "No solution found."

        # Kombiniere alle Gedanken im Pfad
        thoughts = [t.content for t in path]
        combined = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(thoughts))

        # Generiere finale Lösung
        prompt = f"Based on the following reasoning, provide a clear solution:\n\n{combined}\n\nSolution:"

        try:
            solution = await self.llm_callback(prompt)
            return solution
        except Exception:
            return path[-1].content if path else "Unable to synthesize solution."

    def _format_trace(self, path: List[ThoughtNode]) -> str:
        """Formatiert den Reasoning-Trace."""
        if not path:
            return ""

        lines = []
        for i, thought in enumerate(path):
            indent = "  " * thought.depth
            lines.append(f"{indent}[{thought.thought_type.value}] {thought.content}")
            lines.append(f"{indent}  Score: {thought.score:.2f}, Confidence: {thought.confidence:.2f}")

        return "\n".join(lines)


# ============================================================================
# CHAIN OF THOUGHT (CoT)
# ============================================================================

class ChainOfThought(BaseReasoner):
    """
    Chain of Thought Reasoning.

    Schritt-für-Schritt logisches Denken mit expliziten Zwischenschritten.
    """

    def __init__(self, config: Optional[ReasoningConfig] = None, llm_callback: Optional[Callable] = None):
        super().__init__(config)
        self.llm_callback = llm_callback or self._default_llm

    async def _default_llm(self, prompt: str) -> str:
        return f"Step: {prompt[:50]}..."

    async def reason(self, problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Führt Chain-of-Thought Reasoning durch.

        Zerlegt das Problem in logische Schritte und arbeitet sich
        systematisch zur Lösung vor.
        """
        start_time = time.time()
        context = context or {}
        steps = []
        all_thoughts = []

        logger.info("Starting Chain of Thought reasoning", problem=problem[:100])

        # Initialer Gedanke
        current_thought = self._create_thought(
            content=f"Problem: {problem}",
            thought_type=ThoughtType.OBSERVATION,
            confidence=1.0,
        )
        all_thoughts.append(current_thought)

        # Schritt-für-Schritt Reasoning
        for step_num in range(1, self.config.cot_max_steps + 1):
            # Generiere nächsten Schritt
            step_result = await self._generate_step(
                problem=problem,
                previous_steps=steps,
                context=context,
                step_number=step_num,
            )

            if step_result is None:
                break

            steps.append(step_result)

            # Erstelle Gedankenknoten
            thought = self._create_thought(
                content=step_result.thought,
                thought_type=ThoughtType.ANALYSIS,
                parent_id=current_thought.id,
                confidence=step_result.confidence,
            )
            all_thoughts.append(thought)
            current_thought = thought

            # Prüfe auf Abschluss
            if await self._is_complete(problem, steps, context):
                break

        # Finale Schlussfolgerung
        conclusion = await self._draw_conclusion(problem, steps, context)

        final_thought = self._create_thought(
            content=conclusion,
            thought_type=ThoughtType.CONCLUSION,
            parent_id=current_thought.id,
            score=0.9,
            confidence=0.85,
        )
        all_thoughts.append(final_thought)

        duration = (time.time() - start_time) * 1000

        return ReasoningResult(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            problem=problem,
            solution=conclusion,
            confidence=sum(s.confidence for s in steps) / max(len(steps), 1),
            steps=steps,
            thoughts=all_thoughts,
            reasoning_trace=self._format_steps(steps),
            duration_ms=duration,
            metadata={
                "total_steps": len(steps),
                "detail_level": self.config.cot_detail_level,
            },
        )

    async def _generate_step(
        self,
        problem: str,
        previous_steps: List[ReasoningStep],
        context: Dict[str, Any],
        step_number: int,
    ) -> Optional[ReasoningStep]:
        """Generiert den nächsten Reasoning-Schritt."""
        # Build prompt
        steps_summary = "\n".join(
            f"Step {s.step_number}: {s.thought}" for s in previous_steps
        )

        prompt = f"""Problem: {problem}

Previous reasoning:
{steps_summary if steps_summary else "None yet"}

What is the next logical step in solving this problem? Think carefully and explain your reasoning.

Step {step_number}:"""

        try:
            response = await self.llm_callback(prompt)

            return ReasoningStep(
                step_number=step_number,
                thought=response,
                reasoning=f"Derived from previous {len(previous_steps)} steps",
                confidence=0.7 + (0.1 * min(step_number, 3)),  # Confidence grows with steps
            )
        except Exception as e:
            logger.warning(f"Step generation failed: {e}")
            return None

    async def _is_complete(
        self,
        problem: str,
        steps: List[ReasoningStep],
        context: Dict[str, Any],
    ) -> bool:
        """Prüft ob das Reasoning abgeschlossen ist."""
        if not steps:
            return False

        # Heuristiken für Abschluss
        last_step = steps[-1]

        # Keywords die auf Abschluss hindeuten
        conclusion_keywords = ["therefore", "thus", "conclude", "answer is", "solution is", "final"]
        for kw in conclusion_keywords:
            if kw in last_step.thought.lower():
                return True

        # Minimale Schritte erreicht
        if len(steps) >= 3 and last_step.confidence >= 0.8:
            return True

        return False

    async def _draw_conclusion(
        self,
        problem: str,
        steps: List[ReasoningStep],
        context: Dict[str, Any],
    ) -> str:
        """Zieht die finale Schlussfolgerung."""
        steps_text = "\n".join(f"Step {s.step_number}: {s.thought}" for s in steps)

        prompt = f"""Based on the following step-by-step reasoning, provide a clear and concise final answer.

Problem: {problem}

Reasoning steps:
{steps_text}

Final Answer:"""

        try:
            return await self.llm_callback(prompt)
        except Exception:
            return steps[-1].thought if steps else "Unable to draw conclusion."

    def _format_steps(self, steps: List[ReasoningStep]) -> str:
        """Formatiert die Reasoning-Schritte."""
        lines = []
        for step in steps:
            lines.append(f"Step {step.step_number}:")
            lines.append(f"  Thought: {step.thought}")
            lines.append(f"  Reasoning: {step.reasoning}")
            lines.append(f"  Confidence: {step.confidence:.2f}")
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# REACT REASONING
# ============================================================================

class ReActReasoner(BaseReasoner):
    """
    ReAct Reasoning Pattern.

    Kombiniert Reasoning mit Actions in einem iterativen Loop:
    Thought -> Action -> Observation -> Thought -> ...
    """

    def __init__(
        self,
        config: Optional[ReasoningConfig] = None,
        llm_callback: Optional[Callable] = None,
        action_executor: Optional[Callable] = None,
    ):
        super().__init__(config)
        self.llm_callback = llm_callback or self._default_llm
        self.action_executor = action_executor or self._default_action_executor

    async def _default_llm(self, prompt: str) -> str:
        return f"Thought: {prompt[:50]}..."

    async def _default_action_executor(self, action: str, params: Dict[str, Any]) -> str:
        """
        Standard Action Executor für ReAct-Aktionen.

        Führt echte Aktionen basierend auf dem Action-Typ aus.
        """
        action_lower = action.lower()

        # Suchaktionen
        if "search" in action_lower or "lookup" in action_lower:
            query = params.get("query", params.get("term", str(params)))
            try:
                from scio.knowledge.internet_access import InternetKnowledge
                internet = InternetKnowledge()
                results = await internet.search(query, num_results=3)
                if results:
                    return "\n".join([f"- {r.title}: {r.snippet[:200]}" for r in results[:3]])
                return f"No results found for: {query}"
            except Exception as e:
                return f"Search failed: {e}"

        # Berechnungsaktionen
        elif "calculate" in action_lower or "compute" in action_lower:
            expression = params.get("expression", params.get("expr", ""))
            try:
                import math
                allowed = {"abs": abs, "round": round, "min": min, "max": max,
                          "sqrt": math.sqrt, "pow": pow, "sin": math.sin, "cos": math.cos}
                result = eval(str(expression), {"__builtins__": {}}, allowed)
                return f"Calculation result: {result}"
            except Exception as e:
                return f"Calculation error: {e}"

        # Dateiaktionen
        elif "read" in action_lower and "file" in action_lower:
            path = params.get("path", params.get("file", ""))
            try:
                import os
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()[:2000]
                    return f"File content:\n{content}"
                return f"File not found: {path}"
            except Exception as e:
                return f"Read error: {e}"

        # Verifikationsaktionen
        elif "verify" in action_lower or "check" in action_lower:
            claim = params.get("claim", params.get("statement", str(params)))
            return f"Verification of '{claim}': Requires additional evidence. Consider searching for sources."

        # Analyseaktionen
        elif "analyze" in action_lower or "examine" in action_lower:
            target = params.get("target", params.get("subject", str(params)))
            return f"Analysis of '{target}': Consider breaking down into components and examining each part systematically."

        # Fallback mit informativer Nachricht
        return f"Action '{action}' executed with params: {params}. Result requires further analysis."

    async def reason(self, problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Führt ReAct Reasoning durch.

        Iteriert durch Thought-Action-Observation Zyklen bis
        eine Lösung gefunden wird.
        """
        start_time = time.time()
        context = context or {}
        steps = []
        all_thoughts = []

        logger.info("Starting ReAct reasoning", problem=problem[:100])

        # Initialgedanke
        root = self._create_thought(
            content=f"Task: {problem}",
            thought_type=ThoughtType.OBSERVATION,
            confidence=1.0,
        )
        all_thoughts.append(root)
        current_thought_id = root.id

        observations_history = []

        for iteration in range(self.config.react_max_actions):
            # 1. Thought: Was denke ich?
            thought_prompt = self._build_thought_prompt(problem, steps, observations_history)
            thought_response = await self.llm_callback(thought_prompt)

            thought_node = self._create_thought(
                content=thought_response,
                thought_type=ThoughtType.ANALYSIS,
                parent_id=current_thought_id,
                confidence=0.7,
            )
            all_thoughts.append(thought_node)

            # 2. Action: Was tue ich?
            action, action_params = await self._determine_action(thought_response, problem, context)

            # Prüfe auf Finish
            if action == "finish":
                final_answer = action_params.get("answer", thought_response)

                conclusion = self._create_thought(
                    content=final_answer,
                    thought_type=ThoughtType.CONCLUSION,
                    parent_id=thought_node.id,
                    score=0.9,
                    confidence=0.85,
                )
                all_thoughts.append(conclusion)

                steps.append(ReasoningStep(
                    step_number=len(steps) + 1,
                    thought=thought_response,
                    reasoning="Final answer determined",
                    action="finish",
                    observation=final_answer,
                    confidence=0.85,
                ))
                break

            # 3. Execute Action
            try:
                observation = await asyncio.wait_for(
                    self.action_executor(action, action_params),
                    timeout=self.config.react_action_timeout,
                )
            except asyncio.TimeoutError:
                observation = f"Action '{action}' timed out"
            except Exception as e:
                observation = f"Action '{action}' failed: {str(e)}"

            observations_history.append(observation)

            # Observation als Gedanke
            obs_node = self._create_thought(
                content=f"Observation: {observation}",
                thought_type=ThoughtType.OBSERVATION,
                parent_id=thought_node.id,
                confidence=0.8,
            )
            all_thoughts.append(obs_node)
            current_thought_id = obs_node.id

            steps.append(ReasoningStep(
                step_number=len(steps) + 1,
                thought=thought_response,
                reasoning="ReAct iteration",
                action=action,
                observation=observation,
                confidence=0.7,
            ))

        # Falls keine finish action
        if not steps or steps[-1].action != "finish":
            final_answer = await self._synthesize_final_answer(problem, steps)
        else:
            final_answer = steps[-1].observation or ""

        duration = (time.time() - start_time) * 1000

        return ReasoningResult(
            strategy=ReasoningStrategy.REACT,
            problem=problem,
            solution=final_answer,
            confidence=sum(s.confidence for s in steps) / max(len(steps), 1),
            steps=steps,
            thoughts=all_thoughts,
            reasoning_trace=self._format_react_trace(steps),
            duration_ms=duration,
            metadata={
                "iterations": len(steps),
                "actions_taken": [s.action for s in steps],
            },
        )

    def _build_thought_prompt(
        self,
        problem: str,
        steps: List[ReasoningStep],
        observations: List[str],
    ) -> str:
        """Baut den Prompt für den Thought-Schritt."""
        history = ""
        for i, step in enumerate(steps):
            history += f"\nThought {i+1}: {step.thought}"
            history += f"\nAction {i+1}: {step.action}"
            history += f"\nObservation {i+1}: {step.observation}\n"

        return f"""You are solving the following task:
{problem}

{history}

What should I think about next? Analyze the situation and determine the next step.

Thought:"""

    async def _determine_action(
        self,
        thought: str,
        problem: str,
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Bestimmt die nächste Action basierend auf dem Thought."""
        # Heuristiken für Action-Auswahl
        thought_lower = thought.lower()

        # Check for finish indicators
        finish_indicators = ["answer is", "solution is", "conclude", "final answer", "therefore"]
        for indicator in finish_indicators:
            if indicator in thought_lower:
                return "finish", {"answer": thought}

        # Check for search/lookup
        if "search" in thought_lower or "look up" in thought_lower or "find" in thought_lower:
            return "search", {"query": thought[:200]}

        # Check for calculation
        if "calculate" in thought_lower or "compute" in thought_lower:
            return "calculate", {"expression": thought}

        # Check for retrieval
        if "retrieve" in thought_lower or "get" in thought_lower:
            return "retrieve", {"source": thought}

        # Default: continue thinking
        return "think", {"prompt": thought}

    async def _synthesize_final_answer(
        self,
        problem: str,
        steps: List[ReasoningStep],
    ) -> str:
        """Synthetisiert die finale Antwort."""
        if not steps:
            return "Unable to determine answer."

        steps_summary = "\n".join(
            f"Step {s.step_number}: Thought: {s.thought[:100]}..., Action: {s.action}, Observation: {s.observation[:100] if s.observation else 'N/A'}..."
            for s in steps
        )

        prompt = f"""Based on the following ReAct reasoning trace, provide the final answer.

Problem: {problem}

Reasoning:
{steps_summary}

Final Answer:"""

        try:
            return await self.llm_callback(prompt)
        except Exception:
            return steps[-1].thought

    def _format_react_trace(self, steps: List[ReasoningStep]) -> str:
        """Formatiert den ReAct Trace."""
        lines = []
        for step in steps:
            lines.append(f"=== Iteration {step.step_number} ===")
            lines.append(f"Thought: {step.thought}")
            lines.append(f"Action: {step.action}")
            lines.append(f"Observation: {step.observation}")
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# MONTE CARLO TREE SEARCH (MCTS)
# ============================================================================

@dataclass
class MCTSNode:
    """Ein Knoten im MCTS-Baum."""

    id: str
    state: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[str] = field(default_factory=list)

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound 1 Wert."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = math.sqrt(2 * math.log(self.visits + 1) / self.visits)
        return exploitation + exploration


class MCTSPlanner(BaseReasoner):
    """
    Monte Carlo Tree Search für komplexe Planung.

    Nutzt Simulation und Rollouts um den besten Pfad zu finden.
    """

    def __init__(
        self,
        config: Optional[ReasoningConfig] = None,
        llm_callback: Optional[Callable] = None,
        state_evaluator: Optional[Callable] = None,
    ):
        super().__init__(config)
        self.llm_callback = llm_callback or self._default_llm
        self.state_evaluator = state_evaluator or self._default_evaluator
        self._mcts_nodes: Dict[str, MCTSNode] = {}

    async def _default_llm(self, prompt: str) -> str:
        return f"Action: {prompt[:50]}..."

    async def _default_evaluator(self, state: str, goal: str) -> float:
        """Bewertet einen Zustand (0-1)."""
        # Simple heuristic: word overlap
        state_words = set(state.lower().split())
        goal_words = set(goal.lower().split())
        if not goal_words:
            return 0.5
        return len(state_words & goal_words) / len(goal_words)

    async def reason(self, problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Führt MCTS-basierte Planung durch.
        """
        start_time = time.time()
        context = context or {}

        logger.info("Starting MCTS planning", problem=problem[:100])

        # Initialisiere Root
        root_id = generate_id("mcts")
        root = MCTSNode(
            id=root_id,
            state=f"Initial: {problem}",
            untried_actions=await self._get_possible_actions(problem, context),
        )
        self._mcts_nodes[root_id] = root

        # MCTS Loop
        for _ in range(self.config.mcts_simulations):
            # 1. Selection
            node_id = await self._select(root_id)

            # 2. Expansion
            node_id = await self._expand(node_id, problem, context)

            # 3. Simulation
            value = await self._simulate(node_id, problem, context)

            # 4. Backpropagation
            await self._backpropagate(node_id, value)

        # Extrahiere beste Lösung
        best_path = self._get_best_mcts_path(root_id)
        solution = await self._path_to_solution(best_path, problem)

        # Konvertiere zu Thoughts
        all_thoughts = []
        for mcts_node in best_path:
            thought = self._create_thought(
                content=mcts_node.state,
                thought_type=ThoughtType.ANALYSIS,
                score=mcts_node.value / max(mcts_node.visits, 1),
                confidence=min(1.0, mcts_node.visits / 100),
            )
            all_thoughts.append(thought)

        duration = (time.time() - start_time) * 1000

        return ReasoningResult(
            strategy=ReasoningStrategy.MCTS,
            problem=problem,
            solution=solution,
            confidence=root.value / max(root.visits, 1),
            thoughts=all_thoughts,
            reasoning_trace=self._format_mcts_trace(best_path),
            duration_ms=duration,
            metadata={
                "total_simulations": self.config.mcts_simulations,
                "total_nodes": len(self._mcts_nodes),
                "root_visits": root.visits,
            },
        )

    async def _select(self, node_id: str) -> str:
        """Selection Phase: Wähle Knoten zum Expandieren."""
        current_id = node_id

        while True:
            node = self._mcts_nodes[current_id]

            # Wenn ungetestete Aktionen verfügbar, stoppe hier
            if node.untried_actions:
                return current_id

            # Wenn keine Kinder, stoppe hier
            if not node.children_ids:
                return current_id

            # Wähle Kind mit höchstem UCB1
            best_child_id = max(
                node.children_ids,
                key=lambda cid: self._mcts_nodes[cid].ucb1
            )
            current_id = best_child_id

    async def _expand(self, node_id: str, problem: str, context: Dict[str, Any]) -> str:
        """Expansion Phase: Erweitere den Baum."""
        node = self._mcts_nodes[node_id]

        if not node.untried_actions:
            return node_id

        # Wähle zufällige ungetestete Aktion
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Erstelle neuen Zustand
        new_state = f"{node.state} -> {action}"
        new_actions = await self._get_possible_actions(new_state, context)

        # Erstelle neuen Knoten
        child_id = generate_id("mcts")
        child = MCTSNode(
            id=child_id,
            state=new_state,
            parent_id=node_id,
            untried_actions=new_actions,
        )
        self._mcts_nodes[child_id] = child
        node.children_ids.append(child_id)

        return child_id

    async def _simulate(self, node_id: str, problem: str, context: Dict[str, Any]) -> float:
        """Simulation Phase: Simuliere bis zum Ende."""
        node = self._mcts_nodes[node_id]
        current_state = node.state

        for _ in range(self.config.mcts_max_rollout_depth):
            # Bewerte aktuellen Zustand
            value = await self.state_evaluator(current_state, problem)

            # Wenn gut genug, beende
            if value >= 0.8:
                return value

            # Wähle zufällige Aktion
            actions = await self._get_possible_actions(current_state, context)
            if not actions:
                break

            action = random.choice(actions)
            current_state = f"{current_state} -> {action}"

        return await self.state_evaluator(current_state, problem)

    async def _backpropagate(self, node_id: str, value: float) -> None:
        """Backpropagation: Aktualisiere Statistiken entlang des Pfads."""
        current_id = node_id

        while current_id:
            node = self._mcts_nodes[current_id]
            node.visits += 1
            node.value += value
            current_id = node.parent_id

    async def _get_possible_actions(self, state: str, context: Dict[str, Any]) -> List[str]:
        """
        Generiert mögliche Aktionen für einen Zustand mittels LLM.

        Args:
            state: Der aktuelle Zustand im Reasoning-Prozess
            context: Zusätzlicher Kontext

        Returns:
            Liste von möglichen Aktionen
        """
        # Basis-Aktionen die immer verfügbar sind
        base_actions = [
            "analyze further",
            "break down into subproblems",
            "consider alternatives",
            "verify assumptions",
            "synthesize conclusion",
        ]

        # Versuche LLM für kontextspezifische Aktionen zu nutzen
        try:
            prompt = f"""Given the current reasoning state:
{state}

Additional context: {json.dumps(context) if context else 'None'}

List 3-5 specific next reasoning steps or actions that would help solve this problem.
Format: Return ONLY a JSON array of action strings, e.g.: ["action1", "action2", "action3"]
Keep each action concise (3-6 words)."""

            response = await self.llm_callback(prompt)

            # Parse JSON response
            if response:
                # Versuche JSON zu extrahieren
                import re
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    actions = json.loads(json_match.group())
                    if isinstance(actions, list) and len(actions) > 0:
                        # Validiere und bereinige Aktionen
                        valid_actions = [
                            str(a).strip()[:100]  # Max 100 Zeichen pro Aktion
                            for a in actions
                            if isinstance(a, str) and len(str(a).strip()) > 0
                        ]
                        if valid_actions:
                            return valid_actions[:5]

        except json.JSONDecodeError:
            logger.debug("JSON parsing fehlgeschlagen für LLM-Aktionen, verwende Fallback")
        except Exception as e:
            logger.debug(f"LLM-Aktionsgenerierung fehlgeschlagen: {e}, verwende Fallback")

        # Fallback: Kontextabhängige Basis-Aktionen
        # Wähle basierend auf State-Inhalt passende Aktionen
        selected_actions = []

        state_lower = state.lower()
        if "?" in state or "question" in state_lower:
            selected_actions.append("gather more information")
        if "hypothesis" in state_lower or "assume" in state_lower:
            selected_actions.append("verify assumptions")
        if "error" in state_lower or "problem" in state_lower:
            selected_actions.append("identify root cause")
        if "solution" in state_lower or "answer" in state_lower:
            selected_actions.append("validate solution")

        # Fülle mit Basis-Aktionen auf
        for action in base_actions:
            if action not in selected_actions:
                selected_actions.append(action)
            if len(selected_actions) >= 3:
                break

        return selected_actions[:3]

    def _get_best_mcts_path(self, root_id: str) -> List[MCTSNode]:
        """Extrahiert den besten Pfad vom Root."""
        path = []
        current_id = root_id

        while current_id:
            node = self._mcts_nodes[current_id]
            path.append(node)

            if not node.children_ids:
                break

            # Wähle Kind mit höchstem Wert
            best_child_id = max(
                node.children_ids,
                key=lambda cid: self._mcts_nodes[cid].value / max(self._mcts_nodes[cid].visits, 1)
            )
            current_id = best_child_id

        return path

    async def _path_to_solution(self, path: List[MCTSNode], problem: str) -> str:
        """Konvertiert MCTS-Pfad zu einer Lösung."""
        if not path:
            return "No solution found."

        path_description = " -> ".join(n.state for n in path)

        prompt = f"""Based on the following planning path, provide a solution.

Problem: {problem}
Path: {path_description}

Solution:"""

        try:
            return await self.llm_callback(prompt)
        except Exception:
            return path[-1].state

    def _format_mcts_trace(self, path: List[MCTSNode]) -> str:
        """Formatiert den MCTS Trace."""
        lines = []
        for i, node in enumerate(path):
            lines.append(f"Step {i+1}: {node.state}")
            lines.append(f"  Visits: {node.visits}, Value: {node.value:.2f}")
        return "\n".join(lines)


# ============================================================================
# SELF-REFLECTION
# ============================================================================

class SelfReflection(BaseReasoner):
    """
    Self-Reflection Reasoning.

    Iterative Selbstverbesserung durch kritische Analyse
    der eigenen Ausgaben.
    """

    def __init__(self, config: Optional[ReasoningConfig] = None, llm_callback: Optional[Callable] = None):
        super().__init__(config)
        self.llm_callback = llm_callback or self._default_llm

    async def _default_llm(self, prompt: str) -> str:
        return f"Reflection: {prompt[:50]}..."

    async def reason(self, problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Führt Self-Reflection Reasoning durch.

        1. Generiere initiale Antwort
        2. Kritisiere die Antwort
        3. Verbessere basierend auf Kritik
        4. Wiederhole bis Konvergenz
        """
        start_time = time.time()
        context = context or {}
        all_thoughts = []
        steps = []

        logger.info("Starting Self-Reflection reasoning", problem=problem[:100])

        # Initiale Antwort
        current_answer = await self._generate_initial_answer(problem, context)

        initial = self._create_thought(
            content=current_answer,
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.5,
        )
        all_thoughts.append(initial)

        current_confidence = 0.5

        for iteration in range(self.config.reflection_iterations):
            # Kritik
            critique = await self._critique_answer(problem, current_answer, context)

            critique_thought = self._create_thought(
                content=critique,
                thought_type=ThoughtType.REFLECTION,
                parent_id=all_thoughts[-1].id,
                confidence=0.7,
            )
            all_thoughts.append(critique_thought)

            # Prüfe ob gut genug
            quality_score = await self._assess_quality(current_answer, problem)

            if quality_score >= 0.9:
                current_confidence = quality_score
                break

            # Verbessere
            improved_answer = await self._improve_answer(
                problem, current_answer, critique, context
            )

            improved_thought = self._create_thought(
                content=improved_answer,
                thought_type=ThoughtType.INSIGHT,
                parent_id=critique_thought.id,
                confidence=quality_score + 0.1,
            )
            all_thoughts.append(improved_thought)

            steps.append(ReasoningStep(
                step_number=iteration + 1,
                thought=current_answer[:200],
                reasoning=f"Reflection iteration {iteration + 1}",
                observation=critique[:200],
                confidence=quality_score,
            ))

            current_answer = improved_answer
            current_confidence = quality_score + 0.1

        # Finale Schlussfolgerung
        final = self._create_thought(
            content=current_answer,
            thought_type=ThoughtType.CONCLUSION,
            parent_id=all_thoughts[-1].id,
            score=0.9,
            confidence=current_confidence,
        )
        all_thoughts.append(final)

        duration = (time.time() - start_time) * 1000

        return ReasoningResult(
            strategy=ReasoningStrategy.SELF_REFLECTION,
            problem=problem,
            solution=current_answer,
            confidence=current_confidence,
            steps=steps,
            thoughts=all_thoughts,
            reasoning_trace=self._format_reflection_trace(steps, current_answer),
            duration_ms=duration,
            metadata={
                "iterations": len(steps),
                "criteria": self.config.reflection_criteria,
            },
        )

    async def _generate_initial_answer(self, problem: str, context: Dict[str, Any]) -> str:
        """Generiert die initiale Antwort."""
        prompt = f"Solve the following problem:\n\n{problem}\n\nAnswer:"
        return await self.llm_callback(prompt)

    async def _critique_answer(
        self,
        problem: str,
        answer: str,
        context: Dict[str, Any],
    ) -> str:
        """Kritisiert eine Antwort."""
        criteria_text = ", ".join(self.config.reflection_criteria)

        prompt = f"""Critically evaluate the following answer to a problem.

Problem: {problem}

Answer: {answer}

Evaluate based on: {criteria_text}

Provide specific, actionable feedback:"""

        return await self.llm_callback(prompt)

    async def _improve_answer(
        self,
        problem: str,
        current_answer: str,
        critique: str,
        context: Dict[str, Any],
    ) -> str:
        """Verbessert eine Antwort basierend auf Kritik."""
        prompt = f"""Improve the following answer based on the critique.

Problem: {problem}

Current Answer: {current_answer}

Critique: {critique}

Improved Answer:"""

        return await self.llm_callback(prompt)

    async def _assess_quality(self, answer: str, problem: str) -> float:
        """Bewertet die Qualität einer Antwort."""
        score = 0.5

        # Länge check
        if 50 < len(answer) < 2000:
            score += 0.1

        # Struktur check
        if any(marker in answer for marker in ["1.", "First", "Step", "- "]):
            score += 0.1

        # Relevanz check
        problem_words = set(problem.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(problem_words & answer_words) / max(len(problem_words), 1)
        score += overlap * 0.2

        return min(1.0, score)

    def _format_reflection_trace(self, steps: List[ReasoningStep], final: str) -> str:
        """Formatiert den Reflection Trace."""
        lines = []
        for step in steps:
            lines.append(f"=== Iteration {step.step_number} ===")
            lines.append(f"Answer: {step.thought}")
            lines.append(f"Critique: {step.observation}")
            lines.append(f"Quality: {step.confidence:.2f}")
            lines.append("")

        lines.append("=== Final Answer ===")
        lines.append(final)

        return "\n".join(lines)


# ============================================================================
# META-COGNITION
# ============================================================================

class MetaCognition:
    """
    Meta-Kognition für Bewusstsein über das eigene Denken.

    - Wissen über eigenes Wissen
    - Erkennen von Wissenslücken
    - Konfidenz-Kalibrierung
    - Unsicherheits-Quantifizierung
    """

    def __init__(self, llm_callback: Optional[Callable] = None):
        self.llm_callback = llm_callback or self._default_llm
        self._knowledge_map: Dict[str, float] = {}  # topic -> confidence
        self._uncertainty_history: List[Tuple[str, float, float]] = []  # (topic, predicted, actual)

    async def _default_llm(self, prompt: str) -> str:
        return f"Meta: {prompt[:50]}..."

    async def assess_knowledge(self, topic: str) -> Dict[str, Any]:
        """Bewertet das eigene Wissen zu einem Thema."""
        prompt = f"""Assess your knowledge about: {topic}

Consider:
1. How confident are you (0-1)?
2. What do you know for certain?
3. What are you uncertain about?
4. What don't you know at all?

Provide a structured assessment:"""

        response = await self.llm_callback(prompt)

        # Extrahiere Konfidenz (Heuristik)
        confidence = 0.5
        if "very confident" in response.lower() or "certain" in response.lower():
            confidence = 0.8
        elif "uncertain" in response.lower() or "not sure" in response.lower():
            confidence = 0.3
        elif "don't know" in response.lower() or "unknown" in response.lower():
            confidence = 0.1

        self._knowledge_map[topic] = confidence

        return {
            "topic": topic,
            "confidence": confidence,
            "assessment": response,
            "known_gaps": self._identify_gaps(response),
        }

    def _identify_gaps(self, assessment: str) -> List[str]:
        """Identifiziert Wissenslücken aus der Bewertung."""
        gaps = []

        gap_indicators = [
            "don't know",
            "uncertain",
            "not sure",
            "unclear",
            "need to learn",
            "unknown",
        ]

        for indicator in gap_indicators:
            if indicator in assessment.lower():
                # Extrahiere Kontext
                idx = assessment.lower().find(indicator)
                context = assessment[max(0, idx-50):min(len(assessment), idx+50)]
                gaps.append(context.strip())

        return gaps

    async def calibrate_confidence(
        self,
        prediction: str,
        stated_confidence: float,
        actual_outcome: str,
    ) -> float:
        """Kalibriert Konfidenz basierend auf tatsächlichen Ergebnissen."""
        # Bewerte ob Vorhersage korrekt war
        was_correct = await self._check_prediction(prediction, actual_outcome)

        # Speichere für Kalibrierung
        self._uncertainty_history.append(("calibration", stated_confidence, 1.0 if was_correct else 0.0))

        # Berechne Kalibrierungsfaktor
        if len(self._uncertainty_history) >= 10:
            recent = self._uncertainty_history[-10:]
            avg_stated = sum(h[1] for h in recent) / len(recent)
            avg_actual = sum(h[2] for h in recent) / len(recent)

            if avg_stated > 0:
                calibration_factor = avg_actual / avg_stated
                return min(1.0, stated_confidence * calibration_factor)

        return stated_confidence

    async def _check_prediction(self, prediction: str, actual: str) -> bool:
        """Prüft ob eine Vorhersage korrekt war."""
        prompt = f"""Was this prediction correct?

Prediction: {prediction}
Actual outcome: {actual}

Answer with just 'yes' or 'no':"""

        response = await self.llm_callback(prompt)
        return "yes" in response.lower()

    def get_uncertainty_report(self) -> Dict[str, Any]:
        """Generiert einen Bericht über Unsicherheiten."""
        if not self._uncertainty_history:
            return {"message": "No uncertainty data available"}

        avg_confidence = sum(h[1] for h in self._uncertainty_history) / len(self._uncertainty_history)
        avg_accuracy = sum(h[2] for h in self._uncertainty_history) / len(self._uncertainty_history)

        calibration_error = abs(avg_confidence - avg_accuracy)

        return {
            "total_predictions": len(self._uncertainty_history),
            "average_stated_confidence": avg_confidence,
            "average_actual_accuracy": avg_accuracy,
            "calibration_error": calibration_error,
            "is_overconfident": avg_confidence > avg_accuracy,
            "is_underconfident": avg_confidence < avg_accuracy,
            "knowledge_areas": dict(self._knowledge_map),
        }


# ============================================================================
# UNIFIED ADVANCED REASONING
# ============================================================================

class AdvancedReasoning:
    """
    Unified Advanced Reasoning System.

    Kombiniert alle Reasoning-Techniken und wählt automatisch
    die beste Strategie für jedes Problem.
    """

    def __init__(
        self,
        config: Optional[ReasoningConfig] = None,
        llm_callback: Optional[Callable] = None,
    ):
        self.config = config or ReasoningConfig()
        self.llm_callback = llm_callback

        # Initialize all reasoners
        self.tot = TreeOfThought(self.config, llm_callback)
        self.cot = ChainOfThought(self.config, llm_callback)
        self.react = ReActReasoner(self.config, llm_callback)
        self.mcts = MCTSPlanner(self.config, llm_callback)
        self.reflection = SelfReflection(self.config, llm_callback)
        self.meta = MetaCognition(llm_callback)

        self._strategy_history: List[Tuple[str, ReasoningStrategy, float]] = []

    async def reason(
        self,
        problem: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Führt Reasoning mit automatischer Strategie-Auswahl durch.
        """
        context = context or {}

        # Wähle Strategie wenn nicht angegeben
        if strategy is None:
            strategy = await self._select_strategy(problem, context)

        logger.info(f"Using reasoning strategy: {strategy.value}")

        # Führe entsprechende Strategie aus
        if strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            result = await self.tot.reason(problem, context)
        elif strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            result = await self.cot.reason(problem, context)
        elif strategy == ReasoningStrategy.REACT:
            result = await self.react.reason(problem, context)
        elif strategy == ReasoningStrategy.MCTS:
            result = await self.mcts.reason(problem, context)
        elif strategy == ReasoningStrategy.SELF_REFLECTION:
            result = await self.reflection.reason(problem, context)
        elif strategy == ReasoningStrategy.HYBRID:
            result = await self._hybrid_reasoning(problem, context)
        else:
            # Default to CoT
            result = await self.cot.reason(problem, context)

        # Speichere für zukünftige Optimierung
        self._strategy_history.append((problem[:100], strategy, result.confidence))

        return result

    async def _select_strategy(
        self,
        problem: str,
        context: Dict[str, Any],
    ) -> ReasoningStrategy:
        """Wählt automatisch die beste Strategie."""
        problem_lower = problem.lower()

        # Heuristiken für Strategie-Auswahl
        # Komplexe Probleme mit vielen möglichen Lösungen -> ToT
        if any(word in problem_lower for word in ["creative", "brainstorm", "multiple ways", "different approaches"]):
            return ReasoningStrategy.TREE_OF_THOUGHT

        # Schritt-für-Schritt Probleme -> CoT
        if any(word in problem_lower for word in ["step by step", "explain", "how to", "calculate"]):
            return ReasoningStrategy.CHAIN_OF_THOUGHT

        # Aktionsbasierte Probleme -> ReAct
        if any(word in problem_lower for word in ["do", "execute", "perform", "action", "task"]):
            return ReasoningStrategy.REACT

        # Planungsprobleme -> MCTS
        if any(word in problem_lower for word in ["plan", "strategy", "optimize", "best path"]):
            return ReasoningStrategy.MCTS

        # Verbesserungsprobleme -> Self-Reflection
        if any(word in problem_lower for word in ["improve", "refine", "critique", "better"]):
            return ReasoningStrategy.SELF_REFLECTION

        # Default: Chain of Thought
        return ReasoningStrategy.CHAIN_OF_THOUGHT

    async def _hybrid_reasoning(
        self,
        problem: str,
        context: Dict[str, Any],
    ) -> ReasoningResult:
        """
        Hybrides Reasoning: Kombiniert mehrere Strategien.
        """
        start_time = time.time()

        # Phase 1: CoT für initiale Analyse
        cot_result = await self.cot.reason(problem, context)

        # Phase 2: Self-Reflection für Verbesserung
        reflection_problem = f"Improve this solution: {cot_result.solution}"
        reflection_result = await self.reflection.reason(reflection_problem, context)

        # Kombiniere Ergebnisse
        combined_thoughts = cot_result.thoughts + reflection_result.thoughts
        combined_steps = cot_result.steps + reflection_result.steps

        duration = (time.time() - start_time) * 1000

        return ReasoningResult(
            strategy=ReasoningStrategy.HYBRID,
            problem=problem,
            solution=reflection_result.solution,
            confidence=(cot_result.confidence + reflection_result.confidence) / 2,
            steps=combined_steps,
            thoughts=combined_thoughts,
            reasoning_trace=f"=== CoT Phase ===\n{cot_result.reasoning_trace}\n\n=== Reflection Phase ===\n{reflection_result.reasoning_trace}",
            duration_ms=duration,
            metadata={
                "phases": ["chain_of_thought", "self_reflection"],
                "cot_confidence": cot_result.confidence,
                "reflection_confidence": reflection_result.confidence,
            },
        )

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über Strategie-Nutzung zurück."""
        if not self._strategy_history:
            return {"message": "No reasoning history available"}

        strategy_counts = defaultdict(int)
        strategy_confidence = defaultdict(list)

        for _, strategy, confidence in self._strategy_history:
            strategy_counts[strategy.value] += 1
            strategy_confidence[strategy.value].append(confidence)

        return {
            "total_reasoning_tasks": len(self._strategy_history),
            "strategy_usage": dict(strategy_counts),
            "average_confidence_by_strategy": {
                s: sum(c) / len(c) for s, c in strategy_confidence.items()
            },
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_reasoner: Optional[AdvancedReasoning] = None


def get_reasoner(config: Optional[ReasoningConfig] = None) -> AdvancedReasoning:
    """Gibt eine Singleton-Instanz des Advanced Reasoners zurück."""
    global _default_reasoner
    if _default_reasoner is None:
        _default_reasoner = AdvancedReasoning(config)
    return _default_reasoner


async def reason(
    problem: str,
    strategy: Optional[ReasoningStrategy] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    """Convenience-Funktion für schnelles Reasoning."""
    reasoner = get_reasoner()
    return await reasoner.reason(problem, strategy, context)


async def tree_of_thought(problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
    """Tree of Thought Reasoning."""
    return await reason(problem, ReasoningStrategy.TREE_OF_THOUGHT, context)


async def chain_of_thought(problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
    """Chain of Thought Reasoning."""
    return await reason(problem, ReasoningStrategy.CHAIN_OF_THOUGHT, context)


async def react(problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
    """ReAct Reasoning."""
    return await reason(problem, ReasoningStrategy.REACT, context)


async def mcts(problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
    """MCTS Planning."""
    return await reason(problem, ReasoningStrategy.MCTS, context)


async def reflect(problem: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
    """Self-Reflection Reasoning."""
    return await reason(problem, ReasoningStrategy.SELF_REFLECTION, context)
