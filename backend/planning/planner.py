#!/usr/bin/env python3
"""
SCIO - Planner
A*, MCTS und hierarchische Planungsalgorithmen
"""

import heapq
import random
import math
import logging
from typing import Optional, Dict, Any, List, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PlanStatus(str, Enum):
    """Status eines Plans"""
    PENDING = 'pending'
    EXECUTING = 'executing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass
class PlanStep:
    """Ein Schritt in einem Plan"""
    id: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_cost: float = 1.0
    estimated_time_s: float = 1.0
    priority: int = 0

    # Execution tracking
    status: PlanStatus = PlanStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Plan:
    """Ein vollständiger Plan"""
    id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    total_cost: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def add_step(self, step: PlanStep):
        self.steps.append(step)
        self.total_cost += step.estimated_cost

    def get_ready_steps(self) -> List[PlanStep]:
        """Gibt Schritte zurück, die bereit zur Ausführung sind"""
        completed_ids = {s.id for s in self.steps if s.status == PlanStatus.COMPLETED}
        ready = []

        for step in self.steps:
            if step.status != PlanStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                ready.append(step)

        return ready

    def is_complete(self) -> bool:
        return all(s.status == PlanStatus.COMPLETED for s in self.steps)

    def has_failed(self) -> bool:
        return any(s.status == PlanStatus.FAILED for s in self.steps)


class PlanningState(ABC):
    """Abstrakte Klasse für Planungszustände"""

    @abstractmethod
    def get_available_actions(self) -> List[str]:
        pass

    @abstractmethod
    def apply_action(self, action: str) -> 'PlanningState':
        pass

    @abstractmethod
    def is_goal(self) -> bool:
        pass

    @abstractmethod
    def heuristic(self) -> float:
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


@dataclass
class TaskState(PlanningState):
    """Konkreter Zustand für Task-Planung"""
    completed_tasks: frozenset = field(default_factory=frozenset)
    available_resources: Dict[str, float] = field(default_factory=dict)
    goal_tasks: frozenset = field(default_factory=frozenset)
    all_tasks: Dict[str, Dict] = field(default_factory=dict)

    def get_available_actions(self) -> List[str]:
        """Gibt verfügbare Tasks zurück"""
        available = []
        for task_id, task_info in self.all_tasks.items():
            if task_id in self.completed_tasks:
                continue
            # Prüfe Dependencies
            deps = task_info.get("dependencies", [])
            if all(d in self.completed_tasks for d in deps):
                # Prüfe Ressourcen
                required = task_info.get("resources", {})
                can_execute = all(
                    self.available_resources.get(r, 0) >= amount
                    for r, amount in required.items()
                )
                if can_execute:
                    available.append(task_id)
        return available

    def apply_action(self, action: str) -> 'TaskState':
        """Führt einen Task aus und gibt neuen Zustand zurück"""
        task_info = self.all_tasks.get(action, {})

        # Neue completed tasks
        new_completed = self.completed_tasks | {action}

        # Ressourcen aktualisieren
        new_resources = self.available_resources.copy()
        for r, amount in task_info.get("resources", {}).items():
            new_resources[r] = new_resources.get(r, 0) - amount
        for r, amount in task_info.get("produces", {}).items():
            new_resources[r] = new_resources.get(r, 0) + amount

        return TaskState(
            completed_tasks=new_completed,
            available_resources=new_resources,
            goal_tasks=self.goal_tasks,
            all_tasks=self.all_tasks
        )

    def is_goal(self) -> bool:
        return self.goal_tasks.issubset(self.completed_tasks)

    def heuristic(self) -> float:
        """Schätzt verbleibende Kosten"""
        remaining = self.goal_tasks - self.completed_tasks
        return len(remaining)

    def __hash__(self):
        return hash(self.completed_tasks)

    def __eq__(self, other):
        return self.completed_tasks == other.completed_tasks


class MCTSNode:
    """Knoten im Monte Carlo Tree Search"""

    def __init__(self, state: PlanningState, action: str = None, parent: 'MCTSNode' = None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = state.get_available_actions()

    def ucb1(self, exploration: float = 1.414) -> float:
        """Upper Confidence Bound"""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, exploration: float = 1.414) -> 'MCTSNode':
        """Wählt bestes Kind nach UCB1"""
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def expand(self) -> 'MCTSNode':
        """Expandiert einen unversuchten Zug"""
        action = self.untried_actions.pop()
        new_state = self.state.apply_action(action)
        child = MCTSNode(new_state, action, self)
        self.children.append(child)
        return child

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return self.state.is_goal() or not self.state.get_available_actions()


class Planner:
    """
    SCIO Planner

    Unterstützt verschiedene Planungsalgorithmen:
    - A* für optimale Pfadsuche
    - MCTS für komplexe Entscheidungsräume
    - Hierarchische Planung für verschachtelte Aufgaben
    """

    def __init__(self):
        self.plans: Dict[str, Plan] = {}
        self.action_costs: Dict[str, float] = {}
        self.action_executors: Dict[str, Callable] = {}
        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert den Planner"""
        try:
            self._setup_default_actions()
            self._initialized = True
            logger.info("Planner initialisiert")
            return True
        except Exception as e:
            logger.error(f"Planner Fehler: {e}")
            return False

    def _setup_default_actions(self):
        """Definiert Standard-Aktionen und Kosten"""
        self.action_costs = {
            "load_model": 10.0,
            "unload_model": 1.0,
            "process_text": 5.0,
            "process_image": 8.0,
            "process_audio": 7.0,
            "generate_text": 6.0,
            "generate_image": 15.0,
            "embed_text": 2.0,
            "search_vectors": 1.0,
            "save_result": 0.5,
        }

    def astar_search(self,
                     initial_state: PlanningState,
                     max_iterations: int = 10000) -> Optional[List[str]]:
        """
        A* Suche für optimalen Plan

        Args:
            initial_state: Startzustand
            max_iterations: Maximale Iterationen

        Returns:
            Liste von Aktionen oder None
        """
        # Priority Queue: (f_score, counter, state, path)
        counter = 0
        open_set = [(initial_state.heuristic(), counter, initial_state, [])]
        closed_set: Set[int] = set()

        g_scores: Dict[int, float] = {hash(initial_state): 0}

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current, path = heapq.heappop(open_set)

            if current.is_goal():
                return path

            state_hash = hash(current)
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)

            for action in current.get_available_actions():
                next_state = current.apply_action(action)
                next_hash = hash(next_state)

                if next_hash in closed_set:
                    continue

                # Kosten berechnen
                action_cost = self.action_costs.get(action, 1.0)
                tentative_g = g_scores[state_hash] + action_cost

                if next_hash not in g_scores or tentative_g < g_scores[next_hash]:
                    g_scores[next_hash] = tentative_g
                    f_score = tentative_g + next_state.heuristic()
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, next_state, path + [action]))

        return None  # Kein Plan gefunden

    def mcts_search(self,
                    initial_state: PlanningState,
                    iterations: int = 1000,
                    exploration: float = 1.414) -> List[str]:
        """
        Monte Carlo Tree Search für Planung

        Args:
            initial_state: Startzustand
            iterations: Anzahl Simulationen
            exploration: Explorations-Parameter

        Returns:
            Beste gefundene Aktionssequenz
        """
        root = MCTSNode(initial_state)

        for _ in range(iterations):
            node = root

            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(exploration)

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation (Rollout)
            state = node.state
            actions = []
            while not state.is_goal():
                available = state.get_available_actions()
                if not available:
                    break
                action = random.choice(available)
                state = state.apply_action(action)
                actions.append(action)

            # Berechne Reward
            reward = 1.0 if state.is_goal() else 0.0

            # Backpropagation
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent

        # Beste Sequenz extrahieren
        best_actions = []
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            if node.action:
                best_actions.append(node.action)

        return best_actions

    def hierarchical_plan(self,
                          goal: str,
                          decomposition: Dict[str, List[str]]) -> Plan:
        """
        Hierarchische Planung durch Zieldekomposition

        Args:
            goal: Hauptziel
            decomposition: Dict von Ziel -> Unterziele/Aktionen

        Returns:
            Vollständiger Plan
        """
        plan = Plan(
            id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            goal=goal
        )

        step_counter = 0
        visited = set()

        def decompose(subgoal: str, dependencies: List[str] = None):
            nonlocal step_counter

            if subgoal in visited:
                return []
            visited.add(subgoal)

            if subgoal not in decomposition:
                # Atomare Aktion
                step = PlanStep(
                    id=f"step_{step_counter}",
                    action=subgoal,
                    dependencies=dependencies or [],
                    estimated_cost=self.action_costs.get(subgoal, 1.0)
                )
                step_counter += 1
                return [step]

            # Zerlegen in Unteraufgaben
            sub_steps = []
            sub_deps = dependencies or []

            for sub in decomposition[subgoal]:
                steps = decompose(sub, sub_deps.copy())
                sub_steps.extend(steps)
                # Neue Dependencies sind die IDs der erzeugten Schritte
                sub_deps = [s.id for s in steps]

            return sub_steps

        # Plan aufbauen
        all_steps = decompose(goal)
        for step in all_steps:
            plan.add_step(step)

        self.plans[plan.id] = plan
        return plan

    def create_plan(self,
                    goal: str,
                    available_actions: List[str],
                    constraints: Dict[str, Any] = None) -> Plan:
        """
        Erstellt einen Plan für ein Ziel

        Args:
            goal: Das zu erreichende Ziel
            available_actions: Verfügbare Aktionen
            constraints: Optionale Einschränkungen

        Returns:
            Der erstellte Plan
        """
        plan = Plan(
            id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            goal=goal
        )

        # Einfache regelbasierte Planung
        # (In der Praxis würde hier A* oder MCTS verwendet)

        step_counter = 0
        prev_step_id = None

        for action in available_actions:
            step = PlanStep(
                id=f"step_{step_counter}",
                action=action,
                dependencies=[prev_step_id] if prev_step_id else [],
                estimated_cost=self.action_costs.get(action, 1.0)
            )
            plan.add_step(step)
            prev_step_id = step.id
            step_counter += 1

        self.plans[plan.id] = plan
        return plan

    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Führt einen Plan aus

        Args:
            plan_id: ID des Plans

        Returns:
            Ausführungsergebnis
        """
        plan = self.plans.get(plan_id)
        if not plan:
            return {"error": f"Plan {plan_id} not found"}

        plan.status = PlanStatus.EXECUTING
        results = []

        while not plan.is_complete() and not plan.has_failed():
            ready_steps = plan.get_ready_steps()

            if not ready_steps:
                if not plan.is_complete():
                    plan.status = PlanStatus.FAILED
                    return {"error": "Deadlock - no steps ready", "results": results}
                break

            for step in ready_steps:
                step.status = PlanStatus.EXECUTING
                step.started_at = datetime.now()

                try:
                    # Executor aufrufen - muss registriert sein
                    if step.action in self.action_executors:
                        executor = self.action_executors[step.action]
                        step.result = executor(step.params)
                    else:
                        # Kein Executor registriert - Fehler werfen statt simulieren
                        raise NotImplementedError(
                            f"Kein Executor für Action '{step.action}' registriert. "
                            f"Registriere mit register_action_executor('{step.action}', executor_func)"
                        )

                    step.status = PlanStatus.COMPLETED
                    step.completed_at = datetime.now()
                    results.append({
                        "step_id": step.id,
                        "action": step.action,
                        "success": True,
                        "result": step.result
                    })

                except Exception as e:
                    step.status = PlanStatus.FAILED
                    step.error = str(e)
                    plan.status = PlanStatus.FAILED
                    results.append({
                        "step_id": step.id,
                        "action": step.action,
                        "success": False,
                        "error": str(e)
                    })
                    return {"error": f"Step {step.id} failed", "results": results}

        plan.status = PlanStatus.COMPLETED
        return {"success": True, "results": results, "total_cost": plan.total_cost}

    def register_executor(self, action: str, executor: Callable):
        """Registriert einen Executor für eine Aktion"""
        self.action_executors[action] = executor

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Gibt einen Plan zurück"""
        return self.plans.get(plan_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        status_counts = {}
        for plan in self.plans.values():
            status_counts[plan.status.value] = status_counts.get(plan.status.value, 0) + 1

        return {
            "total_plans": len(self.plans),
            "status_distribution": status_counts,
            "registered_actions": len(self.action_costs),
            "registered_executors": len(self.action_executors)
        }


# Singleton
_planner: Optional[Planner] = None

def get_planner() -> Planner:
    """Gibt Singleton-Instanz zurück"""
    global _planner
    if _planner is None:
        _planner = Planner()
    return _planner
