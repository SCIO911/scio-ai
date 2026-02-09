#!/usr/bin/env python3
"""
SCIO - HTN (Hierarchical Task Network) Planner

Ermoeglicht hierarchische Aufgabenplanung mit:
- Task-Dekomposition
- Praeconditions und Effekte
- Ressourcen-Management
- Plan-Optimierung
"""

from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq


class TaskStatus(str, Enum):
    """Task-Status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskType(str, Enum):
    """Task-Typen"""
    PRIMITIVE = "primitive"      # Direkt ausfuehrbar
    COMPOUND = "compound"        # Muss zerlegt werden
    GOAL = "goal"               # Zu erreichendes Ziel
    METHOD = "method"           # Zerlegungsmethode


@dataclass
class Precondition:
    """Vorbedingung fuer eine Task"""
    name: str
    check: Callable[[Dict[str, Any]], bool] = None
    description: str = ""

    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Prueft ob Vorbedingung erfuellt"""
        if self.check:
            return self.check(state)
        return state.get(self.name, False)


@dataclass
class Effect:
    """Effekt einer Task auf den Zustand"""
    name: str
    value: Any = True
    apply: Callable[[Dict[str, Any]], Dict[str, Any]] = None

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wendet Effekt auf Zustand an"""
        new_state = state.copy()
        if self.apply:
            return self.apply(new_state)
        new_state[self.name] = self.value
        return new_state


@dataclass
class Resource:
    """Ressource die fuer Tasks benoetigt wird"""
    name: str
    amount: float = 1.0
    renewable: bool = True
    current: float = 0.0

    def is_available(self, needed: float) -> bool:
        return self.current >= needed

    def consume(self, amount: float) -> bool:
        if self.current >= amount:
            self.current -= amount
            return True
        return False

    def release(self, amount: float):
        if self.renewable:
            self.current += amount


@dataclass
class Task:
    """Eine Task im HTN"""
    name: str
    task_type: TaskType = TaskType.PRIMITIVE
    preconditions: List[Precondition] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    methods: List['Method'] = field(default_factory=list)
    cost: float = 1.0
    duration: float = 1.0
    priority: int = 0
    required_resources: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None

    def __lt__(self, other):
        return self.priority > other.priority  # Hoehre Prioritaet zuerst

    def can_execute(self, state: Dict[str, Any], resources: Dict[str, Resource]) -> bool:
        """Prueft ob Task ausfuehrbar ist"""
        # Preconditions pruefen
        for precond in self.preconditions:
            if not precond.evaluate(state):
                return False

        # Ressourcen pruefen
        for res_name, amount in self.required_resources.items():
            if res_name not in resources or not resources[res_name].is_available(amount):
                return False

        return True

    def apply_effects(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wendet alle Effekte an"""
        for effect in self.effects:
            state = effect.execute(state)
        return state


@dataclass
class Method:
    """Methode zur Zerlegung einer Compound Task"""
    name: str
    preconditions: List[Precondition] = field(default_factory=list)
    subtasks: List[Task] = field(default_factory=list)
    ordering: str = "sequential"  # sequential, parallel, any

    def is_applicable(self, state: Dict[str, Any]) -> bool:
        """Prueft ob Methode anwendbar"""
        return all(p.evaluate(state) for p in self.preconditions)


@dataclass
class Plan:
    """Ein generierter Plan"""
    tasks: List[Task] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration: float = 0.0
    is_valid: bool = True
    final_state: Dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: Task):
        self.tasks.append(task)
        self.total_cost += task.cost
        self.total_duration += task.duration


class HTNPlanner:
    """
    Hierarchical Task Network Planner

    Implementiert:
    - Forward-Chaining HTN Planning
    - Task-Dekomposition
    - Ressourcen-Constraints
    - Plan-Optimierung
    """

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.methods: Dict[str, List[Method]] = defaultdict(list)
        self.resources: Dict[str, Resource] = {}
        self.initial_state: Dict[str, Any] = {}
        self.goal_state: Dict[str, Any] = {}
        self._plan_cache: Dict[str, Plan] = {}

    def register_task(self, task: Task):
        """Registriert eine Task-Definition"""
        self.tasks[task.name] = task

    def register_method(self, task_name: str, method: Method):
        """Registriert eine Zerlegungsmethode"""
        self.methods[task_name].append(method)

    def register_resource(self, resource: Resource):
        """Registriert eine Ressource"""
        self.resources[resource.name] = resource

    def set_initial_state(self, state: Dict[str, Any]):
        """Setzt den Anfangszustand"""
        self.initial_state = state.copy()

    def set_goal(self, goal: Dict[str, Any]):
        """Setzt das Ziel"""
        self.goal_state = goal.copy()

    def _decompose(self, task: Task, state: Dict[str, Any], depth: int = 0) -> Optional[List[Task]]:
        """Zerlegt eine Compound Task"""
        if depth > 100:  # Rekursionsschutz
            return None

        if task.task_type == TaskType.PRIMITIVE:
            if task.can_execute(state, self.resources):
                return [task]
            return None

        # Methoden finden
        methods = self.methods.get(task.name, []) + task.methods

        for method in methods:
            if not method.is_applicable(state):
                continue

            # Subtasks zerlegen
            primitive_tasks = []
            current_state = state.copy()
            valid = True

            for subtask in method.subtasks:
                decomposed = self._decompose(subtask, current_state, depth + 1)
                if decomposed is None:
                    valid = False
                    break

                primitive_tasks.extend(decomposed)
                # Simuliere Effekte
                for t in decomposed:
                    current_state = t.apply_effects(current_state)

            if valid:
                return primitive_tasks

        return None

    def plan(self, tasks: List[Task] = None, max_iterations: int = 1000) -> Plan:
        """
        Generiert einen Plan

        Args:
            tasks: Zu erreichende Tasks (oder goal_state verwenden)
            max_iterations: Maximale Iterationen
        """
        state = self.initial_state.copy()
        plan = Plan()

        # Wenn keine Tasks gegeben, Goal-based Planung
        if tasks is None:
            tasks = self._goals_to_tasks()

        # Tasks zerlegen und planen
        task_queue = list(tasks)
        iterations = 0

        while task_queue and iterations < max_iterations:
            iterations += 1
            task = task_queue.pop(0)

            # Primitive Task?
            if task.task_type == TaskType.PRIMITIVE:
                if task.can_execute(state, self.resources):
                    # Ressourcen verbrauchen
                    for res_name, amount in task.required_resources.items():
                        if res_name in self.resources:
                            self.resources[res_name].consume(amount)

                    # Zum Plan hinzufuegen
                    plan.add_task(task)
                    state = task.apply_effects(state)
                else:
                    plan.is_valid = False
                    break
            else:
                # Zerlegen
                decomposed = self._decompose(task, state)
                if decomposed:
                    # An den Anfang der Queue
                    task_queue = decomposed + task_queue
                else:
                    plan.is_valid = False
                    break

        plan.final_state = state
        return plan

    def _goals_to_tasks(self) -> List[Task]:
        """Konvertiert Goals zu Tasks"""
        tasks = []

        for goal_name, goal_value in self.goal_state.items():
            # Suche Task die diesen Effekt hat
            for task in self.tasks.values():
                for effect in task.effects:
                    if effect.name == goal_name and effect.value == goal_value:
                        tasks.append(task)
                        break

        return tasks

    def optimize_plan(self, plan: Plan) -> Plan:
        """Optimiert einen Plan"""
        if not plan.is_valid or len(plan.tasks) <= 1:
            return plan

        # Einfache Optimierung: Parallele Tasks identifizieren
        optimized = Plan()
        optimized.is_valid = plan.is_valid

        # Tasks nach Abhaengigkeiten sortieren
        scheduled = []
        remaining = list(plan.tasks)
        state = self.initial_state.copy()

        while remaining:
            # Finde ausfuehrbare Tasks
            executable = []
            for task in remaining:
                if task.can_execute(state, self.resources):
                    executable.append(task)

            if not executable:
                break

            # Beste Task waehlen (niedrigste Kosten)
            best = min(executable, key=lambda t: t.cost)
            remaining.remove(best)
            scheduled.append(best)
            state = best.apply_effects(state)

        for task in scheduled:
            optimized.add_task(task)

        optimized.final_state = state
        return optimized


class ResourceManager:
    """Verwaltet Ressourcen fuer den Planner"""

    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.allocations: Dict[str, Dict[str, float]] = defaultdict(dict)

    def add_resource(self, name: str, total: float, renewable: bool = True):
        """Fuegt Ressource hinzu"""
        self.resources[name] = Resource(
            name=name,
            amount=total,
            renewable=renewable,
            current=total
        )

    def allocate(self, task_id: str, resource_name: str, amount: float) -> bool:
        """Allokiert Ressourcen fuer Task"""
        if resource_name not in self.resources:
            return False

        resource = self.resources[resource_name]
        if resource.consume(amount):
            self.allocations[task_id][resource_name] = amount
            return True
        return False

    def release(self, task_id: str):
        """Gibt Ressourcen einer Task frei"""
        if task_id in self.allocations:
            for res_name, amount in self.allocations[task_id].items():
                if res_name in self.resources:
                    self.resources[res_name].release(amount)
            del self.allocations[task_id]

    def get_available(self, resource_name: str) -> float:
        """Gibt verfuegbare Menge zurueck"""
        if resource_name in self.resources:
            return self.resources[resource_name].current
        return 0.0


class PlanExecutor:
    """Fuehrt Plaene aus"""

    def __init__(self, planner: HTNPlanner):
        self.planner = planner
        self.current_task: Optional[Task] = None
        self.executed_tasks: List[Task] = []
        self.state: Dict[str, Any] = {}

    def execute(self, plan: Plan, action_handlers: Dict[str, Callable] = None) -> bool:
        """
        Fuehrt einen Plan aus

        Args:
            plan: Der auszufuehrende Plan
            action_handlers: Dict mit {task_name: handler_function}
        """
        if not plan.is_valid:
            return False

        action_handlers = action_handlers or {}
        self.state = self.planner.initial_state.copy()

        for task in plan.tasks:
            self.current_task = task
            task.status = TaskStatus.IN_PROGRESS

            try:
                # Handler aufrufen wenn vorhanden
                if task.name in action_handlers:
                    result = action_handlers[task.name](task, self.state)
                    task.result = result

                # Effekte anwenden
                self.state = task.apply_effects(self.state)
                task.status = TaskStatus.COMPLETED
                self.executed_tasks.append(task)

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.result = str(e)
                return False

        return True

    def get_progress(self) -> Dict[str, Any]:
        """Gibt Fortschritt zurueck"""
        return {
            "current_task": self.current_task.name if self.current_task else None,
            "executed": len(self.executed_tasks),
            "state": self.state
        }


# Vordefinierte Tasks fuer SCIO
def create_scio_tasks() -> Dict[str, Task]:
    """Erstellt vordefinierte Tasks fuer SCIO"""

    tasks = {}

    # Web-Recherche
    tasks["web_research"] = Task(
        name="web_research",
        task_type=TaskType.COMPOUND,
        cost=1.0,
        duration=5.0,
        methods=[
            Method(
                name="search_and_summarize",
                subtasks=[
                    Task(name="web_search", task_type=TaskType.PRIMITIVE, cost=0.5),
                    Task(name="extract_content", task_type=TaskType.PRIMITIVE, cost=0.3),
                    Task(name="summarize", task_type=TaskType.PRIMITIVE, cost=0.2)
                ]
            )
        ]
    )

    # Code-Entwicklung
    tasks["develop_code"] = Task(
        name="develop_code",
        task_type=TaskType.COMPOUND,
        cost=5.0,
        duration=30.0,
        methods=[
            Method(
                name="code_development_cycle",
                subtasks=[
                    Task(name="analyze_requirements", task_type=TaskType.PRIMITIVE),
                    Task(name="design_solution", task_type=TaskType.PRIMITIVE),
                    Task(name="implement_code", task_type=TaskType.PRIMITIVE),
                    Task(name="test_code", task_type=TaskType.PRIMITIVE),
                    Task(name="review_code", task_type=TaskType.PRIMITIVE)
                ]
            )
        ]
    )

    # Datenanalyse
    tasks["analyze_data"] = Task(
        name="analyze_data",
        task_type=TaskType.COMPOUND,
        cost=3.0,
        duration=15.0,
        methods=[
            Method(
                name="data_analysis_pipeline",
                subtasks=[
                    Task(name="load_data", task_type=TaskType.PRIMITIVE),
                    Task(name="clean_data", task_type=TaskType.PRIMITIVE),
                    Task(name="transform_data", task_type=TaskType.PRIMITIVE),
                    Task(name="compute_statistics", task_type=TaskType.PRIMITIVE),
                    Task(name="generate_report", task_type=TaskType.PRIMITIVE)
                ]
            )
        ]
    )

    # Geld verdienen
    tasks["earn_money"] = Task(
        name="earn_money",
        task_type=TaskType.COMPOUND,
        cost=0.5,
        duration=60.0,
        preconditions=[
            Precondition("gpu_available", lambda s: s.get("gpu_available", False))
        ],
        methods=[
            Method(
                name="gpu_rental",
                subtasks=[
                    Task(name="check_idle_resources", task_type=TaskType.PRIMITIVE),
                    Task(name="set_rental_price", task_type=TaskType.PRIMITIVE),
                    Task(name="activate_rental", task_type=TaskType.PRIMITIVE),
                    Task(name="monitor_earnings", task_type=TaskType.PRIMITIVE)
                ]
            )
        ]
    )

    return tasks


# Singleton
_planner: Optional[HTNPlanner] = None


def get_htn_planner() -> HTNPlanner:
    """Gibt HTN Planner Singleton zurueck"""
    global _planner
    if _planner is None:
        _planner = HTNPlanner()
        # SCIO Tasks registrieren
        for task in create_scio_tasks().values():
            _planner.register_task(task)
    return _planner
