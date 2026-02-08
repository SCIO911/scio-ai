"""
SCIO Execution Engine

Zentrale Engine für die Ausführung von Experimenten.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from scio.core.config import get_config
from scio.core.exceptions import ExecutionError
from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc
from scio.parser.schema import ExperimentSchema, StepSchema

logger = get_logger(__name__)


class ExecutionStatus(str, Enum):
    """Status einer Ausführung."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class StepResult:
    """Ergebnis eines einzelnen Schritts."""

    step_id: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    outputs: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "outputs": self.outputs,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ExecutionResult:
    """Gesamtergebnis einer Experiment-Ausführung."""

    execution_id: str
    experiment_name: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    step_results: list[StepResult] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "step_results": [s.to_dict() for s in self.step_results],
            "outputs": self.outputs,
            "metadata": self.metadata,
        }


class ExecutionEngine:
    """
    Engine für die Ausführung von SCIO-Experimenten.

    Features:
    - Asynchrone Ausführung
    - Dependency-Tracking zwischen Steps
    - Checkpointing und Recovery
    - Ressourcen-Management
    """

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__, component="execution_engine")
        self._running: dict[str, ExecutionResult] = {}
        self._step_handlers: dict[str, Callable] = {}

    def register_step_handler(
        self, step_type: str, handler: Callable[[StepSchema, dict], Any]
    ) -> None:
        """
        Registriert einen Handler für einen Step-Typ.

        Args:
            step_type: Typ des Steps (z.B. 'agent', 'tool')
            handler: Async Callable für die Ausführung
        """
        self._step_handlers[step_type] = handler
        self.logger.debug("Step handler registered", step_type=step_type)

    async def execute(
        self,
        experiment: ExperimentSchema,
        parameters: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Führt ein Experiment aus.

        Args:
            experiment: Das auszuführende Experiment
            parameters: Parameter-Überschreibungen

        Returns:
            ExecutionResult mit allen Ergebnissen
        """
        execution_id = generate_id("exec")
        parameters = parameters or {}

        self.logger.info(
            "Starting experiment execution",
            execution_id=execution_id,
            experiment=experiment.name,
        )

        result = ExecutionResult(
            execution_id=execution_id,
            experiment_name=experiment.name,
            status=ExecutionStatus.RUNNING,
            started_at=now_utc(),
            metadata={
                "parameters": parameters,
                "step_count": len(experiment.steps),
            },
        )

        self._running[execution_id] = result

        try:
            # Berechne Ausführungsreihenfolge
            execution_order = experiment.get_execution_order()
            context = {"parameters": parameters, "outputs": {}}

            # Führe Steps in Reihenfolge aus
            for step_id in execution_order:
                step = next(s for s in experiment.steps if s.id == step_id)
                step_result = await self._execute_step(step, context)
                result.step_results.append(step_result)

                if step_result.status == ExecutionStatus.FAILED:
                    result.status = ExecutionStatus.FAILED
                    break

                # Speichere Outputs für nachfolgende Steps
                context["outputs"][step_id] = step_result.outputs

            if result.status == ExecutionStatus.RUNNING:
                result.status = ExecutionStatus.COMPLETED

            result.completed_at = now_utc()
            result.outputs = context["outputs"]

            self.logger.info(
                "Experiment execution completed",
                execution_id=execution_id,
                status=result.status.value,
                duration_ms=result.duration_ms,
            )

        except Exception as e:
            self.logger.error(
                "Experiment execution failed",
                execution_id=execution_id,
                error=str(e),
            )
            result.status = ExecutionStatus.FAILED
            result.completed_at = now_utc()
            result.metadata["error"] = str(e)

        finally:
            del self._running[execution_id]

        return result

    async def _execute_step(
        self, step: StepSchema, context: dict[str, Any]
    ) -> StepResult:
        """Führt einen einzelnen Step aus."""

        self.logger.debug("Executing step", step_id=step.id, type=step.type.value)

        result = StepResult(
            step_id=step.id,
            status=ExecutionStatus.RUNNING,
            started_at=now_utc(),
        )

        try:
            # Prüfe Bedingung falls vorhanden
            if step.condition:
                if not self._evaluate_condition(step.condition, context):
                    self.logger.info("Step skipped due to condition", step_id=step.id)
                    result.status = ExecutionStatus.COMPLETED
                    result.outputs["skipped"] = True
                    result.completed_at = now_utc()
                    return result

            # Finde und führe Handler aus
            handler = self._step_handlers.get(step.type.value)

            if handler is None:
                raise ExecutionError(
                    f"Kein Handler für Step-Typ: {step.type.value}",
                    step=step.id,
                )

            # Timeout handling
            timeout = step.resources.timeout_seconds

            try:
                outputs = await asyncio.wait_for(
                    handler(step, context),
                    timeout=timeout,
                )
                result.outputs = outputs or {}
                result.status = ExecutionStatus.COMPLETED

            except asyncio.TimeoutError:
                raise ExecutionError(
                    f"Step Timeout nach {timeout}s",
                    step=step.id,
                )

        except Exception as e:
            self.logger.error("Step execution failed", step_id=step.id, error=str(e))
            result.status = ExecutionStatus.FAILED
            result.error = str(e)

        result.completed_at = now_utc()
        if result.started_at:
            delta = result.completed_at - result.started_at
            result.duration_ms = int(delta.total_seconds() * 1000)

        return result

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """
        Evaluiert eine Step-Bedingung sicher.

        Unterstützte Operatoren: ==, !=, <, >, <=, >=, and, or, not
        Unterstützte Werte: Zahlen, Strings, Booleans, Variablen aus context
        """
        import ast
        import operator

        # Erlaubte Operatoren
        SAFE_OPERATORS = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.And: lambda a, b: a and b,
            ast.Or: lambda a, b: a or b,
            ast.Not: operator.not_,
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def safe_eval(node):
            """Rekursive sichere Evaluation."""
            if isinstance(node, ast.Expression):
                return safe_eval(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                if node.id in context:
                    return context[node.id]
                elif node.id == 'True':
                    return True
                elif node.id == 'False':
                    return False
                elif node.id == 'None':
                    return None
                else:
                    raise ValueError(f"Unbekannte Variable: {node.id}")
            elif isinstance(node, ast.Compare):
                left = safe_eval(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    op_func = SAFE_OPERATORS.get(type(op))
                    if op_func is None:
                        raise ValueError(f"Nicht erlaubter Operator: {type(op).__name__}")
                    right = safe_eval(comparator)
                    if not op_func(left, right):
                        return False
                    left = right
                return True
            elif isinstance(node, ast.BoolOp):
                op_func = SAFE_OPERATORS.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Nicht erlaubter Operator: {type(node.op).__name__}")
                values = [safe_eval(v) for v in node.values]
                result = values[0]
                for v in values[1:]:
                    result = op_func(result, v)
                return result
            elif isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.Not):
                    return not safe_eval(node.operand)
                raise ValueError(f"Nicht erlaubter Operator: {type(node.op).__name__}")
            elif isinstance(node, ast.BinOp):
                op_func = SAFE_OPERATORS.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Nicht erlaubter Operator: {type(node.op).__name__}")
                return op_func(safe_eval(node.left), safe_eval(node.right))
            elif isinstance(node, ast.Subscript):
                value = safe_eval(node.value)
                if isinstance(node.slice, ast.Constant):
                    return value[node.slice.value]
                raise ValueError("Nur konstante Subscripts erlaubt")
            else:
                raise ValueError(f"Nicht erlaubter Node-Typ: {type(node).__name__}")

        try:
            tree = ast.parse(condition, mode='eval')
            result = safe_eval(tree)
            self.logger.debug("Condition evaluated", condition=condition, result=result)
            return bool(result)
        except Exception as e:
            self.logger.error("Condition evaluation failed", condition=condition, error=str(e))
            return False

    async def cancel(self, execution_id: str) -> bool:
        """Bricht eine laufende Ausführung ab."""
        if execution_id in self._running:
            self._running[execution_id].status = ExecutionStatus.CANCELLED
            self.logger.info("Execution cancelled", execution_id=execution_id)
            return True
        return False

    def get_status(self, execution_id: str) -> Optional[ExecutionResult]:
        """Gibt den Status einer Ausführung zurück."""
        return self._running.get(execution_id)
