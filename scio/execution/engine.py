"""
SCIO Execution Engine

Zentrale Engine für die Ausführung von Experimenten.
"""

import asyncio
import re
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
    SKIPPED = "skipped"


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


@dataclass
class ExecutionContext:
    """Kontext für die Ausführung."""

    execution_id: str
    experiment_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    step_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)

    def resolve_value(self, value: Any) -> Any:
        """Löst Referenzen in einem Wert auf."""
        if isinstance(value, str):
            return self._resolve_string(value)
        elif isinstance(value, dict):
            return {k: self.resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve_value(v) for v in value]
        return value

    def _resolve_string(self, s: str) -> Any:
        """Löst String-Referenzen auf."""
        # Pattern: ${...}
        pattern = r'\$\{([^}]+)\}'

        # Wenn der gesamte String eine Referenz ist, gib den Wert direkt zurück
        match = re.fullmatch(pattern, s)
        if match:
            return self._get_reference_value(match.group(1))

        # Sonst ersetze alle Referenzen im String
        def replace(m):
            val = self._get_reference_value(m.group(1))
            return str(val) if val is not None else ""

        return re.sub(pattern, replace, s)

    def _get_reference_value(self, path: str) -> Any:
        """Holt den Wert einer Referenz."""
        parts = path.split(".")

        if len(parts) == 1:
            # Parameter: ${param_name}
            return self.parameters.get(parts[0])

        if parts[0] == "var" and len(parts) == 2:
            # Variable: ${var.name}
            return self.variables.get(parts[1])

        if parts[0] == "step" and len(parts) >= 2:
            # Step Output: ${step.step_id.output_name}
            step_id = parts[1]
            if step_id in self.step_outputs:
                if len(parts) == 2:
                    return self.step_outputs[step_id]
                output_key = ".".join(parts[2:])
                return self.step_outputs[step_id].get(output_key)

        if parts[0] == "env" and len(parts) == 2:
            # Environment: ${env.VAR_NAME}
            import os
            return os.environ.get(parts[1])

        return None


class ExecutionEngine:
    """
    Engine für die Ausführung von SCIO-Experimenten.

    Features:
    - Asynchrone Ausführung
    - Dependency-Tracking zwischen Steps
    - Agent und Tool Integration
    - Checkpointing und Recovery
    - Ressourcen-Management
    """

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__, component="execution_engine")
        self._running: dict[str, ExecutionResult] = {}
        self._step_handlers: dict[str, Callable] = {}
        self._agents_cache: dict[str, Any] = {}
        self._tools_cache: dict[str, Any] = {}

        # Registriere Standard-Handler
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Registriert die Standard-Step-Handler."""
        self.register_step_handler("agent", self._handle_agent_step)
        self.register_step_handler("tool", self._handle_tool_step)
        self.register_step_handler("condition", self._handle_condition_step)
        self.register_step_handler("checkpoint", self._handle_checkpoint_step)

    def register_step_handler(
        self, step_type: str, handler: Callable[[StepSchema, ExecutionContext], Any]
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
        on_step_complete: Optional[Callable[[StepResult], None]] = None,
    ) -> ExecutionResult:
        """
        Führt ein Experiment aus.

        Args:
            experiment: Das auszuführende Experiment
            parameters: Parameter-Überschreibungen
            on_step_complete: Callback nach jedem Step

        Returns:
            ExecutionResult mit allen Ergebnissen
        """
        execution_id = generate_id("exec")
        parameters = parameters or {}

        # Merge mit Default-Parametern aus Experiment
        merged_params = {}
        for param in experiment.parameters:
            merged_params[param.name] = param.default
        merged_params.update(parameters)

        self.logger.info(
            "Starting experiment execution",
            execution_id=execution_id,
            experiment=experiment.name,
            parameters=list(merged_params.keys()),
        )

        result = ExecutionResult(
            execution_id=execution_id,
            experiment_name=experiment.name,
            status=ExecutionStatus.RUNNING,
            started_at=now_utc(),
            metadata={
                "parameters": merged_params,
                "step_count": len(experiment.steps),
            },
        )

        self._running[execution_id] = result

        # Erstelle Execution Context
        context = ExecutionContext(
            execution_id=execution_id,
            experiment_name=experiment.name,
            parameters=merged_params,
        )

        # Erstelle Agent-Instanzen aus Experiment
        await self._create_agents(experiment, context)

        try:
            # Berechne Ausführungsreihenfolge
            execution_order = experiment.get_execution_order()
            self.logger.info("Execution order", order=execution_order)

            # Führe Steps in Reihenfolge aus
            for step_id in execution_order:
                step = next(s for s in experiment.steps if s.id == step_id)
                step_result = await self._execute_step(step, context, experiment)
                result.step_results.append(step_result)

                # Callback
                if on_step_complete:
                    on_step_complete(step_result)

                if step_result.status == ExecutionStatus.FAILED:
                    result.status = ExecutionStatus.FAILED
                    result.metadata["failed_step"] = step_id
                    break

                # Speichere Outputs für nachfolgende Steps
                context.step_outputs[step_id] = step_result.outputs

            if result.status == ExecutionStatus.RUNNING:
                result.status = ExecutionStatus.COMPLETED

            result.completed_at = now_utc()
            result.outputs = dict(context.step_outputs)

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
            if execution_id in self._running:
                del self._running[execution_id]
            # Cleanup caches
            self._agents_cache.clear()
            self._tools_cache.clear()

        return result

    async def _create_agents(
        self, experiment: ExperimentSchema, context: ExecutionContext
    ) -> None:
        """Erstellt Agent-Instanzen aus dem Experiment."""
        from scio.agents.registry import AgentRegistry
        from scio.agents.base import AgentContext

        # Import builtins
        import scio.agents.builtin  # noqa

        for agent_def in experiment.agents:
            try:
                # Erstelle Agent-Config
                config = {
                    "name": agent_def.name,
                    "description": agent_def.description,
                }
                if agent_def.config:
                    config.update(agent_def.config)

                # Erstelle Agent-Instanz
                agent = AgentRegistry.create(agent_def.type, config)

                # Erstelle Agent-Context
                agent_ctx = AgentContext(
                    agent_id=agent.agent_id,
                    execution_id=context.execution_id,
                    experiment_name=context.experiment_name,
                    parameters=context.parameters,
                )

                self._agents_cache[agent_def.id] = {
                    "agent": agent,
                    "context": agent_ctx,
                    "definition": agent_def,
                }

                self.logger.debug(
                    "Agent created",
                    agent_id=agent_def.id,
                    type=agent_def.type,
                )

            except Exception as e:
                self.logger.warning(
                    "Failed to create agent",
                    agent_id=agent_def.id,
                    error=str(e),
                )

    async def _execute_step(
        self,
        step: StepSchema,
        context: ExecutionContext,
        experiment: ExperimentSchema,
    ) -> StepResult:
        """Führt einen einzelnen Step aus."""

        self.logger.info(
            "Executing step",
            step_id=step.id,
            type=step.type.value,
        )

        result = StepResult(
            step_id=step.id,
            status=ExecutionStatus.RUNNING,
            started_at=now_utc(),
        )

        try:
            # Prüfe Bedingung falls vorhanden
            if step.condition:
                # Flatten context für Condition-Evaluation
                flat_context = {
                    **context.parameters,
                    **context.variables,
                    "outputs": context.step_outputs,
                }
                if not self._evaluate_condition(step.condition, flat_context):
                    self.logger.info("Step skipped due to condition", step_id=step.id)
                    result.status = ExecutionStatus.SKIPPED
                    result.outputs["skipped"] = True
                    result.outputs["reason"] = f"Condition not met: {step.condition}"
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
                    handler(step, context, experiment),
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

    async def _handle_agent_step(
        self,
        step: StepSchema,
        context: ExecutionContext,
        experiment: ExperimentSchema,
    ) -> dict[str, Any]:
        """Handler für Agent-Steps."""
        agent_id = step.agent
        if not agent_id:
            raise ExecutionError("Agent-Step ohne Agent-ID", step=step.id)

        # Hole Agent aus Cache
        agent_data = self._agents_cache.get(agent_id)
        if not agent_data:
            raise ExecutionError(f"Agent nicht gefunden: {agent_id}", step=step.id)

        agent = agent_data["agent"]
        agent_ctx = agent_data["context"]

        # Resolve Inputs
        inputs = context.resolve_value(step.inputs)

        self.logger.debug(
            "Calling agent",
            agent_id=agent_id,
            inputs=list(inputs.keys()) if inputs else [],
        )

        # Führe Agent aus
        output = await agent.execute(inputs, agent_ctx)

        # Konvertiere Output
        if isinstance(output, dict):
            return output
        if hasattr(output, "to_dict"):
            return output.to_dict()
        if hasattr(output, "model_dump"):
            return output.model_dump()
        return {"result": output}

    async def _handle_tool_step(
        self,
        step: StepSchema,
        context: ExecutionContext,
        experiment: ExperimentSchema,
    ) -> dict[str, Any]:
        """Handler für Tool-Steps."""
        from scio.tools.registry import ToolRegistry

        # Import builtins
        import scio.tools.builtin  # noqa

        tool_name = step.tool
        if not tool_name:
            # Versuche aus inputs zu ermitteln
            tool_name = step.inputs.get("tool") if step.inputs else None

        if not tool_name:
            raise ExecutionError("Tool-Step ohne Tool-Name", step=step.id)

        # Hole oder erstelle Tool
        if tool_name not in self._tools_cache:
            tool = ToolRegistry.create(tool_name)
            self._tools_cache[tool_name] = tool
        else:
            tool = self._tools_cache[tool_name]

        # Resolve Inputs
        inputs = context.resolve_value(step.inputs) or {}

        self.logger.debug(
            "Calling tool",
            tool_name=tool_name,
            inputs=list(inputs.keys()),
        )

        # Führe Tool aus
        result = await tool.run(inputs)

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

    async def _handle_condition_step(
        self,
        step: StepSchema,
        context: ExecutionContext,
        experiment: ExperimentSchema,
    ) -> dict[str, Any]:
        """Handler für Condition-Steps."""
        condition = step.condition or step.inputs.get("condition", "True")

        flat_context = {
            **context.parameters,
            **context.variables,
            "outputs": context.step_outputs,
        }

        result = self._evaluate_condition(condition, flat_context)

        # Setze Variable für nachfolgende Steps
        if step.outputs:
            for output_name in step.outputs:
                context.variables[output_name] = result

        return {
            "condition": condition,
            "result": result,
        }

    async def _handle_checkpoint_step(
        self,
        step: StepSchema,
        context: ExecutionContext,
        experiment: ExperimentSchema,
    ) -> dict[str, Any]:
        """Handler für Checkpoint-Steps."""
        from scio.memory.context import ContextManager
        from scio.memory.store import MemoryStore

        # Erstelle Checkpoint
        store = MemoryStore()
        checkpoint_id = generate_id("ckpt")

        checkpoint_data = {
            "execution_id": context.execution_id,
            "experiment_name": context.experiment_name,
            "step_id": step.id,
            "parameters": context.parameters,
            "step_outputs": context.step_outputs,
            "variables": context.variables,
            "timestamp": now_utc().isoformat(),
        }

        store.set(
            key=f"checkpoint:{context.execution_id}:{checkpoint_id}",
            value=checkpoint_data,
            tags=["checkpoint", context.execution_id],
        )

        self.logger.info(
            "Checkpoint created",
            checkpoint_id=checkpoint_id,
            step_id=step.id,
        )

        return {
            "checkpoint_id": checkpoint_id,
            "step_id": step.id,
        }

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
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
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
                elif isinstance(node.slice, ast.Name):
                    key = safe_eval(node.slice)
                    return value[key]
                raise ValueError("Nur konstante Subscripts erlaubt")
            elif isinstance(node, ast.Attribute):
                value = safe_eval(node.value)
                return getattr(value, node.attr)
            elif isinstance(node, ast.List):
                return [safe_eval(e) for e in node.elts]
            elif isinstance(node, ast.Tuple):
                return tuple(safe_eval(e) for e in node.elts)
            elif isinstance(node, ast.Dict):
                return {safe_eval(k): safe_eval(v) for k, v in zip(node.keys, node.values)}
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

    def list_running(self) -> list[str]:
        """Gibt alle laufenden Ausführungen zurück."""
        return list(self._running.keys())
