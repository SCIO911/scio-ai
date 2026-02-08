"""
SCIO Context Management

Verwaltung von Ausführungskontext und Variablen.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from scio.core.logging import get_logger
from scio.core.utils import now_utc, generate_id
from scio.memory.store import MemoryStore

logger = get_logger(__name__)


@dataclass
class ExecutionContext:
    """
    Ausführungskontext für ein Experiment.

    Enthält alle Informationen die während der Ausführung
    benötigt werden.
    """

    execution_id: str
    experiment_name: str
    started_at: datetime = field(default_factory=now_utc)

    # Parameter und Variablen
    parameters: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)

    # Step-Outputs
    step_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Liest eine Variable."""
        return self.variables.get(name, self.parameters.get(name, default))

    def set_variable(self, name: str, value: Any) -> None:
        """Setzt eine Variable."""
        self.variables[name] = value

    def get_step_output(self, step_id: str, key: str, default: Any = None) -> Any:
        """Liest einen Step-Output."""
        step_data = self.step_outputs.get(step_id, {})
        return step_data.get(key, default)

    def set_step_output(self, step_id: str, outputs: dict[str, Any]) -> None:
        """Setzt Step-Outputs."""
        self.step_outputs[step_id] = outputs

    def resolve_reference(self, ref: str) -> Any:
        """
        Löst eine Referenz auf.

        Unterstützte Formate:
        - ${param_name} - Parameter
        - ${var.var_name} - Variable
        - ${step.step_id.output_name} - Step Output

        Args:
            ref: Referenz-String

        Returns:
            Aufgelöster Wert
        """
        if not ref.startswith("${") or not ref.endswith("}"):
            return ref

        path = ref[2:-1]  # Entferne ${ und }

        parts = path.split(".")

        if len(parts) == 1:
            # Parameter
            return self.parameters.get(parts[0])

        if parts[0] == "var" and len(parts) == 2:
            # Variable
            return self.variables.get(parts[1])

        if parts[0] == "step" and len(parts) >= 3:
            # Step Output
            step_id = parts[1]
            output_key = ".".join(parts[2:])
            return self.get_step_output(step_id, output_key)

        logger.warning("Unknown reference format", ref=ref)
        return None

    def to_dict(self) -> dict[str, Any]:
        import copy
        return {
            "execution_id": self.execution_id,
            "experiment_name": self.experiment_name,
            "started_at": self.started_at.isoformat(),
            "parameters": copy.deepcopy(self.parameters),
            "variables": copy.deepcopy(self.variables),
            "step_outputs": copy.deepcopy(self.step_outputs),
            "metadata": copy.deepcopy(self.metadata),
        }


class ContextManager:
    """
    Verwaltet Ausführungskontexte.

    Features:
    - Kontext-Erstellung und -Verwaltung
    - Persistenz über MemoryStore
    - Checkpoint/Restore
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory = memory_store or MemoryStore()
        self.logger = get_logger(__name__, component="context_manager")
        self._active_contexts: dict[str, ExecutionContext] = {}

    def create_context(
        self,
        experiment_name: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> ExecutionContext:
        """
        Erstellt einen neuen Ausführungskontext.

        Args:
            experiment_name: Name des Experiments
            parameters: Initiale Parameter

        Returns:
            Neuer ExecutionContext
        """
        ctx = ExecutionContext(
            execution_id=generate_id("ctx"),
            experiment_name=experiment_name,
            parameters=parameters or {},
        )

        self._active_contexts[ctx.execution_id] = ctx
        self.logger.info(
            "Context created",
            execution_id=ctx.execution_id,
            experiment=experiment_name,
        )

        return ctx

    def get_context(self, execution_id: str) -> Optional[ExecutionContext]:
        """Gibt einen aktiven Kontext zurück."""
        return self._active_contexts.get(execution_id)

    def save_checkpoint(self, ctx: ExecutionContext) -> str:
        """
        Speichert einen Checkpoint.

        Args:
            ctx: Zu speichernder Kontext

        Returns:
            Checkpoint-ID
        """
        checkpoint_id = generate_id("ckpt")
        key = f"checkpoint:{ctx.execution_id}:{checkpoint_id}"

        self.memory.set(
            key=key,
            value=ctx.to_dict(),
            tags=["checkpoint", ctx.execution_id],
            metadata={"checkpoint_id": checkpoint_id},
        )

        self.logger.info(
            "Checkpoint saved",
            execution_id=ctx.execution_id,
            checkpoint_id=checkpoint_id,
        )

        return checkpoint_id

    def restore_checkpoint(
        self, execution_id: str, checkpoint_id: str
    ) -> Optional[ExecutionContext]:
        """
        Stellt einen Checkpoint wieder her.

        Args:
            execution_id: Ausführungs-ID
            checkpoint_id: Checkpoint-ID

        Returns:
            Wiederhergestellter Kontext oder None
        """
        key = f"checkpoint:{execution_id}:{checkpoint_id}"
        data = self.memory.get(key)

        if data is None:
            self.logger.warning(
                "Checkpoint not found",
                execution_id=execution_id,
                checkpoint_id=checkpoint_id,
            )
            return None

        ctx = ExecutionContext(
            execution_id=data["execution_id"],
            experiment_name=data["experiment_name"],
            started_at=datetime.fromisoformat(data["started_at"]),
            parameters=data["parameters"],
            variables=data["variables"],
            step_outputs=data["step_outputs"],
            metadata=data["metadata"],
        )

        self._active_contexts[ctx.execution_id] = ctx
        self.logger.info("Checkpoint restored", checkpoint_id=checkpoint_id)

        return ctx

    def close_context(self, execution_id: str) -> None:
        """Schließt einen Kontext."""
        if execution_id in self._active_contexts:
            del self._active_contexts[execution_id]
            self.logger.debug("Context closed", execution_id=execution_id)
