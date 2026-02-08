"""
SCIO Agent Base

Basis-Klassen für das Agenten-System.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AgentState(str, Enum):
    """Zustand eines Agenten."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentConfig(BaseModel):
    """Basis-Konfiguration für Agenten."""

    name: str = Field(..., description="Name des Agenten")
    description: Optional[str] = Field(default=None)
    max_iterations: int = Field(default=100, ge=1)
    timeout_seconds: int = Field(default=300, ge=1)
    retry_on_failure: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0)

    class Config:
        extra = "allow"


@dataclass
class AgentContext:
    """Ausführungskontext für einen Agenten."""

    agent_id: str
    execution_id: str
    experiment_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    started_at: datetime = field(default_factory=now_utc)


@dataclass
class AgentResult:
    """Ergebnis einer Agenten-Ausführung."""

    agent_id: str
    state: AgentState
    outputs: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "outputs": self.outputs,
            "iterations": self.iterations,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class Agent(ABC, Generic[InputT, OutputT]):
    """
    Abstrakte Basis-Klasse für SCIO-Agenten.

    Agenten sind autonome Einheiten, die spezifische Aufgaben ausführen können.
    Sie können iterativ arbeiten und Entscheidungen treffen.

    Implementiere `execute` für die Hauptlogik.
    """

    # Klassen-Attribute
    agent_type: str = "base"
    version: str = "1.0"

    def __init__(self, config: AgentConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = AgentConfig(**config)

        self.config = config
        self.agent_id = generate_id("agent")
        self.state = AgentState.IDLE
        self.logger = get_logger(
            __name__,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
        )

    @abstractmethod
    async def execute(
        self, input_data: InputT, context: AgentContext
    ) -> OutputT:
        """
        Führt die Hauptlogik des Agenten aus.

        Args:
            input_data: Eingabedaten
            context: Ausführungskontext

        Returns:
            Ausgabedaten
        """
        pass

    async def run(
        self,
        input_data: InputT,
        parameters: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Führt den Agenten aus mit vollständigem Lifecycle.

        Args:
            input_data: Eingabedaten
            parameters: Optionale Parameter

        Returns:
            AgentResult mit Ergebnissen
        """
        context = AgentContext(
            agent_id=self.agent_id,
            execution_id=generate_id("exec"),
            parameters=parameters or {},
        )

        result = AgentResult(
            agent_id=self.agent_id,
            state=AgentState.RUNNING,
            started_at=now_utc(),
        )

        self.state = AgentState.RUNNING
        self.logger.info("Agent starting", execution_id=context.execution_id)

        try:
            # Vor-Ausführung Hook
            await self.on_start(context)

            # Hauptausführung mit Iteration-Limit
            while context.iteration < self.config.max_iterations:
                context.iteration += 1

                try:
                    output = await self.execute(input_data, context)

                    # Prüfe ob Agent fertig ist
                    if await self.is_complete(output, context):
                        result.outputs = self._serialize_output(output)
                        result.state = AgentState.COMPLETED
                        break

                except Exception as e:
                    if self.config.retry_on_failure and context.iteration < self.config.max_retries:
                        self.logger.warning(
                            "Agent iteration failed, retrying",
                            iteration=context.iteration,
                            error=str(e),
                        )
                        continue
                    raise

            else:
                # Max iterations erreicht
                self.logger.warning(
                    "Agent reached max iterations",
                    max_iterations=self.config.max_iterations,
                )
                result.state = AgentState.COMPLETED

            result.iterations = context.iteration

        except Exception as e:
            self.logger.error("Agent execution failed", error=str(e))
            result.state = AgentState.FAILED
            result.error = str(e)

        finally:
            result.completed_at = now_utc()
            self.state = result.state

            # Nach-Ausführung Hook
            await self.on_complete(result, context)

        return result

    async def on_start(self, context: AgentContext) -> None:
        """Hook vor der Ausführung."""
        pass

    async def on_complete(self, result: AgentResult, context: AgentContext) -> None:
        """Hook nach der Ausführung."""
        pass

    async def is_complete(self, output: OutputT, context: AgentContext) -> bool:
        """
        Prüft ob der Agent fertig ist.

        Überschreibe für iterative Agenten.
        """
        return True

    def _serialize_output(self, output: OutputT) -> dict[str, Any]:
        """Serialisiert Output zu Dictionary."""
        if isinstance(output, dict):
            return output
        if hasattr(output, "to_dict"):
            return output.to_dict()
        if hasattr(output, "model_dump"):
            return output.model_dump()
        return {"result": output}
