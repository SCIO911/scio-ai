"""
SCIO Tool Base

Basis-Klassen für das Tool-System.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from scio.core.logging import get_logger

logger = get_logger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ToolConfig(BaseModel):
    """Basis-Konfiguration für Tools."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Tool-Name")
    description: Optional[str] = Field(default=None)
    timeout_seconds: int = Field(default=60, ge=1)
    requires_sandbox: bool = Field(default=True)


@dataclass
class ToolResult:
    """Ergebnis einer Tool-Ausführung."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


class Tool(ABC, Generic[InputT, OutputT]):
    """
    Abstrakte Basis-Klasse für SCIO-Tools.

    Tools sind atomare Operationen die von Agenten verwendet werden können.
    """

    # Klassen-Attribute
    tool_name: str = "base"
    version: str = "1.0"

    def __init__(self, config: Optional[ToolConfig | dict[str, Any]] = None):
        if config is None:
            config = ToolConfig(name=self.tool_name)
        elif isinstance(config, dict):
            config = ToolConfig(**config)

        self.config = config
        self.logger = get_logger(__name__, tool=self.tool_name)

    @abstractmethod
    async def execute(self, input_data: InputT) -> OutputT:
        """
        Führt das Tool aus.

        Args:
            input_data: Eingabedaten

        Returns:
            Tool-Output
        """
        pass

    async def run(self, input_data: InputT) -> ToolResult:
        """
        Führt das Tool aus mit Fehlerbehandlung.

        Args:
            input_data: Eingabedaten

        Returns:
            ToolResult
        """
        self.logger.debug("Tool executing", input=str(input_data)[:100])

        try:
            output = await self.execute(input_data)
            return ToolResult(success=True, output=output)

        except Exception as e:
            self.logger.error("Tool execution failed", error=str(e))
            return ToolResult(success=False, error=str(e))

    def get_schema(self) -> dict[str, Any]:
        """
        Gibt das Input-Schema zurück (für LLM-Integration).

        Überschreibe für detailliertes Schema.
        """
        return {
            "name": self.tool_name,
            "description": self.config.description or f"Tool: {self.tool_name}",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
