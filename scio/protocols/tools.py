"""
Tool Protocol
=============

Standardisierte Schnittstellen fuer Tools und deren Ein-/Ausgaben.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type, Union
from enum import Enum, auto
from datetime import datetime
from abc import ABC, abstractmethod
import uuid
import json


class ToolStatus(Enum):
    """Tool-Ausfuehrungsstatus"""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class ToolCategory(Enum):
    """Tool-Kategorien"""
    DATA = "data"
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"
    IO = "io"
    NETWORK = "network"
    SYSTEM = "system"
    ML = "ml"
    VISUALIZATION = "visualization"
    UTILITY = "utility"


@dataclass
class ToolMetadata:
    """Metadaten eines Tools"""

    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    category: ToolCategory = ToolCategory.UTILITY
    tags: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)  # Dependencies
    deprecated: bool = False
    deprecated_message: str = ""

    # Laufzeit-Eigenschaften
    timeout_seconds: Optional[float] = None
    max_retries: int = 0
    cacheable: bool = False
    cache_ttl: int = 3600  # Sekunden

    # Ressourcen-Anforderungen
    requires_gpu: bool = False
    min_memory_mb: int = 0
    max_concurrent: int = 0  # 0 = unbegrenzt

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "category": self.category.value,
            "tags": self.tags,
            "requires": self.requires,
            "deprecated": self.deprecated,
            "timeout_seconds": self.timeout_seconds,
            "cacheable": self.cacheable,
            "requires_gpu": self.requires_gpu,
        }


@dataclass
class ToolInput:
    """Eingabe fuer ein Tool"""

    parameters: Dict[str, Any] = field(default_factory=dict)
    data: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Holt Parameter oder Default"""
        return self.parameters.get(key, default)

    def require(self, key: str) -> Any:
        """Holt erforderlichen Parameter"""
        if key not in self.parameters:
            raise ValueError(f"Erforderlicher Parameter fehlt: {key}")
        return self.parameters[key]

    def validate(self, schema: Dict[str, type]) -> tuple[bool, List[str]]:
        """Validiert Parameter gegen Schema"""
        errors = []

        for param_name, param_type in schema.items():
            if param_name not in self.parameters:
                errors.append(f"Parameter fehlt: {param_name}")
            elif not isinstance(self.parameters[param_name], param_type):
                errors.append(f"Falscher Typ fuer {param_name}: erwartet {param_type.__name__}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "parameters": self.parameters,
            "data": self.data,
            "context": self.context,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolInput':
        """Erstellt aus Dictionary"""
        return cls(
            parameters=data.get("parameters", {}),
            data=data.get("data"),
            context=data.get("context", {}),
            options=data.get("options", {}),
        )


@dataclass
class ToolOutput:
    """Ausgabe eines Tools"""

    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # Zusaetzliche Ausgaben
    logs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_artifact(self, name: str, value: Any) -> None:
        """Fuegt Artefakt hinzu"""
        self.artifacts[name] = value

    def add_log(self, message: str) -> None:
        """Fuegt Log-Eintrag hinzu"""
        self.logs.append(f"[{datetime.now().isoformat()}] {message}")

    def add_warning(self, message: str) -> None:
        """Fuegt Warnung hinzu"""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "result": self.result,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "logs": self.logs,
            "warnings": self.warnings,
        }


@dataclass
class ToolResult:
    """Vollstaendiges Ergebnis einer Tool-Ausfuehrung"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    status: ToolStatus = ToolStatus.PENDING
    input: Optional[ToolInput] = None
    output: Optional[ToolOutput] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    retries: int = 0

    @property
    def success(self) -> bool:
        """Ob Ausfuehrung erfolgreich war"""
        return self.status == ToolStatus.SUCCESS

    @property
    def failed(self) -> bool:
        """Ob Ausfuehrung fehlgeschlagen ist"""
        return self.status in (ToolStatus.FAILED, ToolStatus.TIMEOUT)

    def mark_started(self) -> None:
        """Markiert als gestartet"""
        self.status = ToolStatus.RUNNING
        self.start_time = datetime.now()

    def mark_success(self, output: ToolOutput) -> None:
        """Markiert als erfolgreich"""
        self.status = ToolStatus.SUCCESS
        self.output = output
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def mark_failed(self, error: str, details: Dict[str, Any] = None) -> None:
        """Markiert als fehlgeschlagen"""
        self.status = ToolStatus.FAILED
        self.error = error
        self.error_details = details
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def mark_cancelled(self) -> None:
        """Markiert als abgebrochen"""
        self.status = ToolStatus.CANCELLED
        self.end_time = datetime.now()

    def mark_timeout(self) -> None:
        """Markiert als Timeout"""
        self.status = ToolStatus.TIMEOUT
        self.error = "Execution timed out"
        self.end_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "status": self.status.name,
            "input": self.input.to_dict() if self.input else None,
            "output": self.output.to_dict() if self.output else None,
            "error": self.error,
            "error_details": self.error_details,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
        }


class ToolInterface(ABC):
    """Abstrakte Basis-Klasse fuer Tools"""

    def __init__(self):
        self._metadata: Optional[ToolMetadata] = None

    @property
    def metadata(self) -> ToolMetadata:
        """Tool-Metadaten"""
        if self._metadata is None:
            self._metadata = self.get_metadata()
        return self._metadata

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Gibt Tool-Metadaten zurueck"""
        pass

    @abstractmethod
    def execute(self, input: ToolInput) -> ToolOutput:
        """Fuehrt Tool aus"""
        pass

    def validate_input(self, input: ToolInput) -> tuple[bool, List[str]]:
        """Validiert Eingabe (kann ueberschrieben werden)"""
        return True, []

    def run(self, input: ToolInput) -> ToolResult:
        """Fuehrt Tool mit vollstaendiger Ergebnis-Verfolgung aus"""
        result = ToolResult(
            tool_name=self.metadata.name,
            input=input,
        )

        # Validierung
        valid, errors = self.validate_input(input)
        if not valid:
            result.mark_failed(
                "Input validation failed",
                {"validation_errors": errors}
            )
            return result

        # Ausfuehrung
        result.mark_started()

        try:
            output = self.execute(input)
            result.mark_success(output)
        except Exception as e:
            result.mark_failed(str(e), {"exception_type": type(e).__name__})

        return result

    async def run_async(self, input: ToolInput) -> ToolResult:
        """Asynchrone Ausfuehrung"""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.run, input
        )

    def __call__(self, **kwargs) -> Any:
        """Kurzform fuer Ausfuehrung"""
        input = ToolInput(parameters=kwargs)
        result = self.run(input)

        if result.failed:
            raise RuntimeError(f"Tool failed: {result.error}")

        return result.output.result if result.output else None


__all__ = [
    'ToolStatus',
    'ToolCategory',
    'ToolMetadata',
    'ToolInput',
    'ToolOutput',
    'ToolResult',
    'ToolInterface',
]
