"""
Execution Protocol
==================

Protokolle fuer Ausfuehrungskontexte, Checkpoints und Fortschritt.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum, auto
from datetime import datetime
import uuid
import json
import hashlib


class ExecutionState(Enum):
    """Ausfuehrungszustand"""
    PENDING = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    RESUMING = auto()
    COMPLETING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()

    @property
    def is_terminal(self) -> bool:
        """Ob Zustand ein Endzustand ist"""
        return self in (
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
            ExecutionState.TIMEOUT,
        )

    @property
    def is_active(self) -> bool:
        """Ob Ausfuehrung aktiv ist"""
        return self in (
            ExecutionState.RUNNING,
            ExecutionState.INITIALIZING,
            ExecutionState.RESUMING,
            ExecutionState.COMPLETING,
        )

    @property
    def can_pause(self) -> bool:
        """Ob pausiert werden kann"""
        return self == ExecutionState.RUNNING

    @property
    def can_resume(self) -> bool:
        """Ob fortgesetzt werden kann"""
        return self == ExecutionState.PAUSED


@dataclass
class Progress:
    """Fortschritts-Tracking"""

    current: int = 0
    total: int = 100
    stage: str = ""
    message: str = ""
    sub_progress: Optional['Progress'] = None
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def percent(self) -> float:
        """Fortschritt in Prozent"""
        if self.total == 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)

    @property
    def is_complete(self) -> bool:
        """Ob abgeschlossen"""
        return self.current >= self.total

    @property
    def elapsed_seconds(self) -> float:
        """Vergangene Zeit in Sekunden"""
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def eta_seconds(self) -> Optional[float]:
        """Geschaetzte verbleibende Zeit"""
        if self.current == 0:
            return None
        rate = self.current / self.elapsed_seconds
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else None

    def update(self, current: int = None, message: str = None, stage: str = None) -> None:
        """Aktualisiert Fortschritt"""
        if current is not None:
            self.current = current
        if message is not None:
            self.message = message
        if stage is not None:
            self.stage = stage
        self.updated_at = datetime.now()

    def increment(self, amount: int = 1) -> None:
        """Erhoeht Fortschritt"""
        self.current = min(self.total, self.current + amount)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "current": self.current,
            "total": self.total,
            "percent": self.percent,
            "stage": self.stage,
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "sub_progress": self.sub_progress.to_dict() if self.sub_progress else None,
        }


@dataclass
class Checkpoint:
    """Checkpoint fuer Wiederaufnahme"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = ""
    name: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    progress: Optional[Progress] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None

    def compute_checksum(self) -> str:
        """Berechnet Pruefsumme des States"""
        state_json = json.dumps(self.state, sort_keys=True, default=str)
        self.checksum = hashlib.sha256(state_json.encode()).hexdigest()[:16]
        return self.checksum

    def verify(self) -> bool:
        """Verifiziert Checkpoint-Integritaet"""
        if not self.checksum:
            return True
        expected = hashlib.sha256(
            json.dumps(self.state, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return self.checksum == expected

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "name": self.name,
            "state": self.state,
            "progress": self.progress.to_dict() if self.progress else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Erstellt aus Dictionary"""
        progress = None
        if data.get("progress"):
            p = data["progress"]
            progress = Progress(
                current=p.get("current", 0),
                total=p.get("total", 100),
                stage=p.get("stage", ""),
                message=p.get("message", ""),
            )

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            execution_id=data.get("execution_id", ""),
            name=data.get("name", ""),
            state=data.get("state", {}),
            progress=progress,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            checksum=data.get("checksum"),
        )


@dataclass
class ExecutionContext:
    """Ausfuehrungskontext mit vollstaendigem State"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    state: ExecutionState = ExecutionState.PENDING
    parameters: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    progress: Progress = field(default_factory=Progress)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Parent/Child Beziehungen
    parent_id: Optional[str] = None
    child_ids: Set[str] = field(default_factory=set)

    def start(self) -> None:
        """Startet Ausfuehrung"""
        self.state = ExecutionState.RUNNING
        self.started_at = datetime.now()

    def pause(self) -> None:
        """Pausiert Ausfuehrung"""
        if self.state.can_pause:
            self.state = ExecutionState.PAUSED

    def resume(self) -> None:
        """Setzt Ausfuehrung fort"""
        if self.state.can_resume:
            self.state = ExecutionState.RESUMING

    def complete(self) -> None:
        """Markiert als abgeschlossen"""
        self.state = ExecutionState.COMPLETED
        self.completed_at = datetime.now()
        self.progress.current = self.progress.total

    def fail(self, error: str, details: Dict[str, Any] = None) -> None:
        """Markiert als fehlgeschlagen"""
        self.state = ExecutionState.FAILED
        self.completed_at = datetime.now()
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "details": details or {},
        })

    def cancel(self) -> None:
        """Bricht Ausfuehrung ab"""
        self.state = ExecutionState.CANCELLED
        self.completed_at = datetime.now()

    def log(self, level: str, message: str, data: Dict[str, Any] = None) -> None:
        """Fuegt Log-Eintrag hinzu"""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data or {},
        })

    def set_variable(self, key: str, value: Any) -> None:
        """Setzt Variable im Kontext"""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Holt Variable aus Kontext"""
        return self.variables.get(key, default)

    def create_checkpoint(self, name: str = "") -> Checkpoint:
        """Erstellt Checkpoint"""
        checkpoint = Checkpoint(
            execution_id=self.id,
            name=name or f"checkpoint_{len(self.checkpoints) + 1}",
            state=self.variables.copy(),
            progress=Progress(
                current=self.progress.current,
                total=self.progress.total,
                stage=self.progress.stage,
            ),
            metadata={"state": self.state.name},
        )
        checkpoint.compute_checksum()
        self.checkpoints.append(checkpoint)
        return checkpoint

    def restore_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Stellt von Checkpoint wieder her"""
        if not checkpoint.verify():
            return False

        self.variables = checkpoint.state.copy()
        if checkpoint.progress:
            self.progress.current = checkpoint.progress.current
            self.progress.stage = checkpoint.progress.stage
        return True

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Holt letzten Checkpoint"""
        return self.checkpoints[-1] if self.checkpoints else None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Ausfuehrungsdauer in Sekunden"""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.name,
            "parameters": self.parameters,
            "variables": self.variables,
            "progress": self.progress.to_dict(),
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "logs": self.logs,
            "errors": self.errors,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "parent_id": self.parent_id,
            "child_ids": list(self.child_ids),
        }


@dataclass
class ExecutionResult:
    """Ergebnis einer Ausfuehrung"""

    execution_id: str = ""
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    context: Optional[ExecutionContext] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_context(cls, context: ExecutionContext, result: Any = None) -> 'ExecutionResult':
        """Erstellt Result aus Context"""
        return cls(
            execution_id=context.id,
            success=context.state == ExecutionState.COMPLETED,
            result=result,
            error=context.errors[-1]["error"] if context.errors else None,
            error_details=context.errors[-1].get("details") if context.errors else None,
            context=context,
            metrics={
                "duration_seconds": context.duration_seconds or 0,
                "checkpoint_count": len(context.checkpoints),
                "log_count": len(context.logs),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "execution_id": self.execution_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "error_details": self.error_details,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }


__all__ = [
    'ExecutionState',
    'Progress',
    'Checkpoint',
    'ExecutionContext',
    'ExecutionResult',
]
