"""
SCIO Experiment Protocol

Erstellt detaillierte, reproduzierbare Protokolle von Experiment-Ausführungen.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from scio.core.config import get_config
from scio.core.logging import get_logger
from scio.core.utils import now_utc
from scio.execution.engine import ExecutionEvent, Event, ExecutionResult, StepResult

logger = get_logger(__name__)


@dataclass
class ProtocolEntry:
    """Ein Eintrag im Experiment-Protokoll."""

    timestamp: datetime
    event_type: str
    step_id: Optional[str]
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "step_id": self.step_id,
            "message": self.message,
            "data": self.data,
            "level": self.level,
        }


class ExperimentProtocol:
    """
    Erstellt detaillierte Protokolle für Experimente.

    Features:
    - Chronologische Aufzeichnung aller Events
    - Reproduzierbare Experiment-Logs
    - Export in verschiedene Formate
    """

    def __init__(self, execution_id: str, experiment_name: str):
        self.execution_id = execution_id
        self.experiment_name = experiment_name
        self.entries: list[ProtocolEntry] = []
        self.started_at = now_utc()
        self.completed_at: Optional[datetime] = None
        self.metadata: dict[str, Any] = {}

    def log(
        self,
        event_type: str,
        message: str,
        step_id: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        """
        Fügt einen Eintrag zum Protokoll hinzu.

        Args:
            event_type: Typ des Events
            message: Nachricht
            step_id: Optional Step-ID
            data: Zusätzliche Daten
            level: Log-Level
        """
        entry = ProtocolEntry(
            timestamp=now_utc(),
            event_type=event_type,
            step_id=step_id,
            message=message,
            data=data or {},
            level=level,
        )
        self.entries.append(entry)

    def log_execution_start(self, parameters: dict[str, Any]) -> None:
        """Protokolliert den Start einer Ausführung."""
        self.log(
            event_type="EXECUTION_START",
            message=f"Experiment '{self.experiment_name}' gestartet",
            data={
                "execution_id": self.execution_id,
                "parameters": parameters,
            },
        )

    def log_execution_complete(self, result: ExecutionResult) -> None:
        """Protokolliert das Ende einer Ausführung."""
        self.completed_at = now_utc()
        self.log(
            event_type="EXECUTION_COMPLETE",
            message=f"Experiment abgeschlossen mit Status: {result.status.value}",
            data={
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "step_count": len(result.step_results),
            },
        )

    def log_step_start(self, step_id: str, step_type: str, inputs: dict[str, Any]) -> None:
        """Protokolliert den Start eines Steps."""
        self.log(
            event_type="STEP_START",
            message=f"Step '{step_id}' gestartet ({step_type})",
            step_id=step_id,
            data={
                "type": step_type,
                "inputs": self._sanitize_data(inputs),
            },
        )

    def log_step_complete(self, step_result: StepResult) -> None:
        """Protokolliert das Ende eines Steps."""
        self.log(
            event_type="STEP_COMPLETE",
            message=f"Step '{step_result.step_id}' abgeschlossen: {step_result.status.value}",
            step_id=step_result.step_id,
            data={
                "status": step_result.status.value,
                "duration_ms": step_result.duration_ms,
                "outputs": self._sanitize_data(step_result.outputs),
                "error": step_result.error,
            },
            level="ERROR" if step_result.error else "INFO",
        )

    def log_checkpoint(self, checkpoint_id: str) -> None:
        """Protokolliert einen Checkpoint."""
        self.log(
            event_type="CHECKPOINT",
            message=f"Checkpoint erstellt: {checkpoint_id}",
            data={"checkpoint_id": checkpoint_id},
        )

    def log_warning(self, message: str, step_id: Optional[str] = None, data: Optional[dict] = None) -> None:
        """Protokolliert eine Warnung."""
        self.log(
            event_type="WARNING",
            message=message,
            step_id=step_id,
            data=data or {},
            level="WARNING",
        )

    def log_error(self, message: str, step_id: Optional[str] = None, error: Optional[str] = None) -> None:
        """Protokolliert einen Fehler."""
        self.log(
            event_type="ERROR",
            message=message,
            step_id=step_id,
            data={"error": error} if error else {},
            level="ERROR",
        )

    def _sanitize_data(self, data: Any, max_length: int = 1000) -> Any:
        """Bereinigt Daten für das Protokoll (begrenzt Länge, entfernt Binärdaten)."""
        if data is None:
            return None

        if isinstance(data, (str, int, float, bool)):
            if isinstance(data, str) and len(data) > max_length:
                return data[:max_length] + "... (truncated)"
            return data

        if isinstance(data, bytes):
            return f"<binary data, {len(data)} bytes>"

        if isinstance(data, dict):
            return {k: self._sanitize_data(v, max_length) for k, v in data.items()}

        if isinstance(data, list):
            if len(data) > 100:
                return [self._sanitize_data(v, max_length) for v in data[:100]] + ["... (truncated)"]
            return [self._sanitize_data(v, max_length) for v in data]

        return str(data)[:max_length]

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert das Protokoll zu einem Dictionary."""
        return {
            "execution_id": self.execution_id,
            "experiment_name": self.experiment_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        """Exportiert das Protokoll als JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Exportiert das Protokoll als Markdown."""
        lines = [
            f"# Experiment Protocol: {self.experiment_name}",
            "",
            f"**Execution ID:** `{self.execution_id}`",
            f"**Started:** {self.started_at.isoformat()}",
            f"**Completed:** {self.completed_at.isoformat() if self.completed_at else 'In Progress'}",
            "",
            "## Timeline",
            "",
        ]

        for entry in self.entries:
            time_str = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
            icon = self._get_icon(entry.level, entry.event_type)
            step_info = f" [{entry.step_id}]" if entry.step_id else ""

            lines.append(f"- `{time_str}` {icon}{step_info} {entry.message}")

            if entry.data and entry.event_type not in ("STEP_START",):
                for key, value in entry.data.items():
                    if key != "inputs" and value is not None:
                        lines.append(f"  - **{key}:** {value}")

        lines.append("")
        lines.append("---")
        lines.append(f"*Generated by SCIO at {now_utc().isoformat()}*")

        return "\n".join(lines)

    def _get_icon(self, level: str, event_type: str) -> str:
        """Gibt ein Icon für den Event-Typ zurück."""
        if level == "ERROR":
            return "[X]"
        if level == "WARNING":
            return "[!]"
        if "START" in event_type:
            return "[>]"
        if "COMPLETE" in event_type:
            return "[OK]"
        if "CHECKPOINT" in event_type:
            return "[#]"
        return "[*]"


class ProtocolManager:
    """
    Verwaltet Experiment-Protokolle.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        config = get_config()
        self.storage_path = storage_path or Path(config.data_dir) / "protocols"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, protocol: ExperimentProtocol) -> Path:
        """
        Speichert ein Protokoll.

        Args:
            protocol: Das zu speichernde Protokoll

        Returns:
            Pfad zur gespeicherten Datei
        """
        # JSON-Format
        json_path = self.storage_path / f"{protocol.execution_id}.json"
        json_path.write_text(protocol.to_json(), encoding="utf-8")

        # Markdown-Format
        md_path = self.storage_path / f"{protocol.execution_id}.md"
        md_path.write_text(protocol.to_markdown(), encoding="utf-8")

        logger.info(
            "Protocol saved",
            execution_id=protocol.execution_id,
            path=str(json_path),
        )

        return json_path

    def load(self, execution_id: str) -> Optional[dict[str, Any]]:
        """
        Lädt ein Protokoll.

        Args:
            execution_id: Die Execution-ID

        Returns:
            Das Protokoll als Dict oder None
        """
        json_path = self.storage_path / f"{execution_id}.json"

        if not json_path.exists():
            return None

        return json.loads(json_path.read_text(encoding="utf-8"))

    def list_protocols(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Listet alle Protokolle.

        Args:
            limit: Maximale Anzahl

        Returns:
            Liste von Protokoll-Infos
        """
        protocols = []

        for json_file in sorted(
            self.storage_path.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]:
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                protocols.append({
                    "execution_id": data["execution_id"],
                    "experiment_name": data["experiment_name"],
                    "started_at": data["started_at"],
                    "completed_at": data.get("completed_at"),
                    "entry_count": len(data.get("entries", [])),
                })
            except Exception as e:
                logger.warning(f"Failed to parse protocol: {json_file}", error=str(e))

        return protocols

    def delete(self, execution_id: str) -> bool:
        """Löscht ein Protokoll."""
        deleted = False

        for ext in (".json", ".md"):
            path = self.storage_path / f"{execution_id}{ext}"
            if path.exists():
                path.unlink()
                deleted = True

        return deleted
