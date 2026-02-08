"""
SCIO Checkpoint System

Ermöglicht das Fortsetzen von Experimenten nach Unterbrechungen.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from scio.core.config import get_config
from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


@dataclass
class Checkpoint:
    """Ein Checkpoint für ein Experiment."""

    checkpoint_id: str
    execution_id: str
    experiment_name: str
    step_index: int
    completed_steps: list[str]
    step_outputs: dict[str, dict[str, Any]]
    parameters: dict[str, Any]
    variables: dict[str, Any]
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "experiment_name": self.experiment_name,
            "step_index": self.step_index,
            "completed_steps": self.completed_steps,
            "step_outputs": self.step_outputs,
            "parameters": self.parameters,
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            execution_id=data["execution_id"],
            experiment_name=data["experiment_name"],
            step_index=data["step_index"],
            completed_steps=data["completed_steps"],
            step_outputs=data["step_outputs"],
            parameters=data["parameters"],
            variables=data["variables"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class CheckpointManager:
    """
    Verwaltet Checkpoints für Experimente.

    Features:
    - Automatische Checkpoint-Erstellung
    - Resume von Checkpoints
    - Checkpoint-Cleanup
    """

    def __init__(self, storage_path: Optional[Path] = None):
        config = get_config()
        self.storage_path = storage_path or Path(config.data_dir) / "checkpoints"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        execution_id: str,
        experiment_name: str,
        step_index: int,
        completed_steps: list[str],
        step_outputs: dict[str, dict[str, Any]],
        parameters: dict[str, Any],
        variables: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Erstellt einen neuen Checkpoint.

        Args:
            execution_id: Die Execution-ID
            experiment_name: Name des Experiments
            step_index: Index des aktuellen Steps
            completed_steps: Liste der abgeschlossenen Steps
            step_outputs: Outputs der abgeschlossenen Steps
            parameters: Aktuelle Parameter
            variables: Aktuelle Variablen
            metadata: Optionale Metadaten

        Returns:
            Der erstellte Checkpoint
        """
        checkpoint = Checkpoint(
            checkpoint_id=generate_id("ckpt"),
            execution_id=execution_id,
            experiment_name=experiment_name,
            step_index=step_index,
            completed_steps=list(completed_steps),
            step_outputs=dict(step_outputs),
            parameters=dict(parameters),
            variables=dict(variables),
            created_at=now_utc(),
            metadata=metadata or {},
        )

        # Speichere Checkpoint
        self._save_checkpoint(checkpoint)

        logger.info(
            "Checkpoint created",
            checkpoint_id=checkpoint.checkpoint_id,
            execution_id=execution_id,
            step_index=step_index,
        )

        return checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Speichert einen Checkpoint auf Disk."""
        # Speichere nach Execution-ID gruppiert
        exec_dir = self.storage_path / checkpoint.execution_id
        exec_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = exec_dir / f"{checkpoint.checkpoint_id}.json"
        checkpoint_file.write_text(
            json.dumps(checkpoint.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        # Update latest symlink/marker
        latest_file = exec_dir / "latest.json"
        latest_file.write_text(
            json.dumps(checkpoint.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def get_latest_checkpoint(self, execution_id: str) -> Optional[Checkpoint]:
        """
        Holt den neuesten Checkpoint für eine Ausführung.

        Args:
            execution_id: Die Execution-ID

        Returns:
            Der Checkpoint oder None
        """
        latest_file = self.storage_path / execution_id / "latest.json"

        if not latest_file.exists():
            return None

        data = json.loads(latest_file.read_text(encoding="utf-8"))
        return Checkpoint.from_dict(data)

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Holt einen spezifischen Checkpoint.

        Args:
            checkpoint_id: Die Checkpoint-ID

        Returns:
            Der Checkpoint oder None
        """
        # Suche in allen Execution-Verzeichnissen
        for exec_dir in self.storage_path.iterdir():
            if not exec_dir.is_dir():
                continue

            checkpoint_file = exec_dir / f"{checkpoint_id}.json"
            if checkpoint_file.exists():
                data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                return Checkpoint.from_dict(data)

        return None

    def list_checkpoints(
        self,
        execution_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[Checkpoint]:
        """
        Listet Checkpoints.

        Args:
            execution_id: Filter nach Execution-ID
            experiment_name: Filter nach Experiment-Name
            limit: Maximale Anzahl

        Returns:
            Liste von Checkpoints
        """
        checkpoints = []

        if execution_id:
            # Nur ein Verzeichnis durchsuchen
            exec_dir = self.storage_path / execution_id
            if exec_dir.exists():
                for ckpt_file in exec_dir.glob("ckpt-*.json"):
                    data = json.loads(ckpt_file.read_text(encoding="utf-8"))
                    checkpoints.append(Checkpoint.from_dict(data))
        else:
            # Alle Verzeichnisse durchsuchen
            for exec_dir in self.storage_path.iterdir():
                if not exec_dir.is_dir():
                    continue

                for ckpt_file in exec_dir.glob("ckpt-*.json"):
                    data = json.loads(ckpt_file.read_text(encoding="utf-8"))
                    ckpt = Checkpoint.from_dict(data)

                    if experiment_name and ckpt.experiment_name != experiment_name:
                        continue

                    checkpoints.append(ckpt)

                    if len(checkpoints) >= limit:
                        break

                if len(checkpoints) >= limit:
                    break

        # Sortiere nach Erstellungsdatum (neueste zuerst)
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)

        return checkpoints[:limit]

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Löscht einen Checkpoint.

        Args:
            checkpoint_id: Die Checkpoint-ID

        Returns:
            True wenn gelöscht
        """
        for exec_dir in self.storage_path.iterdir():
            if not exec_dir.is_dir():
                continue

            checkpoint_file = exec_dir / f"{checkpoint_id}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info("Checkpoint deleted", checkpoint_id=checkpoint_id)
                return True

        return False

    def cleanup_execution(self, execution_id: str) -> int:
        """
        Löscht alle Checkpoints einer Ausführung.

        Args:
            execution_id: Die Execution-ID

        Returns:
            Anzahl gelöschter Checkpoints
        """
        exec_dir = self.storage_path / execution_id

        if not exec_dir.exists():
            return 0

        count = 0
        for ckpt_file in exec_dir.glob("*.json"):
            ckpt_file.unlink()
            count += 1

        # Lösche leeres Verzeichnis
        try:
            exec_dir.rmdir()
        except OSError:
            pass

        logger.info(
            "Execution checkpoints cleaned",
            execution_id=execution_id,
            count=count,
        )

        return count

    def get_resumable_executions(self) -> list[dict[str, Any]]:
        """
        Listet alle fortsetzbaren Ausführungen.

        Returns:
            Liste von fortsetzbaren Ausführungen
        """
        resumable = []

        for exec_dir in self.storage_path.iterdir():
            if not exec_dir.is_dir():
                continue

            latest_file = exec_dir / "latest.json"
            if latest_file.exists():
                data = json.loads(latest_file.read_text(encoding="utf-8"))
                resumable.append({
                    "execution_id": data["execution_id"],
                    "experiment_name": data["experiment_name"],
                    "checkpoint_id": data["checkpoint_id"],
                    "step_index": data["step_index"],
                    "completed_steps": len(data["completed_steps"]),
                    "created_at": data["created_at"],
                })

        # Sortiere nach Datum (neueste zuerst)
        resumable.sort(key=lambda x: x["created_at"], reverse=True)

        return resumable
