"""
SCIO Persistence Store

Speichert Experimente und Ergebnisse.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from scio.core.config import get_config
from scio.core.logging import get_logger
from scio.core.utils import now_utc

if TYPE_CHECKING:
    from scio.execution.engine import ExecutionResult, ExecutionStatus, StepResult
    from scio.parser.schema import ExperimentSchema

logger = get_logger(__name__)


class ExperimentStore:
    """
    Speichert und lädt Experiment-Definitionen.

    Unterstützt JSON und SQLite als Backend.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        config = get_config()
        self.storage_path = storage_path or Path(config.data_dir) / "experiments"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "experiments.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialisiert die Datenbank."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags TEXT,
                    author TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_name
                ON experiments(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_tags
                ON experiments(tags)
            """)

    def save(self, experiment: Any, experiment_id: Optional[str] = None) -> str:
        """
        Speichert ein Experiment.

        Args:
            experiment: Das zu speichernde Experiment
            experiment_id: Optionale ID (wird generiert wenn nicht angegeben)

        Returns:
            Die Experiment-ID
        """
        from scio.core.utils import generate_id

        exp_id = experiment_id or generate_id("exp")
        now = now_utc().isoformat()

        # Konvertiere zu JSON
        content = experiment.model_dump_json(indent=2)

        # Tags als JSON-String
        tags = json.dumps(experiment.metadata.tags) if experiment.metadata.tags else "[]"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (id, name, version, content, created_at, updated_at, tags, author)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id,
                experiment.name,
                experiment.version,
                content,
                now,
                now,
                tags,
                experiment.metadata.author,
            ))

        # Speichere auch als JSON-Datei
        json_path = self.storage_path / f"{exp_id}.json"
        json_path.write_text(content, encoding="utf-8")

        logger.info("Experiment saved", experiment_id=exp_id, name=experiment.name)
        return exp_id

    def load(self, experiment_id: str) -> Optional[Any]:
        """
        Lädt ein Experiment.

        Args:
            experiment_id: Die Experiment-ID

        Returns:
            Das Experiment (ExperimentSchema) oder None
        """
        from scio.parser.schema import ExperimentSchema

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT content FROM experiments WHERE id = ?",
                (experiment_id,)
            ).fetchone()

        if row:
            return ExperimentSchema.model_validate_json(row[0])
        return None

    def list(
        self,
        name_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Listet gespeicherte Experimente.

        Args:
            name_filter: Filter nach Name (LIKE)
            tag_filter: Filter nach Tag
            limit: Maximale Anzahl

        Returns:
            Liste von Experiment-Infos
        """
        query = "SELECT id, name, version, created_at, author, tags FROM experiments"
        params = []
        conditions = []

        if name_filter:
            conditions.append("name LIKE ?")
            params.append(f"%{name_filter}%")

        if tag_filter:
            conditions.append("tags LIKE ?")
            params.append(f"%{tag_filter}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "version": row["version"],
                "created_at": row["created_at"],
                "author": row["author"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
            }
            for row in rows
        ]

    def delete(self, experiment_id: str) -> bool:
        """Löscht ein Experiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM experiments WHERE id = ?",
                (experiment_id,)
            )

        # Lösche auch JSON-Datei
        json_path = self.storage_path / f"{experiment_id}.json"
        if json_path.exists():
            json_path.unlink()

        return cursor.rowcount > 0


class ResultStore:
    """
    Speichert und lädt Ausführungsergebnisse.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        config = get_config()
        self.storage_path = storage_path or Path(config.data_dir) / "results"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "results.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialisiert die Datenbank."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    execution_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    experiment_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_ms INTEGER,
                    content TEXT NOT NULL,
                    step_count INTEGER,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_experiment
                ON results(experiment_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_status
                ON results(status)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS step_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_ms INTEGER,
                    outputs TEXT,
                    error TEXT,
                    FOREIGN KEY (execution_id) REFERENCES results(execution_id)
                )
            """)

    def save(
        self,
        result: Any,  # ExecutionResult
        experiment_id: Optional[str] = None,
    ) -> str:
        """
        Speichert ein Ausführungsergebnis.

        Args:
            result: Das Ergebnis (ExecutionResult)
            experiment_id: Optionale Referenz zum Experiment

        Returns:
            Die Execution-ID
        """
        content = json.dumps(result.to_dict(), indent=2, default=str)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO results
                (execution_id, experiment_id, experiment_name, status,
                 started_at, completed_at, duration_ms, content, step_count, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.execution_id,
                experiment_id,
                result.experiment_name,
                result.status.value,
                result.started_at.isoformat(),
                result.completed_at.isoformat() if result.completed_at else None,
                result.duration_ms,
                content,
                len(result.step_results),
                result.metadata.get("error"),
            ))

            # Speichere Step-Ergebnisse
            for step_result in result.step_results:
                conn.execute("""
                    INSERT INTO step_results
                    (execution_id, step_id, status, started_at, completed_at,
                     duration_ms, outputs, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.execution_id,
                    step_result.step_id,
                    step_result.status.value,
                    step_result.started_at.isoformat(),
                    step_result.completed_at.isoformat() if step_result.completed_at else None,
                    step_result.duration_ms,
                    json.dumps(step_result.outputs, default=str),
                    step_result.error,
                ))

        # Speichere auch als JSON
        json_path = self.storage_path / f"{result.execution_id}.json"
        json_path.write_text(content, encoding="utf-8")

        logger.info(
            "Result saved",
            execution_id=result.execution_id,
            status=result.status.value,
        )
        return result.execution_id

    def load(self, execution_id: str) -> Optional[dict[str, Any]]:
        """
        Lädt ein Ausführungsergebnis.

        Args:
            execution_id: Die Execution-ID

        Returns:
            Das Ergebnis als Dict oder None
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT content FROM results WHERE execution_id = ?",
                (execution_id,)
            ).fetchone()

        if row:
            return json.loads(row[0])
        return None

    def list(
        self,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Listet Ausführungsergebnisse.

        Args:
            experiment_id: Filter nach Experiment-ID
            experiment_name: Filter nach Experiment-Name
            status: Filter nach Status
            limit: Maximale Anzahl

        Returns:
            Liste von Ergebnis-Infos
        """
        query = """
            SELECT execution_id, experiment_id, experiment_name, status,
                   started_at, completed_at, duration_ms, step_count, error
            FROM results
        """
        params = []
        conditions = []

        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)

        if experiment_name:
            conditions.append("experiment_name LIKE ?")
            params.append(f"%{experiment_name}%")

        if status:
            conditions.append("status = ?")
            params.append(status)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_step_results(self, execution_id: str) -> list[dict[str, Any]]:
        """Lädt die Step-Ergebnisse für eine Ausführung."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM step_results WHERE execution_id = ? ORDER BY id",
                (execution_id,)
            ).fetchall()

        return [
            {
                **dict(row),
                "outputs": json.loads(row["outputs"]) if row["outputs"] else {},
            }
            for row in rows
        ]

    def delete(self, execution_id: str) -> bool:
        """Löscht ein Ergebnis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM step_results WHERE execution_id = ?",
                (execution_id,)
            )
            cursor = conn.execute(
                "DELETE FROM results WHERE execution_id = ?",
                (execution_id,)
            )

        # Lösche auch JSON-Datei
        json_path = self.storage_path / f"{execution_id}.json"
        if json_path.exists():
            json_path.unlink()

        return cursor.rowcount > 0

    def get_statistics(
        self,
        experiment_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Berechnet Statistiken über Ausführungen.

        Args:
            experiment_name: Optional filter by experiment name

        Returns:
            Statistiken
        """
        base_query = "SELECT status, COUNT(*) as count FROM results"
        params = []

        if experiment_name:
            base_query += " WHERE experiment_name = ?"
            params.append(experiment_name)

        base_query += " GROUP BY status"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Status-Verteilung
            status_rows = conn.execute(base_query, params).fetchall()
            status_counts = {row["status"]: row["count"] for row in status_rows}

            # Durchschnittliche Dauer
            duration_query = "SELECT AVG(duration_ms) as avg_duration FROM results WHERE duration_ms IS NOT NULL"
            if experiment_name:
                duration_query += " AND experiment_name = ?"

            avg_duration = conn.execute(
                duration_query,
                [experiment_name] if experiment_name else []
            ).fetchone()

        total = sum(status_counts.values())

        return {
            "total_executions": total,
            "status_distribution": status_counts,
            "success_rate": status_counts.get("completed", 0) / total if total > 0 else 0,
            "average_duration_ms": avg_duration["avg_duration"] if avg_duration else None,
        }
