"""
SCIO Experiment History

Verfolgt und analysiert die Experiment-Historie.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from scio.core.config import get_config
from scio.core.logging import get_logger
from scio.persistence.store import ExperimentStore, ResultStore

logger = get_logger(__name__)


class ExperimentHistory:
    """
    Verwaltet die Historie von Experimenten und Ausführungen.

    Bietet Funktionen für:
    - Zeitreihenanalyse
    - Vergleich von Ausführungen
    - Trendanalyse
    """

    def __init__(self, storage_path: Optional[Path] = None):
        config = get_config()
        base_path = storage_path or Path(config.data_dir)
        self.experiment_store = ExperimentStore(base_path / "experiments")
        self.result_store = ResultStore(base_path / "results")

    def get_experiment_runs(
        self,
        experiment_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Holt alle Ausführungen eines Experiments.

        Args:
            experiment_name: Name des Experiments
            limit: Maximale Anzahl

        Returns:
            Liste der Ausführungen mit Details
        """
        return self.result_store.list(
            experiment_name=experiment_name,
            limit=limit,
        )

    def compare_runs(
        self,
        execution_ids: list[str],
    ) -> dict[str, Any]:
        """
        Vergleicht mehrere Ausführungen.

        Args:
            execution_ids: Liste von Execution-IDs

        Returns:
            Vergleichsdaten
        """
        results = []
        for exec_id in execution_ids:
            result = self.result_store.load(exec_id)
            if result:
                results.append(result)

        if not results:
            return {"error": "Keine Ergebnisse gefunden"}

        # Analysiere Unterschiede
        comparison = {
            "run_count": len(results),
            "runs": [],
            "duration_comparison": [],
            "status_comparison": [],
            "step_comparison": {},
        }

        for result in results:
            comparison["runs"].append({
                "execution_id": result["execution_id"],
                "status": result["status"],
                "duration_ms": result.get("duration_ms"),
                "started_at": result["started_at"],
            })
            comparison["duration_comparison"].append(result.get("duration_ms", 0))
            comparison["status_comparison"].append(result["status"])

            # Vergleiche Steps
            for step in result.get("step_results", []):
                step_id = step["step_id"]
                if step_id not in comparison["step_comparison"]:
                    comparison["step_comparison"][step_id] = []
                comparison["step_comparison"][step_id].append({
                    "execution_id": result["execution_id"],
                    "status": step["status"],
                    "duration_ms": step.get("duration_ms"),
                })

        # Berechne Statistiken
        durations = [d for d in comparison["duration_comparison"] if d]
        if durations:
            comparison["statistics"] = {
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "success_rate": comparison["status_comparison"].count("completed") / len(results),
            }

        return comparison

    def get_trends(
        self,
        experiment_name: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Analysiert Trends über Zeit.

        Args:
            experiment_name: Name des Experiments
            days: Zeitraum in Tagen

        Returns:
            Trendanalyse
        """
        runs = self.get_experiment_runs(experiment_name, limit=1000)

        # Filtere nach Zeitraum
        cutoff = datetime.now() - timedelta(days=days)
        recent_runs = [
            r for r in runs
            if datetime.fromisoformat(r["started_at"].replace("Z", "+00:00")) > cutoff
        ]

        if not recent_runs:
            return {"error": "Keine Daten im Zeitraum"}

        # Gruppiere nach Tag
        daily_stats = {}
        for run in recent_runs:
            day = run["started_at"][:10]  # YYYY-MM-DD
            if day not in daily_stats:
                daily_stats[day] = {
                    "count": 0,
                    "completed": 0,
                    "failed": 0,
                    "total_duration_ms": 0,
                }

            daily_stats[day]["count"] += 1
            if run["status"] == "completed":
                daily_stats[day]["completed"] += 1
            elif run["status"] == "failed":
                daily_stats[day]["failed"] += 1

            if run.get("duration_ms"):
                daily_stats[day]["total_duration_ms"] += run["duration_ms"]

        # Berechne Durchschnitte
        for day, stats in daily_stats.items():
            if stats["count"] > 0:
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["success_rate"] = stats["completed"] / stats["count"]

        # Sortiere nach Datum
        sorted_days = sorted(daily_stats.keys())

        return {
            "experiment_name": experiment_name,
            "period_days": days,
            "total_runs": len(recent_runs),
            "daily_stats": {day: daily_stats[day] for day in sorted_days},
            "overall": {
                "success_rate": sum(1 for r in recent_runs if r["status"] == "completed") / len(recent_runs),
                "avg_duration_ms": sum(r.get("duration_ms", 0) for r in recent_runs) / len(recent_runs),
            },
        }

    def get_failing_steps(
        self,
        experiment_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Identifiziert häufig fehlschlagende Steps.

        Args:
            experiment_name: Optional filter by experiment
            limit: Maximum number of results

        Returns:
            Liste der häufigsten Fehler
        """
        # Hole alle fehlgeschlagenen Ausführungen
        failed_runs = self.result_store.list(
            experiment_name=experiment_name,
            status="failed",
            limit=500,
        )

        # Sammle fehlgeschlagene Steps
        step_failures = {}
        for run in failed_runs:
            step_results = self.result_store.get_step_results(run["execution_id"])
            for step in step_results:
                if step["status"] == "failed":
                    key = f"{run.get('experiment_name', 'unknown')}:{step['step_id']}"
                    if key not in step_failures:
                        step_failures[key] = {
                            "experiment_name": run.get("experiment_name"),
                            "step_id": step["step_id"],
                            "failure_count": 0,
                            "last_error": None,
                            "last_failure": None,
                        }
                    step_failures[key]["failure_count"] += 1
                    step_failures[key]["last_error"] = step.get("error")
                    step_failures[key]["last_failure"] = run["started_at"]

        # Sortiere nach Häufigkeit
        sorted_failures = sorted(
            step_failures.values(),
            key=lambda x: x["failure_count"],
            reverse=True,
        )

        return sorted_failures[:limit]

    def export_history(
        self,
        experiment_name: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Exportiert die Historie als JSON.

        Args:
            experiment_name: Optional filter
            output_path: Ausgabepfad

        Returns:
            Pfad zur exportierten Datei
        """
        if output_path is None:
            output_path = Path(f"scio_history_{datetime.now():%Y%m%d_%H%M%S}.json")

        data = {
            "exported_at": datetime.now().isoformat(),
            "experiments": self.experiment_store.list(name_filter=experiment_name),
            "results": self.result_store.list(experiment_name=experiment_name, limit=10000),
        }

        output_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info("History exported", path=str(output_path))
        return output_path

    def cleanup_old_results(
        self,
        days: int = 90,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """
        Bereinigt alte Ergebnisse.

        Args:
            days: Ergebnisse älter als X Tage löschen
            dry_run: Nur simulieren, nicht löschen

        Returns:
            Cleanup-Ergebnis
        """
        cutoff = datetime.now() - timedelta(days=days)
        all_results = self.result_store.list(limit=10000)

        to_delete = []
        for result in all_results:
            started = datetime.fromisoformat(result["started_at"].replace("Z", "+00:00"))
            if started < cutoff:
                to_delete.append(result["execution_id"])

        if not dry_run:
            for exec_id in to_delete:
                self.result_store.delete(exec_id)

        return {
            "dry_run": dry_run,
            "cutoff_date": cutoff.isoformat(),
            "results_to_delete": len(to_delete),
            "deleted": 0 if dry_run else len(to_delete),
        }
