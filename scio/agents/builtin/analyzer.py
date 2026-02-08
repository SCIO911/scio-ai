"""
SCIO Analyzer Agent

Agent für statistische Analysen und Datenverarbeitung.
"""

import statistics
from typing import Any, Optional

from scio.agents.base import Agent, AgentConfig, AgentContext
from scio.agents.registry import register_agent
from scio.core.exceptions import AgentError
from scio.core.logging import get_logger

logger = get_logger(__name__)


class AnalyzerConfig(AgentConfig):
    """Konfiguration für den Analyzer Agent."""

    methods: list[str] = ["mean", "std", "median", "min", "max", "count"]
    decimal_places: int = 4
    handle_missing: str = "skip"  # skip, zero, error


@register_agent("analyzer")
class AnalyzerAgent(Agent[dict[str, Any], dict[str, Any]]):
    """
    Agent für statistische Analysen.

    Unterstützte Methoden: mean, std, median, min, max, count, sum, variance
    """

    agent_type = "analyzer"
    version = "1.0"

    AVAILABLE_METHODS = {
        "mean": statistics.mean,
        "median": statistics.median,
        "std": statistics.stdev,
        "variance": statistics.variance,
        "min": min,
        "max": max,
        "sum": sum,
        "count": len,
    }

    def __init__(self, config: AnalyzerConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = AnalyzerConfig(**config)
        super().__init__(config)
        self.config: AnalyzerConfig = config

    async def execute(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Führt statistische Analyse durch."""

        data = input_data.get("data")
        if data is None:
            raise AgentError("Keine Daten zum Analysieren", agent_id=self.agent_id)

        columns = input_data.get("columns")  # Optional: spezifische Spalten

        self.logger.info(
            "Analyzing data",
            methods=self.config.methods,
            data_type=type(data).__name__,
        )

        results = {}

        # Wenn Liste von Dictionaries (tabellarische Daten)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            results = self._analyze_tabular(data, columns)

        # Wenn einfache Liste von Zahlen
        elif isinstance(data, list):
            results = self._analyze_list(data)

        # Wenn Dictionary mit numerischen Werten
        elif isinstance(data, dict):
            results = self._analyze_dict(data)

        else:
            raise AgentError(
                f"Nicht unterstützter Datentyp: {type(data).__name__}",
                agent_id=self.agent_id,
            )

        return {
            "statistics": results,
            "methods_applied": self.config.methods,
            "data_points": len(data) if isinstance(data, list) else None,
        }

    def _analyze_list(self, data: list) -> dict[str, Any]:
        """Analysiert eine Liste von Zahlen."""
        numeric_data = self._extract_numeric(data)

        if not numeric_data:
            return {"error": "Keine numerischen Daten gefunden"}

        return self._compute_statistics(numeric_data)

    def _analyze_tabular(
        self, data: list[dict], columns: Optional[list[str]] = None
    ) -> dict[str, dict[str, Any]]:
        """Analysiert tabellarische Daten."""
        results = {}

        # Bestimme Spalten
        if columns is None:
            columns = list(data[0].keys()) if data else []

        for col in columns:
            values = [row.get(col) for row in data if col in row]
            numeric_values = self._extract_numeric(values)

            if numeric_values:
                results[col] = self._compute_statistics(numeric_values)
            else:
                results[col] = {
                    "count": len(values),
                    "type": "non-numeric",
                    "unique": len(set(str(v) for v in values if v is not None)),
                }

        return results

    def _analyze_dict(self, data: dict) -> dict[str, Any]:
        """Analysiert ein Dictionary."""
        numeric_values = self._extract_numeric(list(data.values()))

        if numeric_values:
            return self._compute_statistics(numeric_values)

        return {"keys": list(data.keys()), "count": len(data)}

    def _extract_numeric(self, values: list) -> list[float]:
        """Extrahiert numerische Werte aus einer Liste."""
        result = []

        for v in values:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                result.append(float(v))
            elif isinstance(v, str):
                try:
                    result.append(float(v))
                except ValueError:
                    if self.config.handle_missing == "zero":
                        result.append(0.0)
                    elif self.config.handle_missing == "error":
                        raise AgentError(f"Nicht-numerischer Wert: {v}")
                    # skip: nichts tun
            elif v is None:
                if self.config.handle_missing == "zero":
                    result.append(0.0)
                elif self.config.handle_missing == "error":
                    raise AgentError("None-Wert in Daten")

        return result

    def _compute_statistics(self, data: list[float]) -> dict[str, Any]:
        """Berechnet Statistiken für numerische Daten."""
        results = {}
        dp = self.config.decimal_places

        for method in self.config.methods:
            if method not in self.AVAILABLE_METHODS:
                self.logger.warning(f"Unbekannte Methode: {method}")
                continue

            try:
                func = self.AVAILABLE_METHODS[method]

                # std und variance brauchen mindestens 2 Werte
                if method in ("std", "variance") and len(data) < 2:
                    results[method] = None
                    continue

                value = func(data)
                results[method] = round(value, dp) if isinstance(value, float) else value

            except Exception as e:
                self.logger.warning(f"Fehler bei {method}: {e}")
                results[method] = None

        return results
