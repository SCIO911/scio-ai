"""
SCIO Data Loader Agent

Agent zum Laden und Validieren von Daten aus verschiedenen Quellen.
"""

import json
from pathlib import Path
from typing import Any, Optional

from scio.agents.base import Agent, AgentConfig, AgentContext, AgentResult
from scio.agents.registry import register_agent
from scio.core.exceptions import AgentError
from scio.core.logging import get_logger

logger = get_logger(__name__)


class DataLoaderConfig(AgentConfig):
    """Konfiguration für den DataLoader Agent."""

    supported_formats: list[str] = ["csv", "json", "yaml", "parquet", "txt"]
    validate_schema: bool = True
    encoding: str = "utf-8"
    max_file_size_mb: int = 100


@register_agent("data_loader")
class DataLoaderAgent(Agent[dict[str, Any], dict[str, Any]]):
    """
    Agent zum Laden von Daten aus Dateien.

    Unterstützt: CSV, JSON, YAML, Parquet, TXT
    """

    agent_type = "data_loader"
    version = "1.0"

    def __init__(self, config: DataLoaderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = DataLoaderConfig(**config)
        super().__init__(config)
        self.config: DataLoaderConfig = config

    async def execute(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Lädt Daten aus einer Datei oder direkt aus dem Input."""

        # Support für direkte Daten-Eingabe
        if "data" in input_data:
            data = input_data["data"]
            fmt = input_data.get("format", "json")

            # Parse string data based on format
            if isinstance(data, str):
                if fmt == "json":
                    data = json.load(__import__("io").StringIO(data)) if data.strip().startswith(("{", "[")) else data
                    try:
                        data = json.loads(data) if isinstance(data, str) else data
                    except json.JSONDecodeError:
                        pass

            return {
                "data": data,
                "metadata": {
                    "source": "inline",
                    "format": fmt,
                },
            }

        path = input_data.get("path")
        if not path:
            raise AgentError("Kein Dateipfad oder Daten angegeben", agent_id=self.agent_id)

        path = Path(path)
        if not path.exists():
            raise AgentError(f"Datei nicht gefunden: {path}", agent_id=self.agent_id)

        # Prüfe Dateigröße
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise AgentError(
                f"Datei zu groß: {size_mb:.1f}MB > {self.config.max_file_size_mb}MB",
                agent_id=self.agent_id,
            )

        # Lade basierend auf Format
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in self.config.supported_formats:
            raise AgentError(
                f"Nicht unterstütztes Format: {suffix}",
                agent_id=self.agent_id,
                details={"supported": self.config.supported_formats},
            )

        self.logger.info("Loading data", path=str(path), format=suffix)

        data = await self._load_file(path, suffix)

        return {
            "data": data,
            "metadata": {
                "path": str(path),
                "format": suffix,
                "size_bytes": path.stat().st_size,
                "rows": len(data) if isinstance(data, list) else None,
            },
        }

    async def _load_file(self, path: Path, format: str) -> Any:
        """Lädt eine Datei basierend auf dem Format."""

        if format == "json":
            with open(path, encoding=self.config.encoding) as f:
                return json.load(f)

        elif format == "yaml" or format == "yml":
            import yaml
            with open(path, encoding=self.config.encoding) as f:
                return yaml.safe_load(f)

        elif format == "csv":
            return self._load_csv(path)

        elif format == "txt":
            with open(path, encoding=self.config.encoding) as f:
                return f.read()

        elif format == "parquet":
            return self._load_parquet(path)

        else:
            raise AgentError(f"Unbekanntes Format: {format}")

    def _load_csv(self, path: Path) -> list[dict[str, Any]]:
        """Lädt eine CSV-Datei."""
        import csv

        with open(path, encoding=self.config.encoding, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _load_parquet(self, path: Path) -> list[dict[str, Any]]:
        """Lädt eine Parquet-Datei."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            return table.to_pylist()
        except ImportError:
            raise AgentError(
                "PyArrow nicht installiert. Installiere mit: pip install pyarrow"
            )
