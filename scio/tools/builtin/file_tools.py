"""
SCIO File Tools

Tools für Dateioperationen.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from scio.core.exceptions import SecurityError
from scio.core.logging import get_logger
from scio.tools.base import Tool, ToolConfig
from scio.tools.registry import register_tool

logger = get_logger(__name__)


class FileReaderConfig(ToolConfig):
    """Konfiguration für FileReader."""

    name: str = "file_reader"
    description: str = "Liest Dateien und gibt deren Inhalt zurück"
    allowed_extensions: list[str] = [".txt", ".json", ".yaml", ".yml", ".csv", ".md"]
    max_file_size_mb: int = 50
    encoding: str = "utf-8"


@register_tool("file_reader")
class FileReaderTool(Tool[dict[str, Any], dict[str, Any]]):
    """Tool zum Lesen von Dateien."""

    tool_name = "file_reader"
    version = "1.0"

    def __init__(self, config: Optional[FileReaderConfig | dict] = None):
        if config is None:
            config = FileReaderConfig(name="file_reader")
        elif isinstance(config, dict):
            config = FileReaderConfig(**config)
        super().__init__(config)
        self.config: FileReaderConfig = config

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Liest eine Datei."""
        path_str = input_data.get("path")
        if not path_str:
            raise ValueError("Kein Pfad angegeben")

        path = Path(path_str)

        # Sicherheitsprüfungen
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")

        ext = path.suffix.lower()
        if self.config.allowed_extensions and ext not in self.config.allowed_extensions:
            raise SecurityError(f"Dateityp nicht erlaubt: {ext}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise SecurityError(f"Datei zu groß: {size_mb:.1f}MB")

        # Lese Datei
        content = path.read_text(encoding=self.config.encoding)

        # Parse basierend auf Format
        if ext == ".json":
            data = json.loads(content)
        elif ext in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(content)
        else:
            data = content

        return {
            "content": data,
            "path": str(path.absolute()),
            "size_bytes": path.stat().st_size,
            "format": ext.lstrip("."),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": "Liest eine Datei und gibt den Inhalt zurück",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Pfad zur Datei",
                    },
                },
                "required": ["path"],
            },
        }


class FileWriterConfig(ToolConfig):
    """Konfiguration für FileWriter."""

    name: str = "file_writer"
    description: str = "Schreibt Inhalte in Dateien"
    allowed_extensions: list[str] = [".txt", ".json", ".yaml", ".yml", ".csv", ".md"]
    create_dirs: bool = True
    overwrite: bool = False
    encoding: str = "utf-8"


@register_tool("file_writer")
class FileWriterTool(Tool[dict[str, Any], dict[str, Any]]):
    """Tool zum Schreiben von Dateien."""

    tool_name = "file_writer"
    version = "1.0"

    def __init__(self, config: Optional[FileWriterConfig | dict] = None):
        if config is None:
            config = FileWriterConfig(name="file_writer")
        elif isinstance(config, dict):
            config = FileWriterConfig(**config)
        super().__init__(config)
        self.config: FileWriterConfig = config

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Schreibt in eine Datei."""
        path_str = input_data.get("path")
        content = input_data.get("content")

        if not path_str:
            raise ValueError("Kein Pfad angegeben")
        if content is None:
            raise ValueError("Kein Inhalt angegeben")

        path = Path(path_str)

        # Sicherheitsprüfungen
        ext = path.suffix.lower()
        if self.config.allowed_extensions and ext not in self.config.allowed_extensions:
            raise SecurityError(f"Dateityp nicht erlaubt: {ext}")

        if path.exists() and not self.config.overwrite:
            raise FileExistsError(f"Datei existiert bereits: {path}")

        # Erstelle Verzeichnisse
        if self.config.create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Konvertiere Inhalt
        if ext == ".json" and isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2, ensure_ascii=False)
        elif ext in (".yaml", ".yml") and isinstance(content, (dict, list)):
            import yaml
            content = yaml.dump(content, allow_unicode=True, default_flow_style=False)
        elif not isinstance(content, str):
            content = str(content)

        # Schreibe Datei
        path.write_text(content, encoding=self.config.encoding)

        return {
            "path": str(path.absolute()),
            "size_bytes": path.stat().st_size,
            "format": ext.lstrip("."),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": "Schreibt Inhalt in eine Datei",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Pfad zur Datei",
                    },
                    "content": {
                        "type": ["string", "object", "array"],
                        "description": "Zu schreibender Inhalt",
                    },
                },
                "required": ["path", "content"],
            },
        }
