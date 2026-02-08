"""
SCIO Reporter Agent

Agent zur Generierung von Berichten und Ausgaben.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from scio.agents.base import Agent, AgentConfig, AgentContext
from scio.agents.registry import register_agent
from scio.core.exceptions import AgentError
from scio.core.logging import get_logger
from scio.core.utils import now_utc

logger = get_logger(__name__)


class ReporterConfig(AgentConfig):
    """Konfiguration für den Reporter Agent."""

    output_format: str = "json"  # json, markdown, html, txt
    include_metadata: bool = True
    include_timestamp: bool = True
    output_dir: Optional[Path] = None


@register_agent("reporter")
class ReporterAgent(Agent[dict[str, Any], dict[str, Any]]):
    """
    Agent zur Berichterstellung.

    Unterstützte Formate: JSON, Markdown, HTML, TXT
    """

    agent_type = "reporter"
    version = "1.0"

    def __init__(self, config: ReporterConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = ReporterConfig(**config)
        super().__init__(config)
        self.config: ReporterConfig = config

    async def execute(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Generiert einen Bericht."""

        title = input_data.get("title", "SCIO Report")
        data = input_data.get("data", input_data)
        format_override = input_data.get("format", self.config.output_format)

        self.logger.info("Generating report", format=format_override, title=title)

        # Erstelle Report-Struktur
        report = {
            "title": title,
            "generated_at": now_utc().isoformat() if self.config.include_timestamp else None,
            "data": data,
        }

        if self.config.include_metadata:
            report["metadata"] = {
                "agent": self.agent_type,
                "version": self.version,
                "experiment": context.experiment_name,
                "execution_id": context.execution_id,
            }

        # Formatiere Report
        if format_override == "json":
            content = self._format_json(report)
        elif format_override == "markdown":
            content = self._format_markdown(report)
        elif format_override == "html":
            content = self._format_html(report)
        elif format_override == "txt":
            content = self._format_txt(report)
        else:
            raise AgentError(f"Unbekanntes Format: {format_override}")

        # Optional: Speichere in Datei
        output_path = None
        if self.config.output_dir:
            output_path = self._save_report(content, format_override, title)

        return {
            "report": content,
            "format": format_override,
            "output_path": str(output_path) if output_path else None,
        }

    def _format_json(self, report: dict) -> str:
        """Formatiert als JSON."""
        return json.dumps(report, indent=2, ensure_ascii=False, default=str)

    def _format_markdown(self, report: dict) -> str:
        """Formatiert als Markdown."""
        lines = [
            f"# {report['title']}",
            "",
        ]

        if report.get("generated_at"):
            lines.append(f"*Generiert: {report['generated_at']}*")
            lines.append("")

        lines.append("## Ergebnisse")
        lines.append("")
        lines.append(self._dict_to_markdown(report["data"]))

        if report.get("metadata"):
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("## Metadata")
            lines.append("")
            for key, value in report["metadata"].items():
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def _dict_to_markdown(self, data: Any, level: int = 0) -> str:
        """Konvertiert ein Dictionary zu Markdown."""
        lines = []
        indent = "  " * level

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{indent}- **{key}**:")
                    lines.append(self._dict_to_markdown(value, level + 1))
                else:
                    lines.append(f"{indent}- **{key}**: {value}")

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent}-")
                    lines.append(self._dict_to_markdown(item, level + 1))
                else:
                    lines.append(f"{indent}- {item}")

        else:
            lines.append(f"{indent}{data}")

        return "\n".join(lines)

    def _format_html(self, report: dict) -> str:
        """Formatiert als HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{report['title']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .timestamp {{ color: #666; font-style: italic; }}
        .data {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .metadata {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <h1>{report['title']}</h1>
"""
        if report.get("generated_at"):
            html += f'    <p class="timestamp">Generiert: {report["generated_at"]}</p>\n'

        html += '    <div class="data">\n'
        html += f'        <pre>{json.dumps(report["data"], indent=2, ensure_ascii=False, default=str)}</pre>\n'
        html += '    </div>\n'

        if report.get("metadata"):
            html += '    <div class="metadata">\n'
            html += '        <h3>Metadata</h3>\n'
            html += '        <ul>\n'
            for key, value in report["metadata"].items():
                html += f'            <li><strong>{key}</strong>: {value}</li>\n'
            html += '        </ul>\n'
            html += '    </div>\n'

        html += """</body>
</html>"""
        return html

    def _format_txt(self, report: dict) -> str:
        """Formatiert als Plain Text."""
        lines = [
            "=" * 60,
            report["title"].center(60),
            "=" * 60,
            "",
        ]

        if report.get("generated_at"):
            lines.append(f"Generiert: {report['generated_at']}")
            lines.append("")

        lines.append("-" * 60)
        lines.append("ERGEBNISSE")
        lines.append("-" * 60)
        lines.append("")
        lines.append(json.dumps(report["data"], indent=2, ensure_ascii=False, default=str))

        if report.get("metadata"):
            lines.append("")
            lines.append("-" * 60)
            lines.append("METADATA")
            lines.append("-" * 60)
            for key, value in report["metadata"].items():
                lines.append(f"  {key}: {value}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _save_report(self, content: str, format: str, title: str) -> Path:
        """Speichert den Report in eine Datei."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Erstelle Dateinamen
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = {"json": "json", "markdown": "md", "html": "html", "txt": "txt"}[format]

        filename = f"{safe_title}_{timestamp}.{ext}"
        output_path = output_dir / filename

        output_path.write_text(content, encoding="utf-8")
        self.logger.info("Report saved", path=str(output_path))

        return output_path
