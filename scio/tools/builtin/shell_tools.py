"""
SCIO Shell Tools

Tools für Shell-Kommandos (eingeschränkt).
"""

import asyncio
import shlex
import subprocess
from typing import Any, Optional

from scio.core.exceptions import SecurityError
from scio.core.logging import get_logger
from scio.tools.base import Tool, ToolConfig
from scio.tools.registry import register_tool

logger = get_logger(__name__)


class ShellConfig(ToolConfig):
    """Konfiguration für Shell Tool."""

    name: str = "shell"
    description: str = "Führt Shell-Kommandos aus (eingeschränkte Befehle)"
    timeout_seconds: int = 60
    max_output_length: int = 50000
    allowed_commands: list[str] = [
        "ls",
        "dir",
        "pwd",
        "echo",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "find",
        "which",
        "whoami",
        "date",
        "uname",
        "python",
        "pip",
        "git",
    ]
    blocked_patterns: list[str] = [
        "rm -rf",
        "sudo",
        "chmod",
        "chown",
        "mkfs",
        "dd if=",
        "> /dev",
        "curl | sh",
        "wget | sh",
        "eval",
        "$()",
        "`",
    ]
    working_dir: Optional[str] = None


@register_tool("shell")
class ShellTool(Tool[dict[str, Any], dict[str, Any]]):
    """Tool für Shell-Kommandos."""

    tool_name = "shell"
    version = "1.0"

    def __init__(self, config: Optional[ShellConfig | dict] = None):
        if config is None:
            config = ShellConfig(name="shell")
        elif isinstance(config, dict):
            config = ShellConfig(**config)
        super().__init__(config)
        self.config: ShellConfig = config

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Führt Shell-Kommando aus."""
        command = input_data.get("command")
        if not command:
            raise ValueError("Kein Kommando angegeben")

        working_dir = input_data.get("working_dir", self.config.working_dir)

        # Sicherheitsprüfungen
        self._validate_command(command)

        self.logger.info("Executing shell command", command=command[:100])

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout_seconds,
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Truncate if needed
            if len(stdout_str) > self.config.max_output_length:
                stdout_str = stdout_str[: self.config.max_output_length] + "\n... (truncated)"
            if len(stderr_str) > self.config.max_output_length:
                stderr_str = stderr_str[: self.config.max_output_length] + "\n... (truncated)"

            return {
                "exit_code": process.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "success": process.returncode == 0,
            }

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Kommando-Timeout nach {self.config.timeout_seconds}s"
            )

    def _validate_command(self, command: str) -> None:
        """Validiert ein Kommando."""
        command_lower = command.lower()

        # Prüfe auf blockierte Patterns
        for pattern in self.config.blocked_patterns:
            if pattern.lower() in command_lower:
                raise SecurityError(f"Blockiertes Pattern: {pattern}")

        # Prüfe erstes Wort (Kommando)
        try:
            parts = shlex.split(command)
            if parts:
                cmd = parts[0].split("/")[-1]  # Entferne Pfad
                if self.config.allowed_commands:
                    if cmd not in self.config.allowed_commands:
                        raise SecurityError(
                            f"Kommando nicht erlaubt: {cmd}. "
                            f"Erlaubt: {self.config.allowed_commands}"
                        )
        except ValueError:
            pass  # shlex konnte nicht parsen, wird später fehlschlagen

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": f"Führt Shell-Kommandos aus. Erlaubt: {', '.join(self.config.allowed_commands)}",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Das auszuführende Shell-Kommando",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Arbeitsverzeichnis für das Kommando",
                    },
                },
                "required": ["command"],
            },
        }
