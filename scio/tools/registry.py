"""
SCIO Tool Registry

Registrierung und Verwaltung von Tools.
"""

from typing import Any, Type

from scio.core.exceptions import PluginError
from scio.core.logging import get_logger
from scio.tools.base import Tool, ToolConfig

logger = get_logger(__name__)


class ToolRegistry:
    """
    Registry für Tools.

    Ermöglicht die Registrierung und Instanziierung von Tools.
    """

    _instance: "ToolRegistry | None" = None
    _tools: dict[str, Type[Tool]] = {}

    def __new__(cls) -> "ToolRegistry":
        """Singleton Pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, tool_name: str, tool_class: Type[Tool]) -> None:
        """
        Registriert ein Tool.

        Args:
            tool_name: Eindeutiger Name
            tool_class: Tool-Klasse
        """
        if tool_name in cls._tools:
            logger.warning("Overwriting existing tool", tool_name=tool_name)

        cls._tools[tool_name] = tool_class
        logger.debug("Tool registered", tool_name=tool_name)

    @classmethod
    def get(cls, tool_name: str) -> Type[Tool]:
        """
        Gibt eine Tool-Klasse zurück.

        Args:
            tool_name: Tool-Name

        Returns:
            Tool-Klasse

        Raises:
            PluginError: Wenn Tool nicht gefunden
        """
        if tool_name not in cls._tools:
            raise PluginError(
                f"Unbekanntes Tool: {tool_name}",
                plugin_name=tool_name,
                details={"available": list(cls._tools.keys())},
            )

        return cls._tools[tool_name]

    @classmethod
    def create(
        cls,
        tool_name: str,
        config: ToolConfig | dict[str, Any] | None = None,
    ) -> Tool:
        """
        Erstellt eine Tool-Instanz.

        Args:
            tool_name: Tool-Name
            config: Optionale Konfiguration

        Returns:
            Tool-Instanz
        """
        tool_class = cls.get(tool_name)
        return tool_class(config)

    @classmethod
    def list_tools(cls) -> list[str]:
        """Gibt alle registrierten Tools zurück."""
        return list(cls._tools.keys())

    @classmethod
    def get_schemas(cls) -> list[dict[str, Any]]:
        """Gibt alle Tool-Schemas zurück (für LLM)."""
        schemas = []
        for tool_name in cls._tools:
            tool = cls.create(tool_name)
            schemas.append(tool.get_schema())
        return schemas

    @classmethod
    def clear(cls) -> None:
        """Löscht alle Registrierungen (für Tests)."""
        cls._tools.clear()

    @classmethod
    def register_builtins(cls) -> None:
        """Registriert alle Builtin-Tools neu."""
        import importlib
        # Reload each submodule first
        import scio.tools.builtin.file_tools as file_tools
        import scio.tools.builtin.http_tools as http_tools
        import scio.tools.builtin.python_executor as python_executor
        import scio.tools.builtin.shell_tools as shell_tools
        import scio.tools.builtin.math_tools as math_tools
        importlib.reload(file_tools)
        importlib.reload(http_tools)
        importlib.reload(python_executor)
        importlib.reload(shell_tools)
        importlib.reload(math_tools)
        # Then reload the main module
        import scio.tools.builtin as builtin_module
        importlib.reload(builtin_module)


def register_tool(tool_name: str):
    """
    Decorator zum Registrieren von Tools.

    Beispiel:
        @register_tool("file_reader")
        class FileReaderTool(Tool):
            ...
    """

    def decorator(cls: Type[Tool]) -> Type[Tool]:
        cls.tool_name = tool_name
        ToolRegistry.register(tool_name, cls)
        return cls

    return decorator
