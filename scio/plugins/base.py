"""
SCIO Plugin Base

Basis-Klassen für das Plugin-System.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from scio.agents.base import Agent
from scio.core.logging import get_logger
from scio.tools.base import Tool

logger = get_logger(__name__)


@dataclass
class PluginMetadata:
    """Metadaten eines Plugins."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    dependencies: list[str] = field(default_factory=list)
    scio_version: str = ">=0.1.0"


class Plugin(ABC):
    """
    Abstrakte Basis-Klasse für SCIO-Plugins.

    Plugins können:
    - Neue Agenten registrieren
    - Neue Tools registrieren
    - Hooks registrieren
    - Konfiguration bereitstellen
    """

    metadata: PluginMetadata = PluginMetadata(
        name="base_plugin",
        version="0.0.0",
    )

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__, plugin=self.metadata.name)
        self._registered_agents: list[str] = []
        self._registered_tools: list[str] = []

    @abstractmethod
    def on_load(self) -> None:
        """
        Wird aufgerufen wenn das Plugin geladen wird.

        Hier sollten Agenten und Tools registriert werden.
        """
        pass

    def on_unload(self) -> None:
        """
        Wird aufgerufen wenn das Plugin entladen wird.

        Hier sollten Ressourcen freigegeben werden.
        """
        pass

    def register_agent(self, agent_type: str, agent_class: Type[Agent]) -> None:
        """Registriert einen Agenten."""
        from scio.agents.registry import AgentRegistry

        AgentRegistry.register(agent_type, agent_class)
        self._registered_agents.append(agent_type)
        self.logger.debug("Agent registered", agent_type=agent_type)

    def register_tool(self, tool_name: str, tool_class: Type[Tool]) -> None:
        """Registriert ein Tool."""
        from scio.tools.registry import ToolRegistry

        ToolRegistry.register(tool_name, tool_class)
        self._registered_tools.append(tool_name)
        self.logger.debug("Tool registered", tool_name=tool_name)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Liest Plugin-Konfiguration."""
        return self.config.get(key, default)


class SimplePlugin(Plugin):
    """
    Einfaches Plugin das Agenten und Tools aus Listen registriert.

    Beispiel:
        class MyPlugin(SimplePlugin):
            metadata = PluginMetadata(name="my_plugin", version="1.0")
            agents = [("my_agent", MyAgentClass)]
            tools = [("my_tool", MyToolClass)]
    """

    agents: list[tuple[str, Type[Agent]]] = []
    tools: list[tuple[str, Type[Tool]]] = []

    def on_load(self) -> None:
        """Registriert alle definierten Agenten und Tools."""
        for agent_type, agent_class in self.agents:
            self.register_agent(agent_type, agent_class)

        for tool_name, tool_class in self.tools:
            self.register_tool(tool_name, tool_class)

        self.logger.info(
            "Plugin loaded",
            agents=len(self.agents),
            tools=len(self.tools),
        )
