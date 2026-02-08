"""
SCIO Agent Registry

Registrierung und Verwaltung von Agenten-Typen.
"""

from typing import Any, Type

from scio.agents.base import Agent, AgentConfig
from scio.core.exceptions import AgentError
from scio.core.logging import get_logger

logger = get_logger(__name__)


class AgentRegistry:
    """
    Registry für Agenten-Typen.

    Ermöglicht die Registrierung und Instanziierung von Agenten
    basierend auf ihrem Typ-Namen.
    """

    _instance: "AgentRegistry | None" = None
    _agents: dict[str, Type[Agent]] = {}

    def __new__(cls) -> "AgentRegistry":
        """Singleton Pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, agent_type: str, agent_class: Type[Agent]) -> None:
        """
        Registriert einen Agenten-Typ.

        Args:
            agent_type: Eindeutiger Typ-Name
            agent_class: Agent-Klasse
        """
        if agent_type in cls._agents:
            logger.warning(
                "Overwriting existing agent type",
                agent_type=agent_type,
            )

        cls._agents[agent_type] = agent_class
        logger.debug("Agent registered", agent_type=agent_type)

    @classmethod
    def get(cls, agent_type: str) -> Type[Agent]:
        """
        Gibt eine Agent-Klasse zurück.

        Args:
            agent_type: Typ-Name

        Returns:
            Agent-Klasse

        Raises:
            AgentError: Wenn Typ nicht gefunden
        """
        if agent_type not in cls._agents:
            raise AgentError(
                f"Unbekannter Agent-Typ: {agent_type}",
                details={"available": list(cls._agents.keys())},
            )

        return cls._agents[agent_type]

    @classmethod
    def create(
        cls,
        agent_type: str,
        config: AgentConfig | dict[str, Any],
    ) -> Agent:
        """
        Erstellt eine Agent-Instanz.

        Args:
            agent_type: Typ-Name
            config: Agent-Konfiguration

        Returns:
            Agent-Instanz
        """
        agent_class = cls.get(agent_type)
        return agent_class(config)

    @classmethod
    def list_types(cls) -> list[str]:
        """Gibt alle registrierten Typen zurück."""
        return list(cls._agents.keys())

    @classmethod
    def clear(cls) -> None:
        """Löscht alle Registrierungen (für Tests)."""
        cls._agents.clear()

    @classmethod
    def register_builtins(cls) -> None:
        """Registriert alle Builtin-Agents neu."""
        import importlib
        # Reload each submodule first
        import scio.agents.builtin.data_loader as data_loader
        import scio.agents.builtin.analyzer as analyzer
        import scio.agents.builtin.reporter as reporter
        import scio.agents.builtin.llm_agent as llm_agent
        import scio.agents.builtin.transformer as transformer
        import scio.agents.builtin.python_expert as python_expert
        importlib.reload(data_loader)
        importlib.reload(analyzer)
        importlib.reload(reporter)
        importlib.reload(llm_agent)
        importlib.reload(transformer)
        importlib.reload(python_expert)
        # Then reload the main module
        import scio.agents.builtin as builtin_module
        importlib.reload(builtin_module)


def register_agent(agent_type: str):
    """
    Decorator zum Registrieren von Agenten.

    Beispiel:
        @register_agent("data_loader")
        class DataLoaderAgent(Agent):
            ...
    """

    def decorator(cls: Type[Agent]) -> Type[Agent]:
        cls.agent_type = agent_type
        AgentRegistry.register(agent_type, cls)
        return cls

    return decorator
