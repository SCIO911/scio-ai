"""Tests für das Agenten-System."""

import pytest
from scio.agents.base import Agent, AgentConfig, AgentContext, AgentResult, AgentState
from scio.agents.registry import AgentRegistry, register_agent
from scio.core.utils import generate_id


class TestAgentBase:
    """Tests für die Agent-Basisklasse."""

    def test_agent_config(self):
        """Testet AgentConfig."""
        config = AgentConfig(name="test")
        assert config.name == "test"
        assert config.max_iterations == 100

    def test_agent_context(self):
        """Testet AgentContext."""
        ctx = AgentContext(
            agent_id="agent-1",
            execution_id="exec-1",
            parameters={"key": "value"},
        )
        assert ctx.agent_id == "agent-1"
        assert ctx.parameters["key"] == "value"


class TestAgentRegistry:
    """Tests für die Agent-Registry."""

    def setup_method(self):
        """Räumt Registry vor jedem Test auf."""
        AgentRegistry.clear()

    def test_register_and_get(self):
        """Testet Registrierung und Abruf."""

        @register_agent("test_agent")
        class TestAgent(Agent):
            async def execute(self, input_data, context):
                return {"result": "ok"}

        assert "test_agent" in AgentRegistry.list_types()
        agent_class = AgentRegistry.get("test_agent")
        assert agent_class == TestAgent

    def test_create_instance(self):
        """Testet Instanz-Erstellung."""

        @register_agent("simple_agent")
        class SimpleAgent(Agent):
            async def execute(self, input_data, context):
                return input_data

        agent = AgentRegistry.create("simple_agent", {"name": "Test"})
        assert agent is not None
        assert agent.config.name == "Test"

    def test_unknown_agent_raises(self):
        """Testet Fehler bei unbekanntem Agent."""
        from scio.core.exceptions import AgentError

        with pytest.raises(AgentError):
            AgentRegistry.get("nonexistent")


class TestBuiltinAgents:
    """Tests für Builtin Agents."""

    def test_data_loader_import(self):
        """Testet Import des DataLoader."""
        from scio.agents.builtin import DataLoaderAgent
        assert DataLoaderAgent is not None

    def test_analyzer_import(self):
        """Testet Import des Analyzers."""
        from scio.agents.builtin import AnalyzerAgent
        assert AnalyzerAgent is not None

    def test_reporter_import(self):
        """Testet Import des Reporters."""
        from scio.agents.builtin import ReporterAgent
        assert ReporterAgent is not None

    def test_transformer_import(self):
        """Testet Import des Transformers."""
        from scio.agents.builtin import TransformerAgent
        assert TransformerAgent is not None
