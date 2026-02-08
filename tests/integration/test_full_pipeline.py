"""Integration Tests für die vollständige Pipeline."""

import pytest
import json
from pathlib import Path

from scio.parser import parse_experiment, validate_experiment
from scio.validation import ValidationChain, ScientificValidator, SecurityValidator


class TestFullPipeline:
    """Integration Tests für vollständige Workflows."""

    def test_parse_validate_experiment(self, temp_yaml_file):
        """Testet Parsen und Validieren."""
        experiment = parse_experiment(temp_yaml_file)

        assert experiment.name == "test_experiment"
        assert len(experiment.steps) == 1

        result = validate_experiment(experiment)
        assert result.is_valid

    def test_validation_chain(self, sample_experiment):
        """Testet die komplette Validierungs-Chain."""
        chain = ValidationChain([
            ScientificValidator(),
            SecurityValidator(),
        ])

        report = chain.validate(sample_experiment)

        # Report sollte Metadaten von beiden Validatoren haben
        assert report is not None

    def test_execution_order(self):
        """Testet Berechnung der Ausführungsreihenfolge."""
        from scio.parser.schema import ExperimentSchema

        experiment = ExperimentSchema(
            name="order_test",
            steps=[
                {"id": "c", "type": "tool", "depends_on": ["a", "b"]},
                {"id": "a", "type": "tool"},
                {"id": "b", "type": "tool", "depends_on": ["a"]},
            ],
        )

        order = experiment.get_execution_order()

        # a muss vor b und c kommen
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        # b muss vor c kommen
        assert order.index("b") < order.index("c")


@pytest.mark.asyncio
class TestToolExecution:
    """Integration Tests für Tool-Ausführung."""

    async def test_math_chain(self):
        """Testet verkettete Math-Operationen."""
        from scio.tools.builtin import MathTool

        tool = MathTool()

        # Berechne (5 + 3) * 2
        r1 = await tool.execute({"operation": "add", "a": 5, "b": 3})
        r2 = await tool.execute({"operation": "multiply", "a": r1["result"], "b": 2})

        assert r2["result"] == 16

    async def test_file_roundtrip(self, tmp_path):
        """Testet Schreiben und Lesen einer Datei."""
        from scio.tools.builtin import FileWriterTool, FileReaderTool

        test_file = tmp_path / "test.json"
        test_data = {"name": "test", "value": 42}

        # Schreiben
        writer = FileWriterTool({"name": "writer", "overwrite": True})
        write_result = await writer.execute({
            "path": str(test_file),
            "content": test_data,
        })
        assert test_file.exists()

        # Lesen
        reader = FileReaderTool()
        read_result = await reader.execute({"path": str(test_file)})

        assert read_result["content"] == test_data


class TestMemoryIntegration:
    """Integration Tests für Memory-System."""

    def test_context_with_store(self):
        """Testet Context mit MemoryStore."""
        from scio.memory import ContextManager, MemoryStore

        store = MemoryStore()
        manager = ContextManager(store)

        ctx = manager.create_context("experiment1", {"seed": 42})
        ctx.set_variable("iteration", 1)

        # Checkpoint speichert den aktuellen Zustand
        cp_id = manager.save_checkpoint(ctx)

        # Modifiziere den aktiven Context
        ctx.set_variable("iteration", 10)

        # Verifiziere, dass der Checkpoint den urspruenglichen Wert hat
        key = f"checkpoint:{ctx.execution_id}:{cp_id}"
        stored_data = store.get(key)
        assert stored_data is not None
        assert stored_data["variables"]["iteration"] == 1

        # Der aktive Context hat den modifizierten Wert
        assert ctx.get_variable("iteration") == 10


class TestPluginSystem:
    """Integration Tests für Plugin-System."""

    def test_plugin_registration(self):
        """Testet Plugin-Registrierung."""
        from scio.plugins import Plugin, PluginMetadata
        from scio.agents.base import Agent, AgentConfig, AgentContext
        from scio.agents.registry import AgentRegistry

        # Cleanup
        AgentRegistry.clear()

        class TestPluginAgent(Agent):
            async def execute(self, input_data, context):
                return {"from_plugin": True}

        class TestPlugin(Plugin):
            metadata = PluginMetadata(
                name="test_plugin",
                version="1.0.0",
                description="Test Plugin",
            )

            def on_load(self):
                self.register_agent("plugin_agent", TestPluginAgent)

        # Lade Plugin
        plugin = TestPlugin({})
        plugin.on_load()

        # Agent sollte registriert sein
        assert "plugin_agent" in AgentRegistry.list_types()
