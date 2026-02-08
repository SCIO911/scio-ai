#!/usr/bin/env python3
"""
SCIO Comprehensive Test Suite
Testet alle Komponenten auf Herz und Nieren.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Farbige Ausgabe
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def ok(msg):
    print(f"  {Colors.GREEN}[OK]{Colors.END} {msg}")

def fail(msg):
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")

def warn(msg):
    print(f"  {Colors.YELLOW}[WARN]{Colors.END} {msg}")

def section(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== {msg} ==={Colors.END}")

def test_result(name, passed, total):
    color = Colors.GREEN if passed == total else Colors.RED
    print(f"\n{color}{name}: {passed}/{total} Tests bestanden{Colors.END}")
    return passed == total


# ============================================================
# 1. CORE TESTS
# ============================================================

def test_core():
    section("1. CORE MODULE")
    passed = 0
    total = 0

    # Config Test
    total += 1
    try:
        from scio.core.config import Config, get_config, set_config
        config = Config(environment="testing", debug=True)
        assert config.environment == "testing"
        assert config.debug == True
        set_config(config)
        assert get_config().debug == True
        ok("Config: Erstellung und Zugriff")
        passed += 1
    except Exception as e:
        fail(f"Config: {e}")

    # Exceptions Test
    total += 1
    try:
        from scio.core.exceptions import (
            SCIOError, ValidationError, ExecutionError,
            SecurityError, AgentError, PluginError
        )
        err = ValidationError("Test", field="test_field")
        assert err.code == "VALIDATION_ERROR"
        assert err.details["field"] == "test_field"
        ok("Exceptions: Hierarchie und Details")
        passed += 1
    except Exception as e:
        fail(f"Exceptions: {e}")

    # Logging Test
    total += 1
    try:
        from scio.core.logging import get_logger, setup_logging
        setup_logging(level="WARNING")
        logger = get_logger("test", component="test_component")
        logger.warning("Test message")
        ok("Logging: Setup und Logger-Erstellung")
        passed += 1
    except Exception as e:
        fail(f"Logging: {e}")

    # Utils Test
    total += 1
    try:
        from scio.core.utils import generate_id, hash_content, deep_merge, safe_get

        id1 = generate_id("test")
        id2 = generate_id("test")
        assert id1 != id2
        assert id1.startswith("test-")

        h = hash_content("hello")
        assert len(h) == 64  # SHA256

        merged = deep_merge({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}})
        assert merged["b"]["c"] == 2
        assert merged["b"]["d"] == 3

        data = {"level1": {"level2": {"key": "value"}}}
        assert safe_get(data, "level1", "level2", "key") == "value"
        assert safe_get(data, "missing", default="default") == "default"

        ok("Utils: ID-Generierung, Hashing, Merge, SafeGet")
        passed += 1
    except Exception as e:
        fail(f"Utils: {e}")

    return test_result("CORE", passed, total)


# ============================================================
# 2. PARSER TESTS
# ============================================================

def test_parser():
    section("2. PARSER MODULE")
    passed = 0
    total = 0

    # YAML Parser Test
    total += 1
    try:
        from scio.parser.yaml_parser import YAMLParser
        parser = YAMLParser()

        yaml_content = """
name: test_experiment
version: "1.0"
steps:
  - id: step1
    type: tool
"""
        data = parser.parse_string(yaml_content)
        assert data["name"] == "test_experiment"
        assert len(data["steps"]) == 1
        ok("YAML Parser: String-Parsing")
        passed += 1
    except Exception as e:
        fail(f"YAML Parser: {e}")

    # Schema Validation Test
    total += 1
    try:
        from scio.parser.schema import ExperimentSchema, StepSchema, AgentSchema

        exp = ExperimentSchema(
            name="schema_test",
            version="1.0",
            agents=[
                {"id": "agent1", "type": "test", "name": "Test Agent"}
            ],
            steps=[
                {"id": "step1", "type": "agent", "agent": "agent1"}
            ]
        )
        assert exp.name == "schema_test"
        assert len(exp.steps) == 1
        assert exp.steps[0].agent == "agent1"
        ok("Schema: ExperimentSchema Validierung")
        passed += 1
    except Exception as e:
        fail(f"Schema: {e}")

    # Execution Order Test
    total += 1
    try:
        from scio.parser.schema import ExperimentSchema

        exp = ExperimentSchema(
            name="order_test",
            steps=[
                {"id": "c", "type": "tool", "tool": "math", "depends_on": ["a", "b"]},
                {"id": "a", "type": "tool", "tool": "math"},
                {"id": "b", "type": "tool", "tool": "math", "depends_on": ["a"]},
            ]
        )
        order = exp.get_execution_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")
        ok("Schema: Execution Order (Topologische Sortierung)")
        passed += 1
    except Exception as e:
        fail(f"Execution Order: {e}")

    # Circular Dependency Detection
    total += 1
    try:
        from scio.parser.schema import ExperimentSchema

        exp = ExperimentSchema(
            name="circular_test",
            steps=[
                {"id": "a", "type": "tool", "tool": "math", "depends_on": ["b"]},
                {"id": "b", "type": "tool", "tool": "math", "depends_on": ["a"]},
            ]
        )
        try:
            exp.get_execution_order()
            fail("Circular Dependency: Sollte Fehler werfen")
        except ValueError as e:
            if "Zyklische" in str(e):
                ok("Schema: Zyklische Abhängigkeiten erkannt")
                passed += 1
            else:
                fail(f"Circular Dependency: Falscher Fehler: {e}")
    except Exception as e:
        fail(f"Circular Dependency: {e}")

    # Invalid Agent Reference
    total += 1
    try:
        from scio.parser.schema import ExperimentSchema

        try:
            ExperimentSchema(
                name="invalid_ref",
                agents=[],
                steps=[{"id": "s1", "type": "agent", "agent": "nonexistent"}]
            )
            fail("Invalid Reference: Sollte Fehler werfen")
        except ValueError:
            ok("Schema: Ungültige Agent-Referenz erkannt")
            passed += 1
    except Exception as e:
        fail(f"Invalid Reference: {e}")

    return test_result("PARSER", passed, total)


# ============================================================
# 3. VALIDATION TESTS
# ============================================================

def test_validation():
    section("3. VALIDATION MODULE")
    passed = 0
    total = 0

    # ValidationReport Test
    total += 1
    try:
        from scio.validation.base import ValidationReport, Severity

        report = ValidationReport()
        assert report.is_valid

        report.add_warning("Test warning", "WARN_001")
        assert report.is_valid
        assert report.has_warnings

        report.add_error("Test error", "ERR_001")
        assert not report.is_valid
        assert report.error_count == 1

        ok("ValidationReport: Errors und Warnings")
        passed += 1
    except Exception as e:
        fail(f"ValidationReport: {e}")

    # Scientific Validator Test
    total += 1
    try:
        from scio.validation.scientific import ScientificValidator
        from scio.parser.schema import ExperimentSchema

        validator = ScientificValidator()

        # Experiment ohne Autor
        exp = ExperimentSchema(
            name="test",
            steps=[{"id": "s1", "type": "tool", "tool": "math"}]
        )
        report = validator.validate(exp)

        warning_codes = [i.code for i in report.issues]
        assert "SCI_NO_AUTHOR" in warning_codes
        ok("ScientificValidator: Fehlender Autor erkannt")
        passed += 1
    except Exception as e:
        fail(f"ScientificValidator: {e}")

    # Security Validator Test
    total += 1
    try:
        from scio.validation.security import SecurityValidator
        from scio.parser.schema import ExperimentSchema

        validator = SecurityValidator()

        # Experiment mit gefährlichem Code
        exp = ExperimentSchema(
            name="dangerous",
            steps=[{
                "id": "s1",
                "type": "tool",
                "tool": "python_executor",
                "inputs": {"cmd": "eval(user_input)"}
            }]
        )
        report = validator.validate(exp)

        error_codes = [i.code for i in report.issues]
        assert "SEC_DANGEROUS_PATTERN" in error_codes
        ok("SecurityValidator: Gefährlicher Code erkannt")
        passed += 1
    except Exception as e:
        fail(f"SecurityValidator: {e}")

    # ValidationChain Test
    total += 1
    try:
        from scio.validation import ValidationChain, ScientificValidator, SecurityValidator
        from scio.parser.schema import ExperimentSchema

        chain = ValidationChain([
            ScientificValidator(),
            SecurityValidator(),
        ])

        exp = ExperimentSchema(
            name="chain_test",
            steps=[{"id": "s1", "type": "tool", "tool": "math"}]
        )
        report = chain.validate(exp)

        # Beide Validatoren sollten gelaufen sein
        assert report is not None
        ok("ValidationChain: Mehrere Validatoren")
        passed += 1
    except Exception as e:
        fail(f"ValidationChain: {e}")

    return test_result("VALIDATION", passed, total)


# ============================================================
# 4. AGENT TESTS
# ============================================================

def test_agents():
    section("4. AGENT SYSTEM")
    passed = 0
    total = 0

    # Agent Registry Test
    total += 1
    try:
        from scio.agents.registry import AgentRegistry, register_agent
        from scio.agents.base import Agent

        AgentRegistry.clear()

        @register_agent("test_agent_123")
        class TestAgent(Agent):
            async def execute(self, input_data, context):
                return {"result": input_data.get("value", 0) * 2}

        assert "test_agent_123" in AgentRegistry.list_types()
        agent = AgentRegistry.create("test_agent_123", {"name": "Test"})
        assert agent is not None
        ok("AgentRegistry: Registrierung und Erstellung")
        passed += 1
    except Exception as e:
        fail(f"AgentRegistry: {e}")

    # Builtin Agents Import Test
    total += 1
    try:
        from scio.agents.builtin import (
            DataLoaderAgent,
            AnalyzerAgent,
            ReporterAgent,
            LLMAgent,
            TransformerAgent,
        )

        # Alle sollten importierbar sein
        assert DataLoaderAgent is not None
        assert AnalyzerAgent is not None
        assert ReporterAgent is not None
        assert LLMAgent is not None
        assert TransformerAgent is not None
        ok("Builtin Agents: Alle 5 importierbar")
        passed += 1
    except Exception as e:
        fail(f"Builtin Agents Import: {e}")

    # DataLoader Agent Test
    total += 1
    try:
        from scio.agents.builtin import DataLoaderAgent

        agent = DataLoaderAgent({"name": "loader"})

        # Erstelle Test-JSON-Datei
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "data", "numbers": [1, 2, 3]}, f)
            temp_path = f.name

        async def test_load():
            from scio.agents.base import AgentContext
            ctx = AgentContext(agent_id="test", execution_id="test")
            result = await agent.execute({"path": temp_path}, ctx)
            return result

        result = asyncio.run(test_load())
        assert result["data"]["test"] == "data"
        Path(temp_path).unlink()
        ok("DataLoaderAgent: JSON-Datei geladen")
        passed += 1
    except Exception as e:
        fail(f"DataLoaderAgent: {e}")

    # Analyzer Agent Test
    total += 1
    try:
        from scio.agents.builtin import AnalyzerAgent

        agent = AnalyzerAgent({
            "name": "analyzer",
            "methods": ["mean", "min", "max", "count"]
        })

        async def test_analyze():
            from scio.agents.base import AgentContext
            ctx = AgentContext(agent_id="test", execution_id="test")
            result = await agent.execute({
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }, ctx)
            return result

        result = asyncio.run(test_analyze())
        stats = result["statistics"]
        assert stats["mean"] == 5.5
        assert stats["min"] == 1
        assert stats["max"] == 10
        assert stats["count"] == 10
        ok("AnalyzerAgent: Statistiken berechnet")
        passed += 1
    except Exception as e:
        fail(f"AnalyzerAgent: {e}")

    # Transformer Agent Test
    total += 1
    try:
        from scio.agents.builtin import TransformerAgent

        agent = TransformerAgent({"name": "transformer"})

        async def test_transform():
            from scio.agents.base import AgentContext
            ctx = AgentContext(agent_id="test", execution_id="test")
            result = await agent.execute({
                "data": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                    {"name": "Charlie", "age": 35},
                ],
                "operations": [
                    {"type": "filter", "field": "age", "condition": "gte", "value": 28},
                    {"type": "sort", "field": "age", "descending": True},
                ]
            }, ctx)
            return result

        result = asyncio.run(test_transform())
        data = result["data"]
        assert len(data) == 2  # Alice und Charlie
        assert data[0]["name"] == "Charlie"  # Ältester zuerst
        ok("TransformerAgent: Filter und Sort")
        passed += 1
    except Exception as e:
        fail(f"TransformerAgent: {e}")

    # Reporter Agent Test
    total += 1
    try:
        from scio.agents.builtin import ReporterAgent

        agent = ReporterAgent({
            "name": "reporter",
            "output_format": "markdown",
            "include_timestamp": False,
        })

        async def test_report():
            from scio.agents.base import AgentContext
            ctx = AgentContext(
                agent_id="test",
                execution_id="test",
                experiment_name="report_test"
            )
            result = await agent.execute({
                "title": "Test Report",
                "data": {"key": "value", "number": 42}
            }, ctx)
            return result

        result = asyncio.run(test_report())
        assert "# Test Report" in result["report"]
        assert result["format"] == "markdown"
        ok("ReporterAgent: Markdown-Report generiert")
        passed += 1
    except Exception as e:
        fail(f"ReporterAgent: {e}")

    return test_result("AGENTS", passed, total)


# ============================================================
# 5. TOOL TESTS
# ============================================================

def test_tools():
    section("5. TOOL SYSTEM")
    passed = 0
    total = 0

    # Tool Registry Test
    total += 1
    try:
        from scio.tools.registry import ToolRegistry, register_tool
        from scio.tools.base import Tool

        ToolRegistry.clear()

        @register_tool("test_tool_123")
        class TestTool(Tool):
            async def execute(self, input_data):
                return {"doubled": input_data.get("value", 0) * 2}

        assert "test_tool_123" in ToolRegistry.list_tools()
        tool = ToolRegistry.create("test_tool_123")
        assert tool is not None
        ok("ToolRegistry: Registrierung und Erstellung")
        passed += 1
    except Exception as e:
        fail(f"ToolRegistry: {e}")

    # Builtin Tools Import Test
    total += 1
    try:
        from scio.tools.builtin import (
            FileReaderTool,
            FileWriterTool,
            HttpClientTool,
            PythonExecutorTool,
            ShellTool,
            MathTool,
        )

        assert FileReaderTool is not None
        assert FileWriterTool is not None
        assert HttpClientTool is not None
        assert PythonExecutorTool is not None
        assert ShellTool is not None
        assert MathTool is not None
        ok("Builtin Tools: Alle 6 importierbar")
        passed += 1
    except Exception as e:
        fail(f"Builtin Tools Import: {e}")

    # Math Tool Test
    total += 1
    try:
        from scio.tools.builtin import MathTool
        import math

        tool = MathTool()

        async def test_math():
            results = []
            results.append(await tool.execute({"operation": "add", "a": 5, "b": 3}))
            results.append(await tool.execute({"operation": "multiply", "a": 4, "b": 7}))
            results.append(await tool.execute({"operation": "sqrt", "a": 16}))
            results.append(await tool.execute({"operation": "power", "a": 2, "b": 10}))
            results.append(await tool.execute({"operation": "constant", "name": "pi"}))
            return results

        results = asyncio.run(test_math())
        assert results[0]["result"] == 8
        assert results[1]["result"] == 28
        assert results[2]["result"] == 4.0
        assert results[3]["result"] == 1024
        assert abs(results[4]["result"] - math.pi) < 0.0001
        ok("MathTool: Alle Operationen korrekt")
        passed += 1
    except Exception as e:
        fail(f"MathTool: {e}")

    # Python Executor Test
    total += 1
    try:
        from scio.tools.builtin import PythonExecutorTool

        tool = PythonExecutorTool()

        async def test_python():
            # Einfache Berechnung
            r1 = await tool.execute({"code": "result = sum(range(1, 11))"})

            # Mit Variablen
            r2 = await tool.execute({
                "code": "result = [x**2 for x in numbers]",
                "variables": {"numbers": [1, 2, 3, 4, 5]}
            })

            # Mit Import (erlaubt)
            r3 = await tool.execute({
                "code": "import math\nresult = math.factorial(5)"
            })

            return r1, r2, r3

        r1, r2, r3 = asyncio.run(test_python())
        assert r1["result"] == 55
        assert r2["result"] == [1, 4, 9, 16, 25]
        assert r3["result"] == 120
        ok("PythonExecutorTool: Code-Ausführung")
        passed += 1
    except Exception as e:
        fail(f"PythonExecutorTool: {e}")

    # Python Executor Security Test
    total += 1
    try:
        from scio.tools.builtin import PythonExecutorTool
        from scio.core.exceptions import SecurityError

        tool = PythonExecutorTool()

        dangerous_codes = [
            "eval('1+1')",
            "exec('x=1')",
            "__import__('os')",
            "open('/etc/passwd')",
        ]

        async def test_security():
            blocked = 0
            for code in dangerous_codes:
                try:
                    await tool.execute({"code": code})
                except SecurityError:
                    blocked += 1
            return blocked

        blocked = asyncio.run(test_security())
        assert blocked == len(dangerous_codes)
        ok(f"PythonExecutorTool: {blocked}/{len(dangerous_codes)} gefährliche Codes blockiert")
        passed += 1
    except Exception as e:
        fail(f"PythonExecutorTool Security: {e}")

    # File Tools Test
    total += 1
    try:
        from scio.tools.builtin import FileWriterTool, FileReaderTool

        async def test_files():
            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "test.json"
                test_data = {"message": "Hello SCIO", "numbers": [1, 2, 3]}

                writer = FileWriterTool({"name": "writer", "overwrite": True})
                reader = FileReaderTool()

                # Schreiben
                write_result = await writer.execute({
                    "path": str(test_file),
                    "content": test_data
                })

                # Lesen
                read_result = await reader.execute({"path": str(test_file)})

                return write_result, read_result, test_data

        w, r, original = asyncio.run(test_files())
        assert r["content"] == original
        ok("FileTools: Schreiben und Lesen")
        passed += 1
    except Exception as e:
        fail(f"FileTools: {e}")

    return test_result("TOOLS", passed, total)


# ============================================================
# 6. MEMORY TESTS
# ============================================================

def test_memory():
    section("6. MEMORY SYSTEM")
    passed = 0
    total = 0

    # MemoryStore Test
    total += 1
    try:
        from scio.memory.store import MemoryStore

        store = MemoryStore()

        # Basis-Operationen
        store.set("key1", "value1")
        store.set("key2", {"nested": "data"})
        store.set("tagged", "data", tags=["important", "test"])

        assert store.get("key1") == "value1"
        assert store.get("key2")["nested"] == "data"
        assert store.exists("key1")
        assert not store.exists("nonexistent")

        # Tags
        tagged = store.find_by_tag("important")
        assert len(tagged) == 1

        # Delete
        store.delete("key1")
        assert not store.exists("key1")

        ok("MemoryStore: CRUD und Tags")
        passed += 1
    except Exception as e:
        fail(f"MemoryStore: {e}")

    # ExecutionContext Test
    total += 1
    try:
        from scio.memory.context import ExecutionContext

        ctx = ExecutionContext(
            execution_id="exec-123",
            experiment_name="test_exp",
            parameters={"seed": 42, "learning_rate": 0.01}
        )

        # Variablen
        ctx.set_variable("iteration", 1)
        ctx.set_variable("loss", 0.5)

        assert ctx.get_variable("iteration") == 1
        assert ctx.get_variable("seed") == 42  # Fallback zu Parameter
        assert ctx.get_variable("missing", "default") == "default"

        # Step Outputs
        ctx.set_step_output("step1", {"result": "ok", "metrics": {"acc": 0.95}})
        assert ctx.get_step_output("step1", "result") == "ok"

        ok("ExecutionContext: Variablen und Outputs")
        passed += 1
    except Exception as e:
        fail(f"ExecutionContext: {e}")

    # Reference Resolution Test
    total += 1
    try:
        from scio.memory.context import ExecutionContext

        ctx = ExecutionContext(
            execution_id="exec-123",
            experiment_name="test",
            parameters={"param1": "value1"}
        )
        ctx.set_variable("var1", "var_value")
        ctx.set_step_output("load", {"data": [1, 2, 3]})

        # Parameter-Referenz
        assert ctx.resolve_reference("${param1}") == "value1"

        # Variablen-Referenz
        assert ctx.resolve_reference("${var.var1}") == "var_value"

        # Step-Output-Referenz
        assert ctx.resolve_reference("${step.load.data}") == [1, 2, 3]

        # Kein Referenz-String
        assert ctx.resolve_reference("plain_string") == "plain_string"

        ok("ExecutionContext: Referenz-Auflösung")
        passed += 1
    except Exception as e:
        fail(f"Reference Resolution: {e}")

    # ContextManager Checkpoint Test
    total += 1
    try:
        from scio.memory.context import ContextManager

        manager = ContextManager()
        ctx = manager.create_context("checkpoint_test", {"x": 100})
        ctx.set_variable("progress", 25)

        # Checkpoint speichern
        cp_id = manager.save_checkpoint(ctx)

        # Modifizieren
        ctx.set_variable("progress", 75)

        # Checkpoint verifizieren
        key = f"checkpoint:{ctx.execution_id}:{cp_id}"
        stored = manager.memory.get(key)
        assert stored["variables"]["progress"] == 25  # Ursprünglicher Wert

        ok("ContextManager: Checkpoint funktioniert")
        passed += 1
    except Exception as e:
        fail(f"ContextManager Checkpoint: {e}")

    return test_result("MEMORY", passed, total)


# ============================================================
# 7. EXECUTION TESTS
# ============================================================

def test_execution():
    section("7. EXECUTION ENGINE")
    passed = 0
    total = 0

    # Sandbox Test
    total += 1
    try:
        from scio.execution.sandbox import Sandbox, SandboxConfig
        from scio.core.exceptions import SecurityError

        config = SandboxConfig(
            enabled=True,
            blocked_modules=["os.system", "subprocess"],
            network_enabled=False,
        )
        sandbox = Sandbox(config)

        # Module blockiert
        blocked = 0
        for mod in ["subprocess", "os.system"]:
            try:
                sandbox.check_module_import(mod)
            except SecurityError:
                blocked += 1

        # Netzwerk blockiert
        try:
            sandbox.check_network_access("example.com", 80)
        except SecurityError:
            blocked += 1

        assert blocked == 3
        ok("Sandbox: Module und Netzwerk blockiert")
        passed += 1
    except Exception as e:
        fail(f"Sandbox: {e}")

    # Sandbox Restricted Globals Test
    total += 1
    try:
        from scio.execution.sandbox import Sandbox

        sandbox = Sandbox()
        globals_dict = sandbox.create_restricted_globals()
        builtins = globals_dict["__builtins__"]

        # Sichere Funktionen vorhanden
        assert "len" in builtins
        assert "print" in builtins
        assert "range" in builtins

        # Gefährliche Funktionen fehlen
        assert "eval" not in builtins
        assert "exec" not in builtins
        assert "open" not in builtins
        assert "__import__" not in builtins

        ok("Sandbox: Restricted Globals")
        passed += 1
    except Exception as e:
        fail(f"Sandbox Globals: {e}")

    # Scheduler Test
    total += 1
    try:
        from scio.execution.scheduler import Scheduler, TaskPriority

        async def test_scheduler():
            scheduler = Scheduler(max_concurrent=2)
            await scheduler.start()

            results = []

            async def task(x):
                results.append(x)
                return x * 2

            await scheduler.submit(task, 1, priority=TaskPriority.LOW)
            await scheduler.submit(task, 2, priority=TaskPriority.HIGH)
            await scheduler.submit(task, 3, priority=TaskPriority.NORMAL)

            all_results = await scheduler.run_all()
            await scheduler.stop()

            return all_results, results

        all_results, execution_order = asyncio.run(test_scheduler())
        assert len(all_results) == 3
        assert execution_order[0] == 2  # HIGH priority first
        ok("Scheduler: Prioritäts-basierte Ausführung")
        passed += 1
    except Exception as e:
        fail(f"Scheduler: {e}")

    # ExecutionEngine Test
    total += 1
    try:
        from scio.execution.engine import ExecutionEngine, ExecutionStatus

        engine = ExecutionEngine()

        async def dummy_handler(step, context):
            return {"step_id": step.id, "executed": True}

        engine.register_step_handler("tool", dummy_handler)
        assert "tool" in engine._step_handlers
        ok("ExecutionEngine: Handler-Registrierung")
        passed += 1
    except Exception as e:
        fail(f"ExecutionEngine: {e}")

    return test_result("EXECUTION", passed, total)


# ============================================================
# 8. PLUGIN TESTS
# ============================================================

def test_plugins():
    section("8. PLUGIN SYSTEM")
    passed = 0
    total = 0

    # Plugin Base Test
    total += 1
    try:
        from scio.plugins import Plugin, PluginMetadata, SimplePlugin
        from scio.agents.base import Agent
        from scio.agents.registry import AgentRegistry

        AgentRegistry.clear()

        class MyTestAgent(Agent):
            async def execute(self, input_data, context):
                return {"from": "plugin"}

        class TestPlugin(Plugin):
            metadata = PluginMetadata(
                name="test_plugin",
                version="1.0.0",
                description="Test Plugin",
                author="Test"
            )

            def on_load(self):
                self.register_agent("plugin_test_agent", MyTestAgent)

        plugin = TestPlugin({})
        plugin.on_load()

        assert "plugin_test_agent" in AgentRegistry.list_types()
        ok("Plugin: Agent-Registrierung")
        passed += 1
    except Exception as e:
        fail(f"Plugin Base: {e}")

    # PluginMetadata Test
    total += 1
    try:
        from scio.plugins import PluginMetadata

        meta = PluginMetadata(
            name="my_plugin",
            version="2.0.0",
            description="My awesome plugin",
            author="Me",
            dependencies=["numpy", "pandas"]
        )

        assert meta.name == "my_plugin"
        assert meta.version == "2.0.0"
        assert len(meta.dependencies) == 2
        ok("PluginMetadata: Vollständige Metadaten")
        passed += 1
    except Exception as e:
        fail(f"PluginMetadata: {e}")

    return test_result("PLUGINS", passed, total)


# ============================================================
# 9. CLI TESTS
# ============================================================

def test_cli():
    section("9. CLI")
    passed = 0
    total = 0

    # CLI Import Test
    total += 1
    try:
        from scio.cli.main import app, validate, run, info, init, agents, tools

        assert app is not None
        ok("CLI: Alle Befehle importierbar")
        passed += 1
    except Exception as e:
        fail(f"CLI Import: {e}")

    # Typer App Test
    total += 1
    try:
        from typer.testing import CliRunner
        from scio.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "SCIO" in result.stdout
        assert "validate" in result.stdout
        assert "run" in result.stdout
        ok("CLI: Help-Ausgabe")
        passed += 1
    except Exception as e:
        fail(f"CLI Help: {e}")

    # CLI Info Test
    total += 1
    try:
        from typer.testing import CliRunner
        from scio.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "SCIO" in result.stdout
        assert "Python" in result.stdout
        ok("CLI: info Befehl")
        passed += 1
    except Exception as e:
        fail(f"CLI Info: {e}")

    # CLI Validate Test
    total += 1
    try:
        from typer.testing import CliRunner
        from scio.cli.main import app

        runner = CliRunner()

        # Erstelle temporäre Experiment-Datei
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
name: cli_test
version: "1.0"
metadata:
  author: Test
steps:
  - id: step1
    type: tool
    tool: math
""")
            temp_path = f.name

        result = runner.invoke(app, ["validate", temp_path])
        Path(temp_path).unlink()

        assert result.exit_code == 0
        # Check for "OK" without ANSI codes interference
        assert "OK" in result.stdout and "gueltig" in result.stdout
        ok("CLI: validate Befehl")
        passed += 1
    except Exception as e:
        fail(f"CLI Validate: {e}")

    return test_result("CLI", passed, total)


# ============================================================
# 10. API TESTS
# ============================================================

def test_api():
    section("10. REST API")
    passed = 0
    total = 0

    # API Import Test
    total += 1
    try:
        from scio.api import create_app, app, router

        assert app is not None
        assert router is not None
        ok("API: Import erfolgreich")
        passed += 1
    except Exception as e:
        fail(f"API Import: {e}")

    # FastAPI Client Test
    total += 1
    try:
        from fastapi.testclient import TestClient
        from scio.api.app import app

        client = TestClient(app)

        # Health Check
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        ok("API: Health Endpoint")
        passed += 1
    except Exception as e:
        fail(f"API Health: {e}")

    # API Info Test
    total += 1
    try:
        from fastapi.testclient import TestClient
        from scio.api.app import app

        client = TestClient(app)

        response = client.get("/api/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert "scio_version" in data
        assert "python_version" in data
        ok("API: Info Endpoint")
        passed += 1
    except Exception as e:
        fail(f"API Info: {e}")

    # API Validate Test
    total += 1
    try:
        from fastapi.testclient import TestClient
        from scio.api.app import app

        client = TestClient(app)

        response = client.post("/api/v1/experiments/validate", json={
            "experiment": {
                "name": "api_test",
                "version": "1.0",
                "steps": [{"id": "s1", "type": "tool", "tool": "math"}]
            },
            "strict": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == True
        ok("API: Validate Endpoint")
        passed += 1
    except Exception as e:
        fail(f"API Validate: {e}")

    # API Agents List Test
    total += 1
    try:
        from fastapi.testclient import TestClient
        from scio.api.app import app
        from scio.agents.registry import AgentRegistry

        # Re-register builtins (may have been cleared by earlier tests)
        AgentRegistry.register_builtins()

        client = TestClient(app)

        response = client.get("/api/v1/agents")
        assert response.status_code == 200
        agents = response.json()
        assert isinstance(agents, list)
        agent_types = [a["type"] for a in agents]
        assert "data_loader" in agent_types
        ok("API: Agents Endpoint")
        passed += 1
    except Exception as e:
        fail(f"API Agents: {e}")

    # API Tools List Test
    total += 1
    try:
        from fastapi.testclient import TestClient
        from scio.api.app import app

        client = TestClient(app)

        response = client.get("/api/v1/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        tool_names = [t["name"] for t in tools]
        assert "math" in tool_names
        ok("API: Tools Endpoint")
        passed += 1
    except Exception as e:
        fail(f"API Tools: {e}")

    # API Tool Execute Test
    total += 1
    try:
        from fastapi.testclient import TestClient
        from scio.api.app import app

        client = TestClient(app)

        response = client.post("/api/v1/tools/math/execute", json={
            "operation": "multiply",
            "a": 6,
            "b": 7
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["output"]["result"] == 42
        ok("API: Tool Execute Endpoint")
        passed += 1
    except Exception as e:
        fail(f"API Tool Execute: {e}")

    return test_result("API", passed, total)


# ============================================================
# 11. EDGE CASES & ERROR HANDLING
# ============================================================

def test_edge_cases():
    section("11. EDGE CASES & ERROR HANDLING")
    passed = 0
    total = 0

    # Empty Experiment
    total += 1
    try:
        from scio.parser.schema import ExperimentSchema

        try:
            ExperimentSchema(name="empty", steps=[])
            fail("Empty Steps: Sollte Fehler werfen")
        except Exception:
            ok("Edge Case: Leere Steps abgelehnt")
            passed += 1
    except Exception as e:
        fail(f"Empty Steps: {e}")

    # Invalid Version Format
    total += 1
    try:
        from scio.parser.schema import ExperimentSchema

        try:
            ExperimentSchema(
                name="invalid_version",
                version="not-a-version",
                steps=[{"id": "s1", "type": "tool", "tool": "math"}]
            )
            fail("Invalid Version: Sollte Fehler werfen")
        except Exception:
            ok("Edge Case: Ungültige Version abgelehnt")
            passed += 1
    except Exception as e:
        fail(f"Invalid Version: {e}")

    # Invalid Step ID Format
    total += 1
    try:
        from scio.parser.schema import ExperimentSchema

        try:
            ExperimentSchema(
                name="invalid_id",
                steps=[{"id": "123-invalid", "type": "tool", "tool": "math"}]  # Muss mit Buchstabe beginnen
            )
            fail("Invalid ID: Sollte Fehler werfen")
        except Exception:
            ok("Edge Case: Ungültige Step-ID abgelehnt")
            passed += 1
    except Exception as e:
        fail(f"Invalid ID: {e}")

    # Unknown Agent Type
    total += 1
    try:
        from scio.agents.registry import AgentRegistry
        from scio.core.exceptions import AgentError

        AgentRegistry.clear()

        try:
            AgentRegistry.get("completely_unknown_agent_xyz")
            fail("Unknown Agent: Sollte Fehler werfen")
        except AgentError:
            ok("Edge Case: Unbekannter Agent abgelehnt")
            passed += 1
    except Exception as e:
        fail(f"Unknown Agent: {e}")

    # Unknown Tool Type
    total += 1
    try:
        from scio.tools.registry import ToolRegistry
        from scio.core.exceptions import PluginError

        ToolRegistry.clear()

        try:
            ToolRegistry.get("completely_unknown_tool_xyz")
            fail("Unknown Tool: Sollte Fehler werfen")
        except PluginError:
            ok("Edge Case: Unbekanntes Tool abgelehnt")
            passed += 1
    except Exception as e:
        fail(f"Unknown Tool: {e}")

    # Memory Store - Non-existent Key
    total += 1
    try:
        from scio.memory.store import MemoryStore

        store = MemoryStore()
        result = store.get("nonexistent_key_12345", default="fallback")
        assert result == "fallback"
        ok("Edge Case: Memory Default-Wert")
        passed += 1
    except Exception as e:
        fail(f"Memory Default: {e}")

    # Large Data Handling
    total += 1
    try:
        from scio.tools.builtin import PythonExecutorTool

        tool = PythonExecutorTool()

        async def test_large():
            # Generiere große Liste
            result = await tool.execute({
                "code": "result = list(range(10000))"
            })
            return result

        result = asyncio.run(test_large())
        assert len(result["result"]) == 10000
        ok("Edge Case: Große Datenmengen")
        passed += 1
    except Exception as e:
        fail(f"Large Data: {e}")

    # Deep Nested Data
    total += 1
    try:
        from scio.core.utils import safe_get

        deep_data = {"a": {"b": {"c": {"d": {"e": "found"}}}}}
        result = safe_get(deep_data, "a", "b", "c", "d", "e")
        assert result == "found"

        missing = safe_get(deep_data, "a", "x", "y", default="not_found")
        assert missing == "not_found"
        ok("Edge Case: Tief verschachtelte Daten")
        passed += 1
    except Exception as e:
        fail(f"Deep Nested: {e}")

    return test_result("EDGE CASES", passed, total)


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  SCIO COMPREHENSIVE TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")

    all_passed = True
    total_passed = 0
    total_tests = 0

    results = []

    # Führe alle Tests aus
    test_functions = [
        test_core,
        test_parser,
        test_validation,
        test_agents,
        test_tools,
        test_memory,
        test_execution,
        test_plugins,
        test_cli,
        test_api,
        test_edge_cases,
    ]

    for test_func in test_functions:
        try:
            passed = test_func()
            results.append(passed)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n{Colors.RED}CRITICAL ERROR in {test_func.__name__}: {e}{Colors.END}")
            results.append(False)
            all_passed = False

    # Zusammenfassung
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  ZUSAMMENFASSUNG{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

    modules = [
        "Core", "Parser", "Validation", "Agents", "Tools",
        "Memory", "Execution", "Plugins", "CLI", "API", "Edge Cases"
    ]

    for i, (module, passed) in enumerate(zip(modules, results)):
        status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {i+1:2}. {module:15} [{status}]")

    print()

    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}ALLE TESTS BESTANDEN!{Colors.END}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}EINIGE TESTS FEHLGESCHLAGEN!{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
