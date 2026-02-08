"""Tests für das Tool-System."""

import pytest
from scio.tools.base import Tool, ToolConfig, ToolResult
from scio.tools.registry import ToolRegistry, register_tool


class TestToolBase:
    """Tests für die Tool-Basisklasse."""

    def test_tool_config(self):
        """Testet ToolConfig."""
        config = ToolConfig(name="test_tool")
        assert config.name == "test_tool"
        assert config.timeout_seconds == 60

    def test_tool_result(self):
        """Testet ToolResult."""
        result = ToolResult(success=True, output={"key": "value"})
        assert result.success is True
        assert result.output["key"] == "value"

        result_dict = result.to_dict()
        assert "success" in result_dict


class TestToolRegistry:
    """Tests für die Tool-Registry."""

    def setup_method(self):
        """Räumt Registry vor jedem Test auf."""
        ToolRegistry.clear()

    def test_register_and_get(self):
        """Testet Registrierung und Abruf."""

        @register_tool("test_tool")
        class TestTool(Tool):
            async def execute(self, input_data):
                return {"result": "ok"}

        assert "test_tool" in ToolRegistry.list_tools()
        tool_class = ToolRegistry.get("test_tool")
        assert tool_class == TestTool

    def test_create_instance(self):
        """Testet Instanz-Erstellung."""

        @register_tool("simple_tool")
        class SimpleTool(Tool):
            async def execute(self, input_data):
                return input_data

        tool = ToolRegistry.create("simple_tool")
        assert tool is not None

    def test_get_schemas(self):
        """Testet Schema-Abruf."""

        @register_tool("schema_tool")
        class SchemaTool(Tool):
            async def execute(self, input_data):
                return {}

            def get_schema(self):
                return {"name": "schema_tool", "description": "Test"}

        schemas = ToolRegistry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "schema_tool"


class TestBuiltinTools:
    """Tests für Builtin Tools."""

    def test_file_reader_import(self):
        """Testet Import des FileReaders."""
        from scio.tools.builtin import FileReaderTool
        assert FileReaderTool is not None

    def test_file_writer_import(self):
        """Testet Import des FileWriters."""
        from scio.tools.builtin import FileWriterTool
        assert FileWriterTool is not None

    def test_http_client_import(self):
        """Testet Import des HttpClients."""
        from scio.tools.builtin import HttpClientTool
        assert HttpClientTool is not None

    def test_python_executor_import(self):
        """Testet Import des PythonExecutors."""
        from scio.tools.builtin import PythonExecutorTool
        assert PythonExecutorTool is not None

    def test_math_tool_import(self):
        """Testet Import des MathTools."""
        from scio.tools.builtin import MathTool
        assert MathTool is not None


@pytest.mark.asyncio
class TestMathTool:
    """Async Tests für das Math Tool."""

    async def test_add(self):
        """Testet Addition."""
        from scio.tools.builtin import MathTool

        tool = MathTool()
        result = await tool.execute({"operation": "add", "a": 5, "b": 3})
        assert result["result"] == 8

    async def test_sqrt(self):
        """Testet Quadratwurzel."""
        from scio.tools.builtin import MathTool

        tool = MathTool()
        result = await tool.execute({"operation": "sqrt", "a": 16})
        assert result["result"] == 4.0

    async def test_constant(self):
        """Testet Konstanten."""
        from scio.tools.builtin import MathTool
        import math

        tool = MathTool()
        result = await tool.execute({"operation": "constant", "name": "pi"})
        assert result["result"] == math.pi


@pytest.mark.asyncio
class TestPythonExecutor:
    """Async Tests für den Python Executor."""

    async def test_simple_code(self):
        """Testet einfache Code-Ausführung."""
        from scio.tools.builtin import PythonExecutorTool

        tool = PythonExecutorTool()
        result = await tool.execute({"code": "result = 2 + 2"})
        assert result["result"] == 4

    async def test_with_variables(self):
        """Testet Code mit Variablen."""
        from scio.tools.builtin import PythonExecutorTool

        tool = PythonExecutorTool()
        result = await tool.execute({
            "code": "result = x * 2",
            "variables": {"x": 5},
        })
        assert result["result"] == 10

    async def test_blocked_code(self):
        """Testet Blockierung gefährlicher Codes."""
        from scio.tools.builtin import PythonExecutorTool
        from scio.core.exceptions import SecurityError

        tool = PythonExecutorTool()

        with pytest.raises(SecurityError):
            await tool.execute({"code": "eval('1+1')"})
