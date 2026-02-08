"""SCIO Tools - Werkzeuge für Agenten."""

from scio.tools.base import Tool, ToolResult, ToolConfig
from scio.tools.registry import ToolRegistry, register_tool

# Import builtin tools to trigger registration
from scio.tools.builtin import (
    FileReaderTool,
    FileWriterTool,
    HttpClientTool,
    PythonExecutorTool,
    ShellTool,
    MathTool,
)

__all__ = [
    "Tool",
    "ToolResult",
    "ToolConfig",
    "ToolRegistry",
    "register_tool",
    "FileReaderTool",
    "FileWriterTool",
    "HttpClientTool",
    "PythonExecutorTool",
    "ShellTool",
    "MathTool",
]
