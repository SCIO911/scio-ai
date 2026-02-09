"""
SCIO Python Executor Tool

Tool für sichere Python-Code-Ausführung.
"""

import ast
import sys
from io import StringIO
from typing import Any, Optional

from scio.core.exceptions import SecurityError
from scio.core.logging import get_logger
from scio.execution.sandbox import Sandbox, SandboxConfig
from scio.tools.base import Tool, ToolConfig
from scio.tools.registry import register_tool

logger = get_logger(__name__)


class PythonExecutorConfig(ToolConfig):
    """Konfiguration für PythonExecutor."""

    name: str = "python_executor"
    description: str = "Führt Python-Code sicher in einer Sandbox aus"
    timeout_seconds: int = 30
    max_output_length: int = 10000
    allowed_imports: list[str] = [
        "math",
        "statistics",
        "json",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "re",
        "random",
    ]
    sandbox_enabled: bool = True


@register_tool("python_executor")
class PythonExecutorTool(Tool[dict[str, Any], dict[str, Any]]):
    """Tool für sichere Python-Code-Ausführung."""

    tool_name = "python_executor"
    version = "1.0"

    # Gefährliche Konstrukte
    BLOCKED_NODES = {
        ast.Import,
        ast.ImportFrom,
        ast.AsyncFunctionDef,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.Await,
    }

    BLOCKED_NAMES = {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
        "exit",
        "quit",
    }

    def __init__(self, config: Optional[PythonExecutorConfig | dict] = None):
        if config is None:
            config = PythonExecutorConfig(name="python_executor")
        elif isinstance(config, dict):
            config = PythonExecutorConfig(**config)
        super().__init__(config)
        self.config: PythonExecutorConfig = config

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Führt Python-Code sicher aus."""
        code = input_data.get("code")
        if not code:
            raise ValueError("Kein Code angegeben")

        variables = input_data.get("variables", {})

        # Validiere Code
        self._validate_code(code)

        # Erstelle sichere Umgebung
        safe_globals = self._create_safe_globals()
        safe_locals = dict(variables)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        result = None
        error = None

        try:
            # Kompiliere und führe aus
            compiled = compile(code, "<scio>", "exec")
            exec(compiled, safe_globals, safe_locals)

            # Hole Ergebnis (letzte Expression oder 'result' Variable)
            if "result" in safe_locals:
                result = safe_locals["result"]

            output = sys.stdout.getvalue()
            if len(output) > self.config.max_output_length:
                output = output[: self.config.max_output_length] + "\n... (truncated)"

        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        finally:
            sys.stdout = old_stdout

        return {
            "result": result,
            "output": output if "output" in dir() else "",
            "error": error,
            "variables": {
                k: v
                for k, v in safe_locals.items()
                if not k.startswith("_") and k not in variables
            },
        }

    def _validate_code(self, code: str) -> None:
        """Validiert Code auf gefährliche Konstrukte."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax-Fehler: {e}")

        for node in ast.walk(tree):
            # Prüfe auf blockierte Node-Typen
            if type(node) in self.BLOCKED_NODES:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Erlaube bestimmte Imports
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    else:
                        names = [node.module] if node.module else []

                    for name in names:
                        base_module = name.split(".")[0]
                        if base_module not in self.config.allowed_imports:
                            raise SecurityError(f"Import nicht erlaubt: {name}")
                else:
                    raise SecurityError(
                        f"Nicht erlaubtes Konstrukt: {type(node).__name__}"
                    )

            # Prüfe auf blockierte Namen
            if isinstance(node, ast.Name) and node.id in self.BLOCKED_NAMES:
                raise SecurityError(f"Blockierter Name: {node.id}")

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_NAMES:
                        raise SecurityError(f"Blockierter Aufruf: {node.func.id}")

    def _create_safe_globals(self) -> dict[str, Any]:
        """Erstellt sichere globals."""
        import math
        import statistics
        import json
        import datetime
        import collections
        import itertools
        import functools
        import re
        import random

        # Pre-imported modules for import statements
        allowed_modules = {
            "math": math,
            "statistics": statistics,
            "json": json,
            "datetime": datetime,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "re": re,
            "random": random,
        }

        def safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: tuple = (), level: int = 0) -> Any:
            """Sichere Import-Funktion die nur erlaubte Module zulässt."""
            base_module = name.split(".")[0]
            if base_module not in allowed_modules:
                raise ImportError(f"Import nicht erlaubt: {name}")
            return allowed_modules.get(base_module)

        safe_builtins = {
            # Typen
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "frozenset": frozenset,
            "bytes": bytes,
            "bytearray": bytearray,
            # Funktionen
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "chr": chr,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "format": format,
            "hash": hash,
            "hex": hex,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "len": len,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "slice": slice,
            "sorted": sorted,
            "sum": sum,
            "zip": zip,
            # Konstanten
            "True": True,
            "False": False,
            "None": None,
            # Sicherer Import
            "__import__": safe_import,
        }

        return {
            "__builtins__": safe_builtins,
            "math": math,
            "statistics": statistics,
            "json": json,
            "datetime": datetime,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "re": re,
            "random": random,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": "Führt Python-Code sicher in einer Sandbox aus",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python-Code zum Ausführen",
                    },
                    "variables": {
                        "type": "object",
                        "description": "Variablen die im Code verfügbar sein sollen",
                    },
                },
                "required": ["code"],
            },
        }
