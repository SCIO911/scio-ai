"""
SCIO Math Tools

Tools für mathematische Operationen.
"""

import math
from typing import Any, Optional

from scio.core.logging import get_logger
from scio.tools.base import Tool, ToolConfig
from scio.tools.registry import register_tool

logger = get_logger(__name__)


class MathConfig(ToolConfig):
    """Konfiguration für Math Tool."""

    name: str = "math"
    description: str = "Führt mathematische Berechnungen durch"
    precision: int = 10


@register_tool("math")
class MathTool(Tool[dict[str, Any], dict[str, Any]]):
    """Tool für mathematische Berechnungen."""

    tool_name = "math"
    version = "1.0"

    OPERATIONS = {
        # Grundoperationen
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else float("inf"),
        "modulo": lambda a, b: a % b,
        "power": lambda a, b: a**b,
        "root": lambda a, b: a ** (1 / b),
        # Unäre Operationen
        "abs": lambda a, _: abs(a),
        "sqrt": lambda a, _: math.sqrt(a),
        "log": lambda a, b: math.log(a, b) if b else math.log(a),
        "log10": lambda a, _: math.log10(a),
        "exp": lambda a, _: math.exp(a),
        "factorial": lambda a, _: math.factorial(int(a)),
        # Trigonometrie
        "sin": lambda a, _: math.sin(a),
        "cos": lambda a, _: math.cos(a),
        "tan": lambda a, _: math.tan(a),
        "asin": lambda a, _: math.asin(a),
        "acos": lambda a, _: math.acos(a),
        "atan": lambda a, _: math.atan(a),
        "atan2": lambda a, b: math.atan2(a, b),
        # Rundung
        "floor": lambda a, _: math.floor(a),
        "ceil": lambda a, _: math.ceil(a),
        "round": lambda a, b: round(a, int(b) if b else 0),
        "trunc": lambda a, _: math.trunc(a),
        # Konvertierung
        "degrees": lambda a, _: math.degrees(a),
        "radians": lambda a, _: math.radians(a),
        # Sonstiges
        "gcd": lambda a, b: math.gcd(int(a), int(b)),
        "lcm": lambda a, b: abs(int(a) * int(b)) // math.gcd(int(a), int(b)) if a and b else 0,
        "hypot": lambda a, b: math.hypot(a, b),
    }

    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": float("inf"),
        "nan": float("nan"),
    }

    def __init__(self, config: Optional[MathConfig | dict] = None):
        if config is None:
            config = MathConfig(name="math")
        elif isinstance(config, dict):
            config = MathConfig(**config)
        super().__init__(config)
        self.config: MathConfig = config

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Führt mathematische Operation aus."""
        operation = input_data.get("operation")
        a = input_data.get("a")
        b = input_data.get("b")

        # Wenn Konstante angefordert
        if operation == "constant":
            name = input_data.get("name", "pi")
            if name not in self.CONSTANTS:
                raise ValueError(f"Unbekannte Konstante: {name}")
            return {"result": self.CONSTANTS[name], "constant": name}

        # Wenn Expression
        if operation == "evaluate":
            expr = input_data.get("expression")
            if not expr:
                raise ValueError("Keine Expression angegeben")
            result = self._safe_eval(expr)
            return {"result": result, "expression": expr}

        # Normale Operation
        if operation not in self.OPERATIONS:
            raise ValueError(
                f"Unbekannte Operation: {operation}. "
                f"Verfügbar: {list(self.OPERATIONS.keys())}"
            )

        if a is None:
            raise ValueError("Parameter 'a' fehlt")

        try:
            result = self.OPERATIONS[operation](float(a), float(b) if b is not None else None)

            # Runde Ergebnis
            if isinstance(result, float) and not (math.isinf(result) or math.isnan(result)):
                result = round(result, self.config.precision)

            return {
                "result": result,
                "operation": operation,
                "a": a,
                "b": b,
            }

        except Exception as e:
            raise ValueError(f"Berechnungsfehler: {e}")

    def _safe_eval(self, expr: str) -> float:
        """Evaluiert eine mathematische Expression sicher."""
        # Erlaubte Zeichen
        allowed = set("0123456789+-*/().^ ")
        allowed.update(set("sincostanlogexpabssqrtpi eE"))

        # Prüfe auf nicht erlaubte Zeichen
        if not all(c in allowed for c in expr.replace(" ", "")):
            raise ValueError("Expression enthält nicht erlaubte Zeichen")

        # Ersetze ^ durch **
        expr = expr.replace("^", "**")

        # Ersetze Funktionsnamen
        replacements = {
            "sin": "math.sin",
            "cos": "math.cos",
            "tan": "math.tan",
            "log": "math.log",
            "exp": "math.exp",
            "abs": "abs",
            "sqrt": "math.sqrt",
            "pi": "math.pi",
        }

        for name, replacement in replacements.items():
            expr = expr.replace(name, replacement)

        # Evaluiere sicher
        try:
            result = eval(expr, {"__builtins__": {}, "math": math, "abs": abs})
            return float(result)
        except Exception as e:
            raise ValueError(f"Evaluierungsfehler: {e}")

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": "Führt mathematische Berechnungen durch",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": list(self.OPERATIONS.keys()) + ["constant", "evaluate"],
                        "description": "Die mathematische Operation",
                    },
                    "a": {
                        "type": "number",
                        "description": "Erster Operand",
                    },
                    "b": {
                        "type": "number",
                        "description": "Zweiter Operand (optional)",
                    },
                    "expression": {
                        "type": "string",
                        "description": "Mathematische Expression (für 'evaluate')",
                    },
                    "name": {
                        "type": "string",
                        "description": "Konstantenname (für 'constant')",
                    },
                },
                "required": ["operation"],
            },
        }
