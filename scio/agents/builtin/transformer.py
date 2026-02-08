"""
SCIO Transformer Agent

Agent für Datentransformationen.
"""

import copy
import re
from typing import Any, Callable, Optional

from scio.agents.base import Agent, AgentConfig, AgentContext
from scio.agents.registry import register_agent
from scio.core.exceptions import AgentError
from scio.core.logging import get_logger

logger = get_logger(__name__)


class TransformerConfig(AgentConfig):
    """Konfiguration für den Transformer Agent."""

    operations: list[dict[str, Any]] = []
    strict_mode: bool = False


@register_agent("transformer")
class TransformerAgent(Agent[dict[str, Any], dict[str, Any]]):
    """
    Agent für Datentransformationen.

    Unterstützte Operationen:
    - filter: Filtert Daten nach Bedingung
    - map: Wendet Funktion auf alle Elemente an
    - select: Wählt Felder aus
    - rename: Benennt Felder um
    - sort: Sortiert Daten
    - limit: Begrenzt Anzahl
    - aggregate: Aggregiert Daten
    """

    agent_type = "transformer"
    version = "1.0"

    def __init__(self, config: TransformerConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = TransformerConfig(**config)
        super().__init__(config)
        self.config: TransformerConfig = config

    async def execute(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Führt Transformationen durch."""

        data = input_data.get("data")
        if data is None:
            raise AgentError("Keine Daten zum Transformieren", agent_id=self.agent_id)

        operations = input_data.get("operations", self.config.operations)

        self.logger.info(
            "Transforming data",
            operation_count=len(operations),
            data_type=type(data).__name__,
        )

        result = copy.deepcopy(data)

        for i, op in enumerate(operations):
            op_type = op.get("type")
            if not op_type:
                raise AgentError(f"Operation {i} hat keinen Typ")

            handler = getattr(self, f"_op_{op_type}", None)
            if handler is None:
                raise AgentError(f"Unbekannte Operation: {op_type}")

            try:
                result = handler(result, op)
            except Exception as e:
                if self.config.strict_mode:
                    raise AgentError(f"Operation {op_type} fehlgeschlagen: {e}")
                self.logger.warning(f"Operation {op_type} fehlgeschlagen: {e}")

        return {
            "data": result,
            "operations_applied": len(operations),
            "result_type": type(result).__name__,
            "result_count": len(result) if isinstance(result, list) else None,
        }

    def _op_filter(self, data: list[dict], op: dict) -> list[dict]:
        """Filtert Daten nach Bedingung."""
        field = op.get("field")
        condition = op.get("condition")  # eq, ne, gt, lt, gte, lte, contains, regex
        value = op.get("value")

        if not all([field, condition]):
            raise ValueError("filter benötigt field und condition")

        comparators = {
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
            "contains": lambda a, b: b in str(a),
            "regex": lambda a, b: bool(re.search(b, str(a))),
            "in": lambda a, b: a in b,
            "not_in": lambda a, b: a not in b,
            "is_null": lambda a, b: a is None,
            "not_null": lambda a, b: a is not None,
        }

        cmp = comparators.get(condition)
        if cmp is None:
            raise ValueError(f"Unbekannte Bedingung: {condition}")

        return [row for row in data if field in row and cmp(row[field], value)]

    def _op_map(self, data: list[dict], op: dict) -> list[dict]:
        """Wendet Transformation auf Feld an."""
        field = op.get("field")
        transform = op.get("transform")  # upper, lower, trim, round, abs, int, float, str
        target = op.get("target", field)  # Zielfeld (optional)

        if not all([field, transform]):
            raise ValueError("map benötigt field und transform")

        transforms = {
            "upper": lambda x: str(x).upper(),
            "lower": lambda x: str(x).lower(),
            "trim": lambda x: str(x).strip(),
            "round": lambda x: round(float(x), op.get("decimals", 2)),
            "abs": lambda x: abs(float(x)),
            "int": lambda x: int(float(x)),
            "float": lambda x: float(x),
            "str": lambda x: str(x),
            "len": lambda x: len(x) if hasattr(x, "__len__") else 0,
        }

        func = transforms.get(transform)
        if func is None:
            raise ValueError(f"Unbekannte Transformation: {transform}")

        result = []
        for row in data:
            new_row = dict(row)
            if field in new_row:
                try:
                    new_row[target] = func(new_row[field])
                except (ValueError, TypeError):
                    pass  # Behalte Original bei Fehler
            result.append(new_row)

        return result

    def _op_select(self, data: list[dict], op: dict) -> list[dict]:
        """Wählt spezifische Felder aus."""
        fields = op.get("fields", [])

        if not fields:
            raise ValueError("select benötigt fields")

        return [{k: row.get(k) for k in fields if k in row} for row in data]

    def _op_rename(self, data: list[dict], op: dict) -> list[dict]:
        """Benennt Felder um."""
        mapping = op.get("mapping", {})

        if not mapping:
            raise ValueError("rename benötigt mapping")

        result = []
        for row in data:
            new_row = {}
            for k, v in row.items():
                new_key = mapping.get(k, k)
                new_row[new_key] = v
            result.append(new_row)

        return result

    def _op_sort(self, data: list[dict], op: dict) -> list[dict]:
        """Sortiert Daten."""
        field = op.get("field")
        descending = op.get("descending", False)

        if not field:
            raise ValueError("sort benötigt field")

        return sorted(
            data,
            key=lambda x: x.get(field, ""),
            reverse=descending,
        )

    def _op_limit(self, data: list, op: dict) -> list:
        """Begrenzt die Anzahl der Elemente."""
        count = op.get("count", 10)
        offset = op.get("offset", 0)

        return data[offset : offset + count]

    def _op_aggregate(self, data: list[dict], op: dict) -> dict[str, Any]:
        """Aggregiert Daten."""
        group_by = op.get("group_by")
        aggregations = op.get("aggregations", [])

        if not aggregations:
            raise ValueError("aggregate benötigt aggregations")

        if group_by:
            # Gruppierte Aggregation
            groups: dict[Any, list] = {}
            for row in data:
                key = row.get(group_by)
                if key not in groups:
                    groups[key] = []
                groups[key].append(row)

            result = {}
            for key, group in groups.items():
                result[str(key)] = self._compute_aggregations(group, aggregations)
            return result
        else:
            # Globale Aggregation
            return self._compute_aggregations(data, aggregations)

    def _compute_aggregations(
        self, data: list[dict], aggregations: list[dict]
    ) -> dict[str, Any]:
        """Berechnet Aggregationen."""
        result = {}

        for agg in aggregations:
            field = agg.get("field")
            func = agg.get("function")  # count, sum, avg, min, max
            alias = agg.get("alias", f"{func}_{field}")

            values = [row.get(field) for row in data if field in row]
            numeric = [float(v) for v in values if isinstance(v, (int, float))]

            if func == "count":
                result[alias] = len(values)
            elif func == "sum":
                result[alias] = sum(numeric) if numeric else 0
            elif func == "avg":
                result[alias] = sum(numeric) / len(numeric) if numeric else 0
            elif func == "min":
                result[alias] = min(numeric) if numeric else None
            elif func == "max":
                result[alias] = max(numeric) if numeric else None
            elif func == "distinct":
                result[alias] = len(set(str(v) for v in values))

        return result
