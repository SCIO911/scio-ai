#!/usr/bin/env python3
"""
SCIO Security Module
Sichere Expression-Auswertung, Input-Validierung, Rate Limiting
"""

import ast
import operator
import re
import time
import threading
import hashlib
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SAFE EXPRESSION EVALUATOR - Ersetzt unsicheres eval()
# ═══════════════════════════════════════════════════════════════════════════

class SafeExpressionEvaluator:
    """
    Sichere Auswertung von Ausdrücken ohne eval()

    Unterstützt:
    - Vergleichsoperatoren: ==, !=, <, >, <=, >=
    - Logische Operatoren: and, or, not
    - Arithmetik: +, -, *, /, //, %, **
    - Membership: in, not in
    - Attributzugriff: ctx.value, ctx["key"]
    - Literale: int, float, str, bool, None, list, dict
    """

    # Erlaubte Operatoren
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Not: operator.not_,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
    }

    # Erlaubte eingebaute Funktionen
    SAFE_FUNCTIONS = {
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'round': round,
        'sorted': sorted,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'any': any,
        'all': all,
        'isinstance': isinstance,
        'type': lambda x: type(x).__name__,
    }

    # Maximale Rekursionstiefe und Komplexität
    MAX_DEPTH = 20
    MAX_NODES = 100

    def __init__(self, context: Dict[str, Any] = None,
                 allowed_names: Set[str] = None,
                 custom_functions: Dict[str, Callable] = None):
        """
        Args:
            context: Variablen die im Ausdruck verfügbar sind
            allowed_names: Zusätzlich erlaubte Variablennamen
            custom_functions: Zusätzliche sichere Funktionen
        """
        self.context = context or {}
        self.allowed_names = allowed_names or set()
        self.functions = {**self.SAFE_FUNCTIONS}
        if custom_functions:
            self.functions.update(custom_functions)

        self._depth = 0
        self._node_count = 0

    def evaluate(self, expression: str) -> Any:
        """
        Wertet einen Ausdruck sicher aus

        Args:
            expression: Python-ähnlicher Ausdruck

        Returns:
            Ergebnis der Auswertung

        Raises:
            ValueError: Bei ungültigem oder unsicherem Ausdruck
        """
        if not expression or not isinstance(expression, str):
            raise ValueError("Expression must be a non-empty string")

        # Längenprüfung
        if len(expression) > 1000:
            raise ValueError("Expression too long (max 1000 chars)")

        # Gefährliche Patterns blockieren
        dangerous_patterns = [
            r'__\w+__',  # Dunder methods
            r'\bexec\b', r'\beval\b', r'\bcompile\b',
            r'\bimport\b', r'\b__import__\b',
            r'\bopen\b', r'\bfile\b',
            r'\bos\b', r'\bsys\b', r'\bsubprocess\b',
            r'\bglobals\b', r'\blocals\b',
            r'\bgetattr\b', r'\bsetattr\b', r'\bdelattr\b',
            r'\bvars\b', r'\bdir\b',
            r'lambda\s*:',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValueError(f"Forbidden pattern in expression: {pattern}")

        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")

        # Komplexität prüfen
        self._node_count = 0
        self._count_nodes(tree)
        if self._node_count > self.MAX_NODES:
            raise ValueError(f"Expression too complex ({self._node_count} nodes, max {self.MAX_NODES})")

        self._depth = 0
        try:
            return self._eval_node(tree.body)
        except RecursionError:
            raise ValueError("Expression recursion limit exceeded")

    def _count_nodes(self, node):
        """Zählt AST-Knoten"""
        self._node_count += 1
        for child in ast.iter_child_nodes(node):
            self._count_nodes(child)

    def _eval_node(self, node) -> Any:
        """Evaluiert einen AST-Knoten rekursiv"""
        self._depth += 1
        if self._depth > self.MAX_DEPTH:
            raise ValueError("Expression nesting too deep")

        try:
            return self._eval_node_inner(node)
        finally:
            self._depth -= 1

    def _eval_node_inner(self, node) -> Any:
        """Innere Evaluation ohne Tiefentracking"""

        # Literale
        if isinstance(node, ast.Constant):
            return node.value

        # Alte Python-Versionen: Num, Str, NameConstant
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Str):
            return node.s
        if isinstance(node, (ast.NameConstant,)):
            return node.value

        # Namen (Variablen)
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.functions:
                return self.functions[name]
            if name in self.context:
                return self.context[name]
            if name in self.allowed_names:
                return None  # Erlaubt aber nicht definiert
            if name in ('True', 'False', 'None'):
                return {'True': True, 'False': False, 'None': None}[name]
            raise ValueError(f"Unknown name: {name}")

        # Attributzugriff (ctx.value)
        if isinstance(node, ast.Attribute):
            obj = self._eval_node(node.value)
            attr = node.attr

            # Dunder blockieren
            if attr.startswith('_'):
                raise ValueError(f"Access to private attribute forbidden: {attr}")

            if isinstance(obj, dict):
                return obj.get(attr)
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                # Methoden nicht erlauben (außer sichere)
                if callable(val) and not isinstance(obj, dict):
                    raise ValueError(f"Method access not allowed: {attr}")
                return val
            return None

        # Subscript (ctx["key"] oder ctx[0])
        if isinstance(node, ast.Subscript):
            obj = self._eval_node(node.value)

            # Handle different slice types for Python compatibility
            if isinstance(node.slice, ast.Index):
                # Python < 3.9
                key = self._eval_node(node.slice.value)
            else:
                # Python >= 3.9
                key = self._eval_node(node.slice)

            if isinstance(obj, (dict, list, tuple, str)):
                try:
                    return obj[key]
                except (KeyError, IndexError, TypeError):
                    return None
            return None

        # Binäre Operatoren
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)

        # Unäre Operatoren
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(operand)

        # Vergleiche
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                op_func = self.OPERATORS.get(type(op))
                if op_func is None:
                    raise ValueError(f"Unsupported comparison: {type(op).__name__}")
                if not op_func(left, right):
                    return False
                left = right
            return True

        # Logische Operatoren (and, or)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not self._eval_node(value):
                        return False
                return True
            elif isinstance(node.op, ast.Or):
                for value in node.values:
                    if self._eval_node(value):
                        return True
                return False

        # Funktionsaufrufe
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            if not callable(func):
                raise ValueError(f"Not callable: {func}")
            if func not in self.functions.values():
                # Prüfe ob es eine erlaubte Funktion ist
                func_name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
                if func_name not in self.functions:
                    raise ValueError(f"Function not allowed: {func_name}")

            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value) for kw in node.keywords}
            return func(*args, **kwargs)

        # Listen
        if isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]

        # Tupel
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)

        # Dictionaries
        if isinstance(node, ast.Dict):
            return {
                self._eval_node(k): self._eval_node(v)
                for k, v in zip(node.keys, node.values)
            }

        # Sets
        if isinstance(node, ast.Set):
            return {self._eval_node(elt) for elt in node.elts}

        # If-Expression (ternary)
        if isinstance(node, ast.IfExp):
            if self._eval_node(node.test):
                return self._eval_node(node.body)
            return self._eval_node(node.orelse)

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str, context: Dict[str, Any] = None) -> Any:
    """
    Convenience-Funktion für sichere Expression-Auswertung

    Args:
        expression: Auszuwertender Ausdruck
        context: Variablen-Kontext

    Returns:
        Ergebnis der Auswertung
    """
    evaluator = SafeExpressionEvaluator(context)
    return evaluator.evaluate(expression)


# ═══════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationRule:
    """Regel für Input-Validierung"""
    field: str
    type: type
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None


class InputValidator:
    """
    Validiert Eingaben gegen definierte Regeln

    Verwendung:
        validator = InputValidator()
        validator.add_rule(ValidationRule('name', str, min_length=1, max_length=100))
        validator.add_rule(ValidationRule('age', int, min_value=0, max_value=150))

        errors = validator.validate({'name': 'Test', 'age': 25})
    """

    # Gefährliche Patterns für XSS/Injection
    DANGEROUS_PATTERNS = [
        r'<script\b',
        r'javascript:',
        r'on\w+\s*=',
        r'data:text/html',
        r'\bSELECT\b.*\bFROM\b',
        r'\bINSERT\b.*\bINTO\b',
        r'\bDELETE\b.*\bFROM\b',
        r'\bDROP\b.*\bTABLE\b',
        r'\bUNION\b.*\bSELECT\b',
        r';\s*--',
        r'\bexec\s*\(',
        r'\beval\s*\(',
    ]

    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.DANGEROUS_PATTERNS
        ]

    def add_rule(self, rule: ValidationRule):
        """Fügt Validierungsregel hinzu"""
        self.rules[rule.field] = rule

    def validate(self, data: Dict[str, Any]) -> List[str]:
        """
        Validiert Daten gegen alle Regeln

        Returns:
            Liste von Fehlermeldungen (leer wenn valide)
        """
        errors = []

        for field, rule in self.rules.items():
            value = data.get(field)

            # Required Check
            if rule.required and value is None:
                errors.append(f"Field '{field}' is required")
                continue

            if value is None:
                continue

            # Type Check
            if not isinstance(value, rule.type):
                errors.append(f"Field '{field}' must be of type {rule.type.__name__}")
                continue

            # String validations
            if isinstance(value, str):
                # Gefährliche Patterns prüfen
                for pattern in self._compiled_patterns:
                    if pattern.search(value):
                        errors.append(f"Field '{field}' contains potentially dangerous content")
                        break

                if rule.min_length and len(value) < rule.min_length:
                    errors.append(f"Field '{field}' must be at least {rule.min_length} characters")

                if rule.max_length and len(value) > rule.max_length:
                    errors.append(f"Field '{field}' must be at most {rule.max_length} characters")

                if rule.pattern and not re.match(rule.pattern, value):
                    errors.append(f"Field '{field}' does not match required pattern")

            # Numeric validations
            if isinstance(value, (int, float)):
                if rule.min_value is not None and value < rule.min_value:
                    errors.append(f"Field '{field}' must be at least {rule.min_value}")

                if rule.max_value is not None and value > rule.max_value:
                    errors.append(f"Field '{field}' must be at most {rule.max_value}")

            # Allowed values
            if rule.allowed_values and value not in rule.allowed_values:
                errors.append(f"Field '{field}' must be one of {rule.allowed_values}")

            # Custom validator
            if rule.custom_validator:
                try:
                    if not rule.custom_validator(value):
                        errors.append(f"Field '{field}' failed custom validation")
                except Exception as e:
                    errors.append(f"Field '{field}' validation error: {e}")

        return errors

    def is_valid(self, data: Dict[str, Any]) -> bool:
        """Prüft ob Daten valide sind"""
        return len(self.validate(data)) == 0


def sanitize_input(value: str, max_length: int = 10000) -> str:
    """
    Bereinigt einen String von gefährlichen Inhalten

    Args:
        value: Zu bereinigender String
        max_length: Maximale Länge

    Returns:
        Bereinigter String
    """
    if not isinstance(value, str):
        return str(value)[:max_length]

    # Länge begrenzen
    value = value[:max_length]

    # HTML-Entities escapen
    value = value.replace('&', '&amp;')
    value = value.replace('<', '&lt;')
    value = value.replace('>', '&gt;')
    value = value.replace('"', '&quot;')
    value = value.replace("'", '&#x27;')

    # Null-Bytes entfernen
    value = value.replace('\x00', '')

    return value


def validate_tool_params(tool_id: str, params: Dict[str, Any],
                         schema: Dict[str, Any] = None) -> List[str]:
    """
    Validiert Tool-Parameter

    Args:
        tool_id: ID des Tools
        params: Zu validierende Parameter
        schema: Optionales Schema für Validierung

    Returns:
        Liste von Fehlermeldungen
    """
    errors = []

    # Basis-Validierung
    if not isinstance(params, dict):
        errors.append("Parameters must be a dictionary")
        return errors

    # Maximale Größe prüfen
    import json
    try:
        json_str = json.dumps(params)
        if len(json_str) > 1_000_000:  # 1MB max
            errors.append("Parameters too large (max 1MB)")
    except (TypeError, ValueError) as e:
        errors.append(f"Parameters not JSON-serializable: {e}")

    # Schema-basierte Validierung
    if schema:
        for param_name, param_schema in schema.items():
            value = params.get(param_name)

            if param_schema.get('required', False) and value is None:
                errors.append(f"Parameter '{param_name}' is required")
                continue

            if value is not None:
                expected_type = param_schema.get('type')
                if expected_type:
                    type_map = {
                        'string': str,
                        'integer': int,
                        'number': (int, float),
                        'boolean': bool,
                        'array': list,
                        'object': dict,
                    }
                    py_type = type_map.get(expected_type)
                    if py_type and not isinstance(value, py_type):
                        errors.append(f"Parameter '{param_name}' must be {expected_type}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RateLimitConfig:
    """Konfiguration für Rate Limiting"""
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    burst_size: int = 20
    block_duration_seconds: int = 60


class RateLimiter:
    """
    Token Bucket Rate Limiter mit mehreren Zeitfenstern

    Verwendung:
        limiter = RateLimiter()

        if limiter.is_allowed(client_id):
            # Request verarbeiten
        else:
            # Rate limit exceeded
    """

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()

        # Token Buckets pro Client
        self._buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'tokens': self.config.burst_size,
                'last_update': time.time(),
            }
        )

        # Request-Counter pro Zeitfenster
        self._counters: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {'second': [], 'minute': [], 'hour': []}
        )

        # Geblockte Clients
        self._blocked: Dict[str, float] = {}

        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """
        Prüft ob ein Request erlaubt ist

        Args:
            client_id: Eindeutige Client-ID (z.B. IP, API-Key)

        Returns:
            True wenn erlaubt
        """
        with self._lock:
            now = time.time()

            # Geblockt?
            if client_id in self._blocked:
                if now < self._blocked[client_id]:
                    return False
                del self._blocked[client_id]

            # Token Bucket aktualisieren
            bucket = self._buckets[client_id]
            elapsed = now - bucket['last_update']
            bucket['tokens'] = min(
                self.config.burst_size,
                bucket['tokens'] + elapsed * self.config.requests_per_second
            )
            bucket['last_update'] = now

            # Counter aktualisieren
            counters = self._counters[client_id]

            # Alte Einträge entfernen
            counters['second'] = [t for t in counters['second'] if now - t < 1]
            counters['minute'] = [t for t in counters['minute'] if now - t < 60]
            counters['hour'] = [t for t in counters['hour'] if now - t < 3600]

            # Limits prüfen
            if len(counters['second']) >= self.config.requests_per_second:
                return False
            if len(counters['minute']) >= self.config.requests_per_minute:
                return False
            if len(counters['hour']) >= self.config.requests_per_hour:
                self._blocked[client_id] = now + self.config.block_duration_seconds
                return False

            # Token verfügbar?
            if bucket['tokens'] < 1:
                return False

            # Request erlauben
            bucket['tokens'] -= 1
            counters['second'].append(now)
            counters['minute'].append(now)
            counters['hour'].append(now)

            return True

    def get_remaining(self, client_id: str) -> Dict[str, int]:
        """Gibt verbleibende Requests zurück"""
        with self._lock:
            now = time.time()
            counters = self._counters.get(client_id, {'second': [], 'minute': [], 'hour': []})

            return {
                'per_second': max(0, int(self.config.requests_per_second) -
                                len([t for t in counters['second'] if now - t < 1])),
                'per_minute': max(0, int(self.config.requests_per_minute) -
                                 len([t for t in counters['minute'] if now - t < 60])),
                'per_hour': max(0, int(self.config.requests_per_hour) -
                               len([t for t in counters['hour'] if now - t < 3600])),
            }

    def reset(self, client_id: str):
        """Setzt Rate Limit für Client zurück"""
        with self._lock:
            if client_id in self._buckets:
                del self._buckets[client_id]
            if client_id in self._counters:
                del self._counters[client_id]
            if client_id in self._blocked:
                del self._blocked[client_id]


# Globale Instanz
_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    """Gibt globale Rate Limiter Instanz zurück"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limit(func: Callable = None, *,
               get_client_id: Callable = None,
               on_exceed: Callable = None):
    """
    Decorator für Rate Limiting

    Args:
        get_client_id: Funktion die Client-ID aus Request extrahiert
        on_exceed: Callback wenn Limit überschritten
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()

            # Client-ID ermitteln
            if get_client_id:
                client_id = get_client_id(*args, **kwargs)
            else:
                # Fallback: Funktion + erstes Argument
                client_id = f.__name__
                if args:
                    client_id = f"{f.__name__}:{hash(str(args[0]))}"

            if not limiter.is_allowed(client_id):
                if on_exceed:
                    return on_exceed(*args, **kwargs)
                raise RateLimitExceeded(f"Rate limit exceeded for {client_id}")

            return f(*args, **kwargs)
        return wrapper

    if func:
        return decorator(func)
    return decorator


class RateLimitExceeded(Exception):
    """Exception für überschrittenes Rate Limit"""
    pass
