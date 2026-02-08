#!/usr/bin/env python3
"""
SCIO Plugin System
Ermöglicht Erweiterung durch Custom Handler, Hooks und Plugins
"""

import importlib
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PLUGIN REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PluginInfo:
    """Information über ein registriertes Plugin"""
    name: str
    version: str
    description: str
    author: str = ""
    handler: Optional[Callable] = None
    enabled: bool = True
    registered_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """
    Zentrale Registry für Plugins und Custom Handler

    Verwendung:
        registry = PluginRegistry()

        @registry.register("my_handler", version="1.0", description="My custom handler")
        def my_handler(workflow, step):
            return {"result": "processed"}

        # Später aufrufen
        result = registry.execute("my_handler", workflow, step)
    """

    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def register(self, name: str,
                 version: str = "1.0.0",
                 description: str = "",
                 author: str = "",
                 **metadata):
        """
        Decorator zum Registrieren eines Plugins/Handlers

        Args:
            name: Eindeutiger Name des Handlers
            version: Version des Handlers
            description: Beschreibung
            author: Autor
            **metadata: Zusätzliche Metadaten
        """
        def decorator(func: Callable) -> Callable:
            with self._lock:
                if name in self._plugins:
                    logger.warning(f"Plugin {name} already registered, overwriting")

                self._plugins[name] = PluginInfo(
                    name=name,
                    version=version,
                    description=description,
                    author=author,
                    handler=func,
                    metadata=metadata
                )
                logger.info(f"Plugin registered: {name} v{version}")

            return func
        return decorator

    def register_handler(self, name: str, handler: Callable,
                        version: str = "1.0.0",
                        description: str = ""):
        """Registriert Handler direkt (nicht als Decorator)"""
        with self._lock:
            self._plugins[name] = PluginInfo(
                name=name,
                version=version,
                description=description,
                handler=handler
            )
            logger.info(f"Handler registered: {name}")

    def unregister(self, name: str) -> bool:
        """Entfernt ein Plugin"""
        with self._lock:
            if name in self._plugins:
                del self._plugins[name]
                logger.info(f"Plugin unregistered: {name}")
                return True
            return False

    def get(self, name: str) -> Optional[PluginInfo]:
        """Holt Plugin-Info"""
        with self._lock:
            return self._plugins.get(name)

    def exists(self, name: str) -> bool:
        """Prüft ob Plugin existiert"""
        with self._lock:
            return name in self._plugins

    def execute(self, name: str, *args, **kwargs) -> Any:
        """
        Führt Plugin-Handler aus

        Raises:
            PluginNotFoundError: Plugin nicht gefunden
            PluginDisabledError: Plugin ist deaktiviert
        """
        with self._lock:
            plugin = self._plugins.get(name)

        if not plugin:
            raise PluginNotFoundError(f"Plugin not found: {name}")

        if not plugin.enabled:
            raise PluginDisabledError(f"Plugin is disabled: {name}")

        if not plugin.handler:
            raise PluginNotFoundError(f"Plugin has no handler: {name}")

        try:
            return plugin.handler(*args, **kwargs)
        except Exception as e:
            logger.error(f"Plugin {name} execution error: {e}")
            raise

    def enable(self, name: str) -> bool:
        """Aktiviert Plugin"""
        with self._lock:
            if name in self._plugins:
                self._plugins[name].enabled = True
                return True
            return False

    def disable(self, name: str) -> bool:
        """Deaktiviert Plugin"""
        with self._lock:
            if name in self._plugins:
                self._plugins[name].enabled = False
                return True
            return False

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Gibt Liste aller Plugins zurück"""
        with self._lock:
            return [
                {
                    'name': p.name,
                    'version': p.version,
                    'description': p.description,
                    'author': p.author,
                    'enabled': p.enabled,
                    'registered_at': p.registered_at.isoformat()
                }
                for p in self._plugins.values()
            ]

    # ═══════════════════════════════════════════════════════════════════
    # HOOKS
    # ═══════════════════════════════════════════════════════════════════

    def add_hook(self, hook_name: str, callback: Callable):
        """Registriert einen Hook-Callback"""
        with self._lock:
            if hook_name not in self._hooks:
                self._hooks[hook_name] = []
            self._hooks[hook_name].append(callback)
            logger.debug(f"Hook added: {hook_name}")

    def remove_hook(self, hook_name: str, callback: Callable) -> bool:
        """Entfernt Hook-Callback"""
        with self._lock:
            if hook_name in self._hooks:
                try:
                    self._hooks[hook_name].remove(callback)
                    return True
                except ValueError:
                    pass
            return False

    def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Triggert alle Callbacks für einen Hook

        Returns:
            Liste der Rückgabewerte aller Callbacks
        """
        with self._lock:
            hooks = list(self._hooks.get(hook_name, []))

        results = []
        for callback in hooks:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook {hook_name} callback error: {e}")
                results.append(None)

        return results


class PluginNotFoundError(Exception):
    """Plugin wurde nicht gefunden"""
    pass


class PluginDisabledError(Exception):
    """Plugin ist deaktiviert"""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW CUSTOM HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

class WorkflowHandlerRegistry(PluginRegistry):
    """
    Spezielle Registry für Workflow Step Handler

    Verwendung:
        handlers = WorkflowHandlerRegistry()

        @handlers.register("data_transform", description="Transform data")
        def transform_handler(workflow, step):
            data = workflow.context.get('data', {})
            # Transform...
            return {"transformed": True}

        # In WorkflowEngine nutzen
        engine.register_custom_handlers(handlers)
    """

    def validate_handler(self, name: str) -> List[str]:
        """
        Validiert ob Handler korrekte Signatur hat

        Returns:
            Liste von Validierungsfehlern (leer wenn OK)
        """
        plugin = self.get(name)
        if not plugin or not plugin.handler:
            return [f"Handler {name} not found"]

        errors = []

        # Signatur prüfen
        sig = inspect.signature(plugin.handler)
        params = list(sig.parameters.keys())

        if len(params) < 2:
            errors.append(f"Handler {name} must accept at least 2 parameters (workflow, step)")

        return errors

    def create_handler(self, name: str,
                      handler_func: Callable,
                      description: str = "",
                      input_schema: Dict = None,
                      output_schema: Dict = None):
        """
        Erstellt und registriert einen neuen Handler

        Args:
            name: Handler-Name
            handler_func: Die Handler-Funktion
            description: Beschreibung
            input_schema: Schema für erwartete Inputs
            output_schema: Schema für erwartete Outputs
        """
        self.register_handler(
            name=name,
            handler=handler_func,
            description=description
        )

        # Schemas speichern
        plugin = self.get(name)
        if plugin:
            plugin.metadata['input_schema'] = input_schema or {}
            plugin.metadata['output_schema'] = output_schema or {}


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES
# ═══════════════════════════════════════════════════════════════════════════

_plugin_registry: Optional[PluginRegistry] = None
_workflow_handlers: Optional[WorkflowHandlerRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Gibt globale Plugin-Registry zurück"""
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry


def get_workflow_handlers() -> WorkflowHandlerRegistry:
    """Gibt globale Workflow-Handler-Registry zurück"""
    global _workflow_handlers
    if _workflow_handlers is None:
        _workflow_handlers = WorkflowHandlerRegistry()
        _register_default_handlers(_workflow_handlers)
    return _workflow_handlers


def _register_default_handlers(registry: WorkflowHandlerRegistry):
    """Registriert Standard-Handler"""

    @registry.register("echo", description="Echoes input back")
    def echo_handler(workflow, step):
        return {"echo": step.config}

    @registry.register("log", description="Logs a message")
    def log_handler(workflow, step):
        message = step.config.get("message", "Log from workflow")
        level = step.config.get("level", "info")
        getattr(logger, level, logger.info)(f"[Workflow] {message}")
        return {"logged": True}

    @registry.register("set_context", description="Sets context variables")
    def set_context_handler(workflow, step):
        for key, value in step.config.get("variables", {}).items():
            workflow.context[key] = value
        return {"set": list(step.config.get("variables", {}).keys())}

    @registry.register("validate", description="Validates context against schema")
    def validate_handler(workflow, step):
        schema = step.config.get("schema", {})
        errors = []
        for field, rules in schema.items():
            value = workflow.context.get(field)
            if rules.get("required") and value is None:
                errors.append(f"Missing required field: {field}")
            if rules.get("type") and value is not None:
                expected = rules["type"]
                if expected == "string" and not isinstance(value, str):
                    errors.append(f"Field {field} must be string")
                elif expected == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field {field} must be number")
        return {"valid": len(errors) == 0, "errors": errors}

    @registry.register("transform", description="Transforms data using simple expressions")
    def transform_handler(workflow, step):
        transforms = step.config.get("transforms", {})
        results = {}
        for output_key, expression in transforms.items():
            # Einfache Transformationen
            if expression.startswith("ctx."):
                key = expression[4:]
                results[output_key] = workflow.context.get(key)
            elif expression.startswith("upper:"):
                key = expression[6:]
                value = workflow.context.get(key, "")
                results[output_key] = str(value).upper()
            elif expression.startswith("lower:"):
                key = expression[6:]
                value = workflow.context.get(key, "")
                results[output_key] = str(value).lower()
            else:
                results[output_key] = expression

        workflow.context.update(results)
        return {"transformed": list(results.keys())}

    @registry.register("http_request", description="Makes HTTP request")
    def http_handler(workflow, step):
        import urllib.request
        import json

        url = step.config.get("url", "")
        method = step.config.get("method", "GET")
        headers = step.config.get("headers", {})
        body = step.config.get("body")

        if not url:
            return {"error": "URL required"}

        try:
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(url, data=data, method=method)
            for k, v in headers.items():
                req.add_header(k, v)

            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode()
                try:
                    result = json.loads(content)
                except:
                    result = content

                return {
                    "status": response.status,
                    "data": result
                }
        except Exception as e:
            return {"error": str(e)}

    @registry.register("delay", description="Waits for specified time")
    def delay_handler(workflow, step):
        import time
        seconds = step.config.get("seconds", 1)
        time.sleep(min(seconds, 60))  # Max 60 Sekunden
        return {"delayed": seconds}

    @registry.register("branch", description="Conditional branching")
    def branch_handler(workflow, step):
        from backend.core.security import SafeExpressionEvaluator

        condition = step.config.get("condition", "True")
        true_result = step.config.get("on_true", {})
        false_result = step.config.get("on_false", {})

        try:
            evaluator = SafeExpressionEvaluator({"ctx": workflow.context})
            result = evaluator.evaluate(condition)
            output = true_result if result else false_result
            return {"condition_result": result, "output": output}
        except Exception as e:
            return {"error": str(e)}

    logger.info(f"Registered {len(registry._plugins)} default workflow handlers")
