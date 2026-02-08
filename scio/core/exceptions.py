"""
SCIO Exception Hierarchy

Zentrale Fehlerdefinitionen für das gesamte Framework.
"""

from typing import Any, Optional


class SCIOError(Exception):
    """Basis-Exception für alle SCIO-Fehler."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "SCIO_ERROR"
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert die Exception für Logging/API-Responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(SCIOError):
    """Fehler bei der Konfiguration."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="CONFIG_ERROR", details=details)


class ValidationError(SCIOError):
    """Fehler bei der Validierung von Eingaben oder Schemas."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class ParsingError(SCIOError):
    """Fehler beim Parsen von YAML/Konfigurationsdateien."""

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if line is not None:
            details["line"] = line
        if column is not None:
            details["column"] = column
        super().__init__(message, code="PARSING_ERROR", details=details)


class ExecutionError(SCIOError):
    """Fehler während der Ausführung von Experimenten/Agenten."""

    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if step:
            details["step"] = step
        super().__init__(message, code="EXECUTION_ERROR", details=details)


class SecurityError(SCIOError):
    """Sicherheitsrelevante Fehler (Sandbox-Verletzungen, etc.)."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="SECURITY_ERROR", details=details)


class ResourceError(SCIOError):
    """Fehler bei Ressourcenzugriff oder -limits."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        super().__init__(message, code="RESOURCE_ERROR", details=details)


class AgentError(SCIOError):
    """Fehler in Agenten-Komponenten."""

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if agent_id:
            details["agent_id"] = agent_id
        super().__init__(message, code="AGENT_ERROR", details=details)


class PluginError(SCIOError):
    """Fehler beim Laden oder Ausführen von Plugins."""

    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if plugin_name:
            details["plugin_name"] = plugin_name
        super().__init__(message, code="PLUGIN_ERROR", details=details)
