"""
SCIO Security Validation

Sicherheitsvalidierung für Experimente und Agenten.
"""

import re
from typing import Any

from scio.core.config import get_config
from scio.core.logging import get_logger
from scio.parser.schema import ExperimentSchema
from scio.validation.base import ValidationReport, Validator

logger = get_logger(__name__)


class SecurityValidator(Validator[ExperimentSchema]):
    """
    Validiert Sicherheitsaspekte eines Experiments.

    Prüft auf:
    - Gefährliche Operationen
    - Ressourcen-Limits
    - Sandbox-Verletzungen
    """

    name = "security"

    # Patterns für potenziell gefährliche Operationen
    DANGEROUS_PATTERNS = [
        (r"eval\s*\(", "Verwendung von eval()"),
        (r"exec\s*\(", "Verwendung von exec()"),
        (r"__import__\s*\(", "Dynamischer Import"),
        (r"subprocess", "Subprocess-Zugriff"),
        (r"os\.system", "System-Befehlsausführung"),
        (r"open\s*\([^)]*['\"]w['\"]", "Schreibzugriff auf Dateien"),
        (r"rm\s+-rf", "Gefährlicher Löschbefehl"),
        (r"curl\s+.*\|\s*sh", "Remote Code Execution"),
    ]

    # Blockierte Module (aus Config)
    BLOCKED_MODULES = [
        "os.system",
        "subprocess",
        "eval",
        "exec",
        "pickle",  # Unsichere Deserialisierung
        "marshal",
    ]

    def validate(
        self,
        target: ExperimentSchema,
        context: dict[str, Any] | None = None,
    ) -> ValidationReport:
        report = ValidationReport()
        report.metadata["validator"] = self.name

        config = get_config()

        # Prüfe auf gefährliche Patterns
        self._check_dangerous_patterns(target, report)

        # Prüfe Ressourcen-Limits
        self._check_resource_limits(target, report, config)

        # Prüfe Netzwerkzugriff
        self._check_network_access(target, report, config)

        return report

    def _check_dangerous_patterns(
        self, experiment: ExperimentSchema, report: ValidationReport
    ) -> None:
        """Sucht nach gefährlichen Code-Patterns."""

        # Durchsuche alle String-Werte im Experiment
        def scan_value(value: Any, path: str) -> None:
            if isinstance(value, str):
                for pattern, description in self.DANGEROUS_PATTERNS:
                    if re.search(pattern, value, re.IGNORECASE):
                        report.add_error(
                            message=f"Potenziell gefährliche Operation: {description}",
                            code="SEC_DANGEROUS_PATTERN",
                            location=path,
                            suggestion="Entferne oder ersetze durch sichere Alternative",
                        )

            elif isinstance(value, dict):
                for k, v in value.items():
                    scan_value(v, f"{path}.{k}")

            elif isinstance(value, list):
                for i, v in enumerate(value):
                    scan_value(v, f"{path}[{i}]")

        # Scanne Steps
        for step in experiment.steps:
            scan_value(step.inputs, f"steps.{step.id}.inputs")
            if step.condition:
                scan_value(step.condition, f"steps.{step.id}.condition")

        # Scanne globale Config
        scan_value(experiment.config, "config")

    def _check_resource_limits(
        self, experiment: ExperimentSchema, report: ValidationReport, config: Any
    ) -> None:
        """Prüft Ressourcen-Limits."""

        max_memory = config.security.max_memory_mb

        for step in experiment.steps:
            # Speicher-Limit
            if step.resources.memory_mb > max_memory:
                report.add_error(
                    message=f"Speicher-Limit überschritten: {step.resources.memory_mb}MB > {max_memory}MB",
                    code="SEC_MEMORY_LIMIT",
                    location=f"steps.{step.id}.resources.memory_mb",
                    suggestion=f"Reduziere Speicher auf maximal {max_memory}MB",
                )

            # Extrem langer Timeout
            if step.resources.timeout_seconds > 86400:  # 24h
                report.add_warning(
                    message=f"Sehr langer Timeout: {step.resources.timeout_seconds}s",
                    code="SEC_LONG_TIMEOUT",
                    location=f"steps.{step.id}.resources.timeout_seconds",
                )

            # GPU ohne explizite Berechtigung
            if step.resources.gpu:
                report.add_warning(
                    message="GPU-Zugriff angefordert",
                    code="SEC_GPU_ACCESS",
                    location=f"steps.{step.id}.resources.gpu",
                    suggestion="Stelle sicher, dass GPU-Zugriff autorisiert ist",
                )

    def _check_network_access(
        self, experiment: ExperimentSchema, report: ValidationReport, config: Any
    ) -> None:
        """Prüft Netzwerkzugriffs-Versuche."""

        network_patterns = [
            r"https?://",
            r"ftp://",
            r"socket\.",
            r"requests\.",
            r"urllib",
            r"httpx",
            r"aiohttp",
        ]

        if not config.security.network_enabled:
            for step in experiment.steps:
                inputs_str = str(step.inputs)
                for pattern in network_patterns:
                    if re.search(pattern, inputs_str, re.IGNORECASE):
                        report.add_error(
                            message="Netzwerkzugriff in Sandbox nicht erlaubt",
                            code="SEC_NETWORK_BLOCKED",
                            location=f"steps.{step.id}",
                            suggestion="Aktiviere network_enabled in der Konfiguration",
                        )
                        break
