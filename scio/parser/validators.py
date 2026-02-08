"""
SCIO Validators

Zusätzliche Validierungslogik für Experimente.
"""

from typing import Any

from scio.core.exceptions import ValidationError
from scio.core.logging import get_logger
from scio.parser.schema import ExperimentSchema

logger = get_logger(__name__)


class ValidationResult:
    """Ergebnis einer Validierung."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise ValidationError(
                f"Validierung fehlgeschlagen: {len(self.errors)} Fehler",
                details={"errors": self.errors, "warnings": self.warnings},
            )


def validate_experiment(
    experiment: ExperimentSchema,
    strict: bool = False,
) -> ValidationResult:
    """
    Führt erweiterte Validierung eines Experiments durch.

    Args:
        experiment: Das zu validierende Experiment
        strict: Bei True werden Warnungen zu Fehlern

    Returns:
        ValidationResult mit Fehlern und Warnungen
    """
    result = ValidationResult()

    # Prüfe auf leere Schritte
    if not experiment.steps:
        result.add_error("Experiment muss mindestens einen Schritt haben")

    # Prüfe Ausführungsreihenfolge (Zyklen)
    try:
        experiment.get_execution_order()
    except ValueError as e:
        result.add_error(str(e))

    # Prüfe Agent-Nutzung
    used_agents = {s.agent for s in experiment.steps if s.agent}
    defined_agents = {a.id for a in experiment.agents}

    unused_agents = defined_agents - used_agents
    if unused_agents:
        msg = f"Ungenutzte Agenten definiert: {unused_agents}"
        if strict:
            result.add_error(msg)
        else:
            result.add_warning(msg)

    # Prüfe Ressourcen-Limits
    for step in experiment.steps:
        if step.resources.memory_mb > 8192:
            result.add_warning(
                f"Step '{step.id}' fordert viel Speicher: {step.resources.memory_mb}MB"
            )

        if step.resources.timeout_seconds > 3600:
            result.add_warning(
                f"Step '{step.id}' hat langen Timeout: {step.resources.timeout_seconds}s"
            )

    # Prüfe Metadaten
    if not experiment.metadata.author:
        result.add_warning("Experiment hat keinen Autor angegeben")

    if not experiment.metadata.description:
        result.add_warning("Experiment hat keine Beschreibung")

    # Logge Ergebnis
    if result.is_valid:
        logger.info(
            "Experiment validiert",
            name=experiment.name,
            warnings=len(result.warnings),
        )
    else:
        logger.error(
            "Experiment-Validierung fehlgeschlagen",
            name=experiment.name,
            errors=len(result.errors),
        )

    return result


def validate_yaml_structure(data: dict[str, Any]) -> ValidationResult:
    """
    Validiert die grundlegende YAML-Struktur vor Schema-Validierung.

    Args:
        data: Geparstes YAML-Dictionary

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    required_fields = ["name", "steps"]
    for field in required_fields:
        if field not in data:
            result.add_error(f"Pflichtfeld fehlt: {field}")

    if "steps" in data:
        if not isinstance(data["steps"], list):
            result.add_error("'steps' muss eine Liste sein")
        elif len(data["steps"]) == 0:
            result.add_error("'steps' darf nicht leer sein")

    if "agents" in data and not isinstance(data["agents"], list):
        result.add_error("'agents' muss eine Liste sein")

    return result
