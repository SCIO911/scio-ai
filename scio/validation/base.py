"""
SCIO Base Validation

Basis-Klassen für das Validierungssystem.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from scio.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class Severity(str, Enum):
    """Schweregrad von Validierungsproblemen."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Ein einzelnes Validierungsproblem."""

    message: str
    severity: Severity
    code: str
    location: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationReport:
    """Gesamtbericht einer Validierung."""

    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Gültig wenn keine Errors oder Critical."""
        return not any(
            i.severity in {Severity.ERROR, Severity.CRITICAL} for i in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def add_error(
        self,
        message: str,
        code: str,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.add_issue(
            ValidationIssue(
                message=message,
                severity=Severity.ERROR,
                code=code,
                location=location,
                suggestion=suggestion,
            )
        )

    def add_warning(
        self,
        message: str,
        code: str,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.add_issue(
            ValidationIssue(
                message=message,
                severity=Severity.WARNING,
                code=code,
                location=location,
                suggestion=suggestion,
            )
        )

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        """Führt zwei Reports zusammen."""
        merged = ValidationReport(
            issues=self.issues + other.issues,
            metadata={**self.metadata, **other.metadata},
        )
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
        }


class Validator(ABC, Generic[T]):
    """
    Abstrakte Basis-Klasse für Validatoren.

    Implementiere `validate` für spezifische Validierungslogik.
    """

    name: str = "base"

    @abstractmethod
    def validate(self, target: T, context: dict[str, Any] | None = None) -> ValidationReport:
        """
        Führt Validierung durch.

        Args:
            target: Zu validierendes Objekt
            context: Optionaler Kontext

        Returns:
            ValidationReport mit Ergebnissen
        """
        pass


class ValidationChain:
    """
    Führt mehrere Validatoren in Reihe aus.

    Beispiel:
        chain = ValidationChain([
            SecurityValidator(),
            ScientificValidator(),
        ])
        report = chain.validate(experiment)
    """

    def __init__(self, validators: list[Validator] | None = None):
        self.validators = validators or []
        self.logger = get_logger(__name__, component="validation_chain")

    def add(self, validator: Validator) -> "ValidationChain":
        """Fügt einen Validator hinzu."""
        self.validators.append(validator)
        return self

    def validate(
        self,
        target: Any,
        context: dict[str, Any] | None = None,
        fail_fast: bool = False,
    ) -> ValidationReport:
        """
        Führt alle Validatoren aus.

        Args:
            target: Zu validierendes Objekt
            context: Optionaler Kontext
            fail_fast: Bei True wird nach erstem Fehler abgebrochen

        Returns:
            Zusammengeführter ValidationReport
        """
        combined = ValidationReport()
        context = context or {}

        for validator in self.validators:
            self.logger.debug("Running validator", validator=validator.name)

            try:
                report = validator.validate(target, context)
                combined = combined.merge(report)

                if fail_fast and not report.is_valid:
                    self.logger.warning(
                        "Validation failed, stopping chain",
                        validator=validator.name,
                    )
                    break

            except Exception as e:
                self.logger.error(
                    "Validator raised exception",
                    validator=validator.name,
                    error=str(e),
                )
                combined.add_error(
                    message=f"Validator '{validator.name}' crashed: {e}",
                    code="VALIDATOR_CRASH",
                )
                if fail_fast:
                    break

        return combined
