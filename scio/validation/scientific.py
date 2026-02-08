"""
SCIO Scientific Validation

Validierung für wissenschaftliche Reproduzierbarkeit und Best Practices.
"""

from typing import Any

from scio.core.logging import get_logger
from scio.parser.schema import ExperimentSchema
from scio.validation.base import ValidationReport, Validator

logger = get_logger(__name__)


class ScientificValidator(Validator[ExperimentSchema]):
    """
    Validiert wissenschaftliche Aspekte eines Experiments.

    Prüft auf:
    - Reproduzierbarkeit
    - Dokumentation
    - Best Practices
    """

    name = "scientific"

    def validate(
        self,
        target: ExperimentSchema,
        context: dict[str, Any] | None = None,
    ) -> ValidationReport:
        report = ValidationReport()
        report.metadata["validator"] = self.name

        # Prüfe Metadaten
        self._validate_metadata(target, report)

        # Prüfe Reproduzierbarkeit
        self._validate_reproducibility(target, report)

        # Prüfe Dokumentation
        self._validate_documentation(target, report)

        return report

    def _validate_metadata(
        self, experiment: ExperimentSchema, report: ValidationReport
    ) -> None:
        """Validiert Experiment-Metadaten."""

        if not experiment.metadata.author:
            report.add_warning(
                message="Kein Autor angegeben",
                code="SCI_NO_AUTHOR",
                location="metadata.author",
                suggestion="Füge einen Autor für Nachvollziehbarkeit hinzu",
            )

        if not experiment.metadata.description:
            report.add_warning(
                message="Keine Beschreibung vorhanden",
                code="SCI_NO_DESCRIPTION",
                location="metadata.description",
                suggestion="Beschreibe Ziel und Methodik des Experiments",
            )

        if not experiment.metadata.version:
            report.add_warning(
                message="Keine Version angegeben",
                code="SCI_NO_VERSION",
                location="metadata.version",
                suggestion="Nutze semantische Versionierung (z.B. 1.0.0)",
            )

    def _validate_reproducibility(
        self, experiment: ExperimentSchema, report: ValidationReport
    ) -> None:
        """Validiert Reproduzierbarkeits-Aspekte."""

        # Prüfe auf undefinierte Parameter
        for step in experiment.steps:
            if step.inputs:
                for key, value in step.inputs.items():
                    if isinstance(value, str) and value.startswith("${"):
                        # Variable Reference
                        var_name = value[2:-1] if value.endswith("}") else value[2:]
                        param_names = {p.name for p in experiment.parameters}

                        if var_name not in param_names:
                            report.add_warning(
                                message=f"Undefinierte Variable: {var_name}",
                                code="SCI_UNDEFINED_VAR",
                                location=f"steps.{step.id}.inputs.{key}",
                                suggestion=f"Definiere '{var_name}' in parameters",
                            )

        # Prüfe auf Random Seeds
        has_random = any(
            "random" in str(s.config).lower() or "seed" in str(s.inputs).lower()
            for s in experiment.steps
            if hasattr(s, "config")
        )

        if has_random:
            has_seed_param = any(
                "seed" in p.name.lower() for p in experiment.parameters
            )
            if not has_seed_param:
                report.add_warning(
                    message="Experiment nutzt Zufallszahlen ohne definierten Seed",
                    code="SCI_NO_RANDOM_SEED",
                    suggestion="Definiere einen 'random_seed' Parameter",
                )

    def _validate_documentation(
        self, experiment: ExperimentSchema, report: ValidationReport
    ) -> None:
        """Validiert Dokumentation."""

        # Prüfe Step-Beschreibungen
        undocumented_steps = []
        for step in experiment.steps:
            # Steps sollten aussagekräftige IDs haben
            if len(step.id) < 3 or step.id in {"s1", "s2", "step1", "step2"}:
                undocumented_steps.append(step.id)

        if undocumented_steps:
            report.add_warning(
                message=f"Steps mit nicht-aussagekräftigen IDs: {undocumented_steps}",
                code="SCI_POOR_STEP_NAMES",
                suggestion="Nutze beschreibende Step-IDs wie 'load_data', 'train_model'",
            )

        # Prüfe Output-Dokumentation
        if not experiment.outputs:
            report.add_warning(
                message="Keine Outputs dokumentiert",
                code="SCI_NO_OUTPUTS",
                location="outputs",
                suggestion="Dokumentiere erwartete Experiment-Outputs",
            )
