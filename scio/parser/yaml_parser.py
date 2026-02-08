"""
SCIO YAML Parser

Sicherer YAML-Parser mit Validierung und Fehlerbehandlung.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError as PydanticValidationError

from scio.core.exceptions import ParsingError, ValidationError
from scio.core.logging import get_logger
from scio.parser.schema import ExperimentSchema

logger = get_logger(__name__)


class SafeLoader(yaml.SafeLoader):
    """Erweiterter SafeLoader mit SCIO-spezifischen Tags."""

    pass


def _env_constructor(loader: yaml.Loader, node: yaml.Node) -> str:
    """Handler für !env Tag - lädt Umgebungsvariablen."""
    import os

    value = loader.construct_scalar(node)
    return os.environ.get(value, "")


def _include_constructor(loader: yaml.Loader, node: yaml.Node) -> Any:
    """Handler für !include Tag - inkludiert andere YAML-Dateien."""
    filepath = Path(loader.construct_scalar(node))

    if not filepath.is_absolute():
        # Relativer Pfad zur aktuellen Datei
        base_path = Path(loader.name).parent if hasattr(loader, "name") else Path.cwd()
        filepath = base_path / filepath

    if not filepath.exists():
        raise ParsingError(f"Include-Datei nicht gefunden: {filepath}")

    with open(filepath) as f:
        return yaml.load(f, Loader=SafeLoader)


# Registriere Custom Tags
SafeLoader.add_constructor("!env", _env_constructor)
SafeLoader.add_constructor("!include", _include_constructor)


class YAMLParser:
    """
    YAML-Parser für SCIO-Experimente.

    Features:
    - Sichere YAML-Verarbeitung (kein arbitrary code execution)
    - Custom Tags (!env, !include)
    - Detaillierte Fehlermeldungen mit Zeilenangaben
    - Schema-Validierung
    """

    def __init__(self, strict: bool = True):
        """
        Initialisiert den Parser.

        Args:
            strict: Bei True werden Warnungen zu Fehlern
        """
        self.strict = strict
        self.logger = get_logger(__name__, component="yaml_parser")

    def parse_file(self, path: Path | str) -> dict[str, Any]:
        """
        Parst eine YAML-Datei.

        Args:
            path: Pfad zur YAML-Datei

        Returns:
            Geparstes Dictionary

        Raises:
            ParsingError: Bei YAML-Syntaxfehlern
        """
        path = Path(path)

        if not path.exists():
            raise ParsingError(f"Datei nicht gefunden: {path}")

        if not path.suffix.lower() in {".yaml", ".yml"}:
            self.logger.warning("Unerwartete Dateiendung", path=str(path))

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content, source=str(path))

        except Exception as e:
            if isinstance(e, ParsingError):
                raise
            raise ParsingError(f"Fehler beim Lesen der Datei: {e}")

    def parse_string(
        self, content: str, source: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Parst einen YAML-String.

        Args:
            content: YAML-Inhalt
            source: Optionale Quellenangabe für Fehlermeldungen

        Returns:
            Geparstes Dictionary

        Raises:
            ParsingError: Bei YAML-Syntaxfehlern
        """
        try:
            loader = SafeLoader(content)
            if source:
                loader.name = source

            result = loader.get_single_data()

            if result is None:
                return {}

            if not isinstance(result, dict):
                raise ParsingError("YAML-Root muss ein Dictionary sein")

            return result

        except yaml.YAMLError as e:
            line = getattr(e, "problem_mark", None)
            raise ParsingError(
                f"YAML-Syntaxfehler: {e}",
                line=line.line + 1 if line else None,
                column=line.column + 1 if line else None,
            )

    def parse_experiment(self, path: Path | str) -> ExperimentSchema:
        """
        Parst und validiert ein Experiment.

        Args:
            path: Pfad zur Experiment-YAML

        Returns:
            Validiertes ExperimentSchema

        Raises:
            ParsingError: Bei YAML-Fehlern
            ValidationError: Bei Schema-Validierungsfehlern
        """
        data = self.parse_file(path)
        return self.validate_experiment(data)

    def validate_experiment(self, data: dict[str, Any]) -> ExperimentSchema:
        """
        Validiert Experiment-Daten gegen das Schema.

        Args:
            data: Geparstes YAML-Dictionary

        Returns:
            Validiertes ExperimentSchema

        Raises:
            ValidationError: Bei Validierungsfehlern
        """
        try:
            return ExperimentSchema(**data)

        except PydanticValidationError as e:
            errors = []
            for error in e.errors():
                loc = ".".join(str(l) for l in error["loc"])
                errors.append(f"{loc}: {error['msg']}")

            raise ValidationError(
                f"Schema-Validierung fehlgeschlagen: {'; '.join(errors)}",
                details={"errors": e.errors()},
            )


# Convenience-Funktion
def parse_experiment(path: Path | str) -> ExperimentSchema:
    """
    Parst ein Experiment aus einer YAML-Datei.

    Args:
        path: Pfad zur YAML-Datei

    Returns:
        Validiertes ExperimentSchema
    """
    parser = YAMLParser()
    return parser.parse_experiment(path)
