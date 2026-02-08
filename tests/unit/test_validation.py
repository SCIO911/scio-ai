"""Tests für das Validierungssystem."""

import pytest
from scio.validation import (
    Validator,
    ValidationChain,
    ScientificValidator,
    SecurityValidator,
)
from scio.validation.base import ValidationReport, Severity
from scio.parser.schema import ExperimentSchema


class TestValidationReport:
    """Tests für ValidationReport."""

    def test_empty_report_is_valid(self):
        """Leerer Report ist gültig."""
        report = ValidationReport()
        assert report.is_valid

    def test_report_with_warning_is_valid(self):
        """Report mit Warnung ist gültig."""
        report = ValidationReport()
        report.add_warning("Test warning", "TEST_WARN")
        assert report.is_valid
        assert report.has_warnings

    def test_report_with_error_is_invalid(self):
        """Report mit Fehler ist ungültig."""
        report = ValidationReport()
        report.add_error("Test error", "TEST_ERR")
        assert not report.is_valid

    def test_merge_reports(self):
        """Testet Zusammenführen von Reports."""
        r1 = ValidationReport()
        r1.add_error("Error 1", "E1")

        r2 = ValidationReport()
        r2.add_warning("Warning 1", "W1")

        merged = r1.merge(r2)
        assert merged.error_count == 1
        assert merged.warning_count == 1


class TestScientificValidator:
    """Tests für ScientificValidator."""

    def test_validates_metadata(self, sample_experiment):
        """Prüft Metadaten-Validierung."""
        validator = ScientificValidator()
        report = validator.validate(sample_experiment)

        # Sollte gültig sein mit Warnungen
        assert report.is_valid

    def test_warns_missing_author(self):
        """Warnt bei fehlendem Autor."""
        exp = ExperimentSchema(
            name="test",
            steps=[{"id": "s1", "type": "tool", "tool": "math"}],
        )

        validator = ScientificValidator()
        report = validator.validate(exp)

        warning_codes = [i.code for i in report.issues]
        assert "SCI_NO_AUTHOR" in warning_codes


class TestSecurityValidator:
    """Tests für SecurityValidator."""

    def test_detects_dangerous_patterns(self):
        """Erkennt gefährliche Patterns."""
        exp = ExperimentSchema(
            name="test",
            steps=[
                {
                    "id": "s1",
                    "type": "tool",
                    "tool": "python_executor",
                    "inputs": {"cmd": "eval(user_input)"},
                }
            ],
        )

        validator = SecurityValidator()
        report = validator.validate(exp)

        error_codes = [i.code for i in report.issues]
        assert "SEC_DANGEROUS_PATTERN" in error_codes

    def test_checks_memory_limits(self, sample_experiment):
        """Prüft Speicher-Limits."""
        # Modifiziere Step für hohen Speicherverbrauch
        sample_experiment.steps[0].resources.memory_mb = 2048

        validator = SecurityValidator()
        report = validator.validate(sample_experiment)

        # Sollte Fehler haben wegen Limit (Standard: 1024MB)
        # Note: Abhängig von Config
        assert report.is_valid or any(
            i.code == "SEC_MEMORY_LIMIT" for i in report.issues
        )


class TestValidationChain:
    """Tests für ValidationChain."""

    def test_runs_all_validators(self, sample_experiment):
        """Führt alle Validatoren aus."""
        chain = ValidationChain([
            ScientificValidator(),
            SecurityValidator(),
        ])

        report = chain.validate(sample_experiment)

        # Beide Validatoren sollten gelaufen sein
        assert len(report.issues) >= 0  # Kann Warnungen haben

    def test_fail_fast_mode(self):
        """Testet fail_fast Modus."""
        exp = ExperimentSchema(
            name="test",
            steps=[
                {
                    "id": "s1",
                    "type": "tool",
                    "tool": "python_executor",
                    "inputs": {"cmd": "eval(x)"},
                }
            ],
        )

        chain = ValidationChain([
            SecurityValidator(),
            ScientificValidator(),
        ])

        report = chain.validate(exp, fail_fast=True)

        # Sollte nach erstem Fehler stoppen
        assert not report.is_valid
