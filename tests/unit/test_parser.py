"""Tests für den YAML Parser."""

import pytest
from scio.parser import YAMLParser, parse_experiment
from scio.parser.schema import ExperimentSchema
from scio.core.exceptions import ParsingError, ValidationError


class TestYAMLParser:
    """Tests für YAMLParser."""

    def test_parse_valid_file(self, temp_yaml_file):
        """Testet Parsen einer gültigen Datei."""
        parser = YAMLParser()
        data = parser.parse_file(temp_yaml_file)

        assert "name" in data
        assert "steps" in data

    def test_parse_nonexistent_file(self):
        """Testet Fehler bei nicht existierender Datei."""
        parser = YAMLParser()

        with pytest.raises(ParsingError):
            parser.parse_file("/nonexistent/file.yaml")

    def test_parse_invalid_yaml(self, tmp_path):
        """Testet Fehler bei ungültigem YAML."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("key: value\n  invalid: indent")

        parser = YAMLParser()
        with pytest.raises(ParsingError):
            parser.parse_file(invalid_file)

    def test_parse_experiment(self, temp_yaml_file):
        """Testet vollständiges Parsen eines Experiments."""
        experiment = parse_experiment(temp_yaml_file)

        assert isinstance(experiment, ExperimentSchema)
        assert experiment.name == "test_experiment"


class TestExperimentSchema:
    """Tests für ExperimentSchema."""

    def test_valid_schema(self, sample_experiment_data):
        """Testet gültiges Schema."""
        experiment = ExperimentSchema(**sample_experiment_data)

        assert experiment.name == "test_experiment"
        assert len(experiment.steps) == 1
        assert len(experiment.agents) == 1

    def test_missing_name(self):
        """Testet fehlendes Pflichtfeld."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ExperimentSchema(steps=[{"id": "s1", "type": "agent"}])

    def test_invalid_agent_reference(self):
        """Testet ungültige Agent-Referenz."""
        with pytest.raises(ValueError):
            ExperimentSchema(
                name="test",
                agents=[],
                steps=[{"id": "s1", "type": "agent", "agent": "nonexistent"}],
            )

    def test_execution_order(self, sample_experiment):
        """Testet Berechnung der Ausführungsreihenfolge."""
        order = sample_experiment.get_execution_order()

        assert order == ["step_1"]

    def test_circular_dependency_detection(self):
        """Testet Erkennung zyklischer Abhängigkeiten."""
        with pytest.raises(ValueError, match="Zyklische"):
            exp = ExperimentSchema(
                name="test",
                steps=[
                    {"id": "a", "type": "tool", "depends_on": ["b"]},
                    {"id": "b", "type": "tool", "depends_on": ["a"]},
                ],
            )
            exp.get_execution_order()
