"""
SCIO Test Configuration

Pytest Fixtures und Konfiguration.
"""

import pytest
from pathlib import Path
from typing import Any

from scio.core.config import Config, set_config
from scio.core.logging import setup_logging
from scio.parser.schema import ExperimentSchema, StepSchema, AgentSchema


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Initialisiert die Testumgebung."""
    config = Config(
        environment="testing",
        debug=True,
        logging={"level": "DEBUG", "format": "human"},
    )
    set_config(config)
    setup_logging()
    yield


@pytest.fixture
def sample_experiment_data() -> dict[str, Any]:
    """Beispiel-Experiment-Daten."""
    return {
        "name": "test_experiment",
        "version": "1.0",
        "metadata": {
            "author": "Test Author",
            "description": "Ein Test-Experiment",
        },
        "agents": [
            {
                "id": "test_agent",
                "type": "base",
                "name": "Test Agent",
            }
        ],
        "steps": [
            {
                "id": "step_1",
                "type": "agent",
                "agent": "test_agent",
                "inputs": {"value": 42},
            }
        ],
    }


@pytest.fixture
def sample_experiment(sample_experiment_data) -> ExperimentSchema:
    """Beispiel ExperimentSchema."""
    return ExperimentSchema(**sample_experiment_data)


@pytest.fixture
def temp_yaml_file(tmp_path, sample_experiment_data) -> Path:
    """Erstellt eine temporäre YAML-Datei."""
    import yaml

    file_path = tmp_path / "test_experiment.yaml"
    with open(file_path, "w") as f:
        yaml.dump(sample_experiment_data, f)
    return file_path


@pytest.fixture
def invalid_yaml_file(tmp_path) -> Path:
    """Erstellt eine ungültige YAML-Datei."""
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text("name: test\nsteps: not_a_list\n")
    return file_path
