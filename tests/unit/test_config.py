"""Tests für die Konfiguration."""

import pytest
from scio.core.config import Config, LoggingConfig, ExecutionConfig


class TestConfig:
    """Tests für Config-Klasse."""

    def test_default_config(self):
        """Testet Standard-Konfiguration."""
        config = Config()

        assert config.environment == "development"
        assert config.debug is False
        assert config.logging.level == "INFO"

    def test_custom_config(self):
        """Testet benutzerdefinierte Konfiguration."""
        config = Config(
            environment="production",
            debug=True,
            logging=LoggingConfig(level="DEBUG"),
        )

        assert config.environment == "production"
        assert config.debug is True
        assert config.logging.level == "DEBUG"

    def test_invalid_environment(self):
        """Testet ungültige Umgebung."""
        with pytest.raises(ValueError):
            Config(environment="invalid")

    def test_invalid_log_level(self):
        """Testet ungültiges Log-Level."""
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")


class TestExecutionConfig:
    """Tests für ExecutionConfig."""

    def test_defaults(self):
        """Testet Standard-Werte."""
        config = ExecutionConfig()

        assert config.max_concurrent_agents == 4
        assert config.default_timeout == 300
        assert config.sandbox_enabled is True

    def test_validation(self):
        """Testet Validierung."""
        with pytest.raises(ValueError):
            ExecutionConfig(max_concurrent_agents=0)

        with pytest.raises(ValueError):
            ExecutionConfig(max_concurrent_agents=100)
