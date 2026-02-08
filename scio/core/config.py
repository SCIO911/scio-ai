"""
SCIO Configuration Management

Zentrale Konfigurationsverwaltung mit Pydantic v2.
Unterstützt Umgebungsvariablen, YAML-Dateien und programmatische Konfiguration.
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseModel):
    """Konfiguration für das Logging-System."""

    level: str = Field(default="INFO", description="Log-Level (DEBUG, INFO, WARNING, ERROR)")
    format: str = Field(
        default="structured",
        description="Log-Format: 'structured' (JSON) oder 'human' (lesbar)",
    )
    file: Optional[Path] = Field(default=None, description="Optionale Log-Datei")
    rotation: str = Field(default="10 MB", description="Log-Rotation Größe")
    retention: str = Field(default="7 days", description="Log-Aufbewahrungszeit")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper


class ExecutionConfig(BaseModel):
    """Konfiguration für die Execution Engine."""

    max_concurrent_agents: int = Field(
        default=4, ge=1, le=32, description="Maximale parallele Agenten"
    )
    default_timeout: int = Field(
        default=300, ge=1, description="Standard-Timeout in Sekunden"
    )
    sandbox_enabled: bool = Field(
        default=True, description="Sandbox-Modus aktivieren"
    )
    checkpoint_interval: int = Field(
        default=60, ge=0, description="Checkpoint-Intervall in Sekunden (0 = deaktiviert)"
    )


class SecurityConfig(BaseModel):
    """Sicherheitskonfiguration."""

    allowed_paths: list[Path] = Field(
        default_factory=list, description="Erlaubte Dateisystem-Pfade"
    )
    blocked_modules: list[str] = Field(
        default_factory=lambda: ["os.system", "subprocess", "eval", "exec"],
        description="Blockierte Python-Module/Funktionen",
    )
    network_enabled: bool = Field(
        default=False, description="Netzwerkzugriff erlauben"
    )
    max_memory_mb: int = Field(
        default=1024, ge=64, description="Maximaler Speicher pro Agent in MB"
    )


class Config(BaseSettings):
    """
    Hauptkonfiguration für SCIO.

    Lädt Konfiguration aus:
    1. Umgebungsvariablen (Präfix: SCIO_)
    2. .env Datei
    3. Programmatische Überschreibungen
    """

    model_config = SettingsConfigDict(
        env_prefix="SCIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Allgemeine Einstellungen
    project_name: str = Field(default="scio-experiment", description="Projektname")
    environment: str = Field(
        default="development",
        description="Umgebung: development, testing, production",
    )
    debug: bool = Field(default=False, description="Debug-Modus")
    data_dir: Path = Field(
        default=Path("./data"), description="Datenverzeichnis"
    )

    # Nested Configs
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = {"development", "testing", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Lädt Konfiguration aus einer YAML-Datei."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Exportiert Konfiguration als Dictionary."""
        return self.model_dump()


# Globale Konfigurationsinstanz (Lazy Loading)
_config: Optional[Config] = None


def get_config() -> Config:
    """Gibt die globale Konfiguration zurück."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Setzt die globale Konfiguration."""
    global _config
    _config = config
