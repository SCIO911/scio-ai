"""
SCIO Schema Definitions

Pydantic-Schemas für Experimente, Agenten und Workflows.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class StepType(str, Enum):
    """Typen von Experiment-Schritten."""

    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    PARALLEL = "parallel"
    CHECKPOINT = "checkpoint"


class ResourceSpec(BaseModel):
    """Ressourcen-Spezifikation für einen Schritt."""

    memory_mb: int = Field(default=512, ge=64, le=16384)
    cpu_cores: float = Field(default=1.0, ge=0.1, le=32)
    timeout_seconds: int = Field(default=300, ge=1, le=86400)
    gpu: bool = Field(default=False)


class ParameterSchema(BaseModel):
    """Schema für einen einzelnen Parameter."""

    name: str = Field(..., min_length=1, max_length=64)
    type: str = Field(default="string")
    required: bool = Field(default=False)
    default: Optional[Any] = None
    description: Optional[str] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = {"string", "int", "float", "bool", "list", "dict", "path"}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid parameter type: {v}")
        return v.lower()


class AgentSchema(BaseModel):
    """Schema für einen Agenten."""

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    type: str = Field(..., description="Agent-Typ aus Registry")
    name: Optional[str] = Field(default=None, max_length=128)
    description: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    parameters: list[ParameterSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_name_default(self) -> "AgentSchema":
        if self.name is None:
            self.name = self.id
        return self


class StepSchema(BaseModel):
    """Schema für einen Experiment-Schritt."""

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    type: StepType = Field(default=StepType.AGENT)
    agent: Optional[str] = Field(default=None, description="Referenz auf Agent-ID")
    action: Optional[str] = Field(default=None, description="Auszuführende Aktion")
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    retry: int = Field(default=0, ge=0, le=5)
    condition: Optional[str] = Field(default=None, description="Ausführungsbedingung")

    @model_validator(mode="after")
    def validate_agent_step(self) -> "StepSchema":
        if self.type == StepType.AGENT and not self.agent:
            raise ValueError("Agent-Schritte benötigen eine Agent-Referenz")
        return self


class MetadataSchema(BaseModel):
    """Metadaten für ein Experiment."""

    author: Optional[str] = None
    created: Optional[datetime] = None
    version: str = Field(default="1.0")
    tags: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    license: Optional[str] = None
    repository: Optional[str] = None


class ExperimentSchema(BaseModel):
    """
    Hauptschema für ein SCIO-Experiment.

    Definiert die vollständige Struktur eines wissenschaftlichen Experiments
    mit Agenten, Schritten und Konfiguration.
    """

    name: str = Field(..., min_length=1, max_length=128)
    version: str = Field(default="1.0", pattern=r"^\d+\.\d+(\.\d+)?$")
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

    # Experiment-Komponenten
    agents: list[AgentSchema] = Field(default_factory=list)
    steps: list[StepSchema] = Field(..., min_length=1)

    # Globale Konfiguration
    config: dict[str, Any] = Field(default_factory=dict)
    parameters: list[ParameterSchema] = Field(default_factory=list)

    # Outputs
    outputs: dict[str, str] = Field(
        default_factory=dict, description="Mapping von Output-Namen zu Beschreibungen"
    )

    @model_validator(mode="after")
    def validate_references(self) -> "ExperimentSchema":
        """Validiert alle Referenzen zwischen Komponenten."""
        agent_ids = {a.id for a in self.agents}
        step_ids = {s.id for s in self.steps}

        for step in self.steps:
            # Prüfe Agent-Referenzen
            if step.agent and step.agent not in agent_ids:
                raise ValueError(
                    f"Step '{step.id}' referenziert unbekannten Agent: {step.agent}"
                )

            # Prüfe Abhängigkeiten
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step '{step.id}' hat unbekannte Abhängigkeit: {dep}"
                    )

        return self

    def get_execution_order(self) -> list[str]:
        """
        Berechnet die Ausführungsreihenfolge basierend auf Abhängigkeiten.

        Returns:
            Liste von Step-IDs in Ausführungsreihenfolge
        """
        # Topologische Sortierung
        in_degree: dict[str, int] = {s.id: 0 for s in self.steps}
        graph: dict[str, list[str]] = {s.id: [] for s in self.steps}

        for step in self.steps:
            for dep in step.depends_on:
                graph[dep].append(step.id)
                in_degree[step.id] += 1

        # Starte mit Steps ohne Abhängigkeiten
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.steps):
            raise ValueError("Zyklische Abhängigkeiten in Steps gefunden")

        return result
