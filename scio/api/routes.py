"""
SCIO API Routes

REST-Endpunkte für die SCIO API.
"""

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field

from scio import __version__
from scio.agents.registry import AgentRegistry
from scio.core.config import get_config
from scio.core.logging import get_logger
from scio.execution.engine import ExecutionEngine, ExecutionResult
from scio.parser import parse_experiment, validate_experiment
from scio.parser.schema import ExperimentSchema
from scio.tools.registry import ToolRegistry

logger = get_logger(__name__)
router = APIRouter()

# Globale Engine-Instanz
_engine: Optional[ExecutionEngine] = None


def get_engine() -> ExecutionEngine:
    """Gibt die globale Execution Engine zurück."""
    global _engine
    if _engine is None:
        _engine = ExecutionEngine()
    return _engine


# --- Schemas ---


class HealthResponse(BaseModel):
    """Health Check Response."""

    status: str = "healthy"
    version: str
    environment: str


class ExperimentRequest(BaseModel):
    """Request zum Ausführen eines Experiments."""

    experiment: dict[str, Any] = Field(..., description="Experiment-Definition")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Parameter")


class ValidationRequest(BaseModel):
    """Request zur Validierung."""

    experiment: dict[str, Any] = Field(..., description="Experiment-Definition")
    strict: bool = Field(default=False, description="Strikte Validierung")


class ValidationResponse(BaseModel):
    """Validierungs-Response."""

    valid: bool
    errors: list[str] = []
    warnings: list[str] = []


class ExecutionResponse(BaseModel):
    """Ausführungs-Response."""

    execution_id: str
    status: str
    message: str


class AgentInfo(BaseModel):
    """Agent-Informationen."""

    type: str
    version: str
    description: Optional[str] = None


class ToolInfo(BaseModel):
    """Tool-Informationen."""

    name: str
    version: str
    description: Optional[str] = None
    schema: dict[str, Any]


# --- Routes ---


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health Check Endpunkt."""
    config = get_config()
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=config.environment,
    )


@router.get("/info", tags=["System"])
async def system_info() -> dict[str, Any]:
    """Gibt Systeminformationen zurück."""
    import platform
    import sys

    config = get_config()

    return {
        "scio_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "environment": config.environment,
        "debug": config.debug,
        "agents_registered": len(AgentRegistry.list_types()),
        "tools_registered": len(ToolRegistry.list_tools()),
    }


@router.post("/experiments/validate", response_model=ValidationResponse, tags=["Experiments"])
async def validate_experiment_endpoint(request: ValidationRequest) -> ValidationResponse:
    """Validiert eine Experiment-Definition."""
    try:
        # Parse Schema
        experiment = ExperimentSchema(**request.experiment)

        # Erweiterte Validierung
        result = validate_experiment(experiment, strict=request.strict)

        return ValidationResponse(
            valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
        )

    except Exception as e:
        logger.error("Validation error", error=str(e))
        return ValidationResponse(
            valid=False,
            errors=[str(e)],
        )


@router.post("/experiments/run", response_model=ExecutionResponse, tags=["Experiments"])
async def run_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks,
) -> ExecutionResponse:
    """Startet die Ausführung eines Experiments."""
    try:
        # Parse Experiment
        experiment = ExperimentSchema(**request.experiment)

        # Validiere
        validation = validate_experiment(experiment)
        if not validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Validierung fehlgeschlagen: {validation.errors}",
            )

        # Starte Ausführung im Hintergrund
        engine = get_engine()

        # Erstelle Task für Hintergrund-Ausführung
        async def execute_in_background():
            await engine.execute(experiment, request.parameters)

        background_tasks.add_task(execute_in_background)

        return ExecutionResponse(
            execution_id="pending",  # Wird durch Engine ersetzt
            status="started",
            message="Experiment wurde gestartet",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Execution error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/upload", tags=["Experiments"])
async def upload_experiment(file: UploadFile = File(...)) -> dict[str, Any]:
    """Lädt eine Experiment-YAML-Datei hoch und validiert sie."""
    if not file.filename or not file.filename.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="Nur YAML-Dateien erlaubt")

    try:
        content = await file.read()
        content_str = content.decode("utf-8")

        import yaml
        data = yaml.safe_load(content_str)

        experiment = ExperimentSchema(**data)
        validation = validate_experiment(experiment)

        return {
            "filename": file.filename,
            "valid": validation.is_valid,
            "experiment_name": experiment.name,
            "steps": len(experiment.steps),
            "agents": len(experiment.agents),
            "errors": validation.errors,
            "warnings": validation.warnings,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents", response_model=list[AgentInfo], tags=["Agents"])
async def list_agents() -> list[AgentInfo]:
    """Listet alle registrierten Agenten."""
    agents = []

    for agent_type in AgentRegistry.list_types():
        agent_class = AgentRegistry.get(agent_type)
        agents.append(
            AgentInfo(
                type=agent_type,
                version=getattr(agent_class, "version", "1.0"),
                description=agent_class.__doc__,
            )
        )

    return agents


@router.get("/tools", response_model=list[ToolInfo], tags=["Tools"])
async def list_tools() -> list[ToolInfo]:
    """Listet alle registrierten Tools."""
    tools = []

    for tool_name in ToolRegistry.list_tools():
        tool = ToolRegistry.create(tool_name)
        tools.append(
            ToolInfo(
                name=tool_name,
                version=getattr(tool, "version", "1.0"),
                description=tool.config.description,
                schema=tool.get_schema(),
            )
        )

    return tools


@router.post("/tools/{tool_name}/execute", tags=["Tools"])
async def execute_tool(tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Führt ein Tool direkt aus."""
    try:
        tool = ToolRegistry.create(tool_name)
        result = await tool.run(input_data)
        return result.to_dict()

    except Exception as e:
        logger.error("Tool execution error", tool=tool_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
