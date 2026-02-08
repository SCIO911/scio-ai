"""
SCIO CLI

Kommandozeilen-Interface für SCIO.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from scio import __version__
from scio.core.config import Config, get_config, set_config
from scio.core.logging import setup_logging, get_logger
from scio.parser import parse_experiment, validate_experiment

app = typer.Typer(
    name="scio",
    help="SCIO - Scientific Intelligent Operations Framework",
    no_args_is_help=True,
)

console = Console(force_terminal=True, legacy_windows=True)

# ASCII-kompatible Symbole fuer Windows
OK = "[green][OK][/green]"
FAIL = "[red][FAIL][/red]"
WARN = "[yellow][WARN][/yellow]"
BULLET = "*"


def version_callback(value: bool) -> None:
    if value:
        console.print(f"SCIO version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Zeigt die Version an",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Debug-Modus aktivieren",
    ),
) -> None:
    """SCIO - Scientific Intelligent Operations Framework"""
    if debug:
        config = Config(debug=True, logging={"level": "DEBUG"})
        set_config(config)
    setup_logging()


@app.command()
def validate(
    experiment_file: Path = typer.Argument(
        ...,
        help="Pfad zur Experiment-YAML-Datei",
        exists=True,
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Strikte Validierung (Warnungen werden zu Fehlern)",
    ),
) -> None:
    """Validiert ein Experiment."""
    logger = get_logger(__name__)

    console.print(f"\n[bold]Validiere:[/bold] {experiment_file}\n")

    try:
        # Parse Experiment
        experiment = parse_experiment(experiment_file)
        console.print(f"{OK} YAML-Syntax korrekt")
        console.print(f"{OK} Schema-Validierung erfolgreich")

        # Erweiterte Validierung
        result = validate_experiment(experiment, strict=strict)

        if result.errors:
            console.print(f"\n{FAIL} {len(result.errors)} Fehler gefunden:")
            for error in result.errors:
                console.print(f"  {BULLET} {error}")

        if result.warnings:
            console.print(f"\n{WARN} {len(result.warnings)} Warnungen:")
            for warning in result.warnings:
                console.print(f"  {BULLET} {warning}")

        if result.is_valid:
            console.print(f"\n[green bold]{OK} Experiment ist gueltig[/green bold]\n")
        else:
            console.print(f"\n[red bold]{FAIL} Validierung fehlgeschlagen[/red bold]\n")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[red bold]Fehler:[/red bold] {e}\n")
        logger.exception("Validation failed")
        raise typer.Exit(1)


@app.command()
def run(
    experiment_file: Path = typer.Argument(
        ...,
        help="Pfad zur Experiment-YAML-Datei",
        exists=True,
    ),
    param: Optional[list[str]] = typer.Option(
        None,
        "--param",
        "-p",
        help="Parameter im Format key=value",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Nur validieren, nicht ausführen",
    ),
) -> None:
    """Führt ein Experiment aus."""
    logger = get_logger(__name__)

    console.print(f"\n[bold]Experiment:[/bold] {experiment_file}\n")

    # Parse Parameter
    parameters = {}
    if param:
        for p in param:
            if "=" in p:
                key, value = p.split("=", 1)
                parameters[key.strip()] = value.strip()

    try:
        experiment = parse_experiment(experiment_file)

        if dry_run:
            console.print("[yellow]Dry-Run Modus - keine Ausführung[/yellow]\n")

            # Zeige Experiment-Info
            table = Table(title="Experiment Details")
            table.add_column("Eigenschaft", style="cyan")
            table.add_column("Wert", style="green")

            table.add_row("Name", experiment.name)
            table.add_row("Version", experiment.version)
            table.add_row("Steps", str(len(experiment.steps)))
            table.add_row("Agents", str(len(experiment.agents)))

            console.print(table)
            console.print()

            # Zeige Ausführungsreihenfolge
            order = experiment.get_execution_order()
            console.print("[bold]Ausführungsreihenfolge:[/bold]")
            for i, step_id in enumerate(order, 1):
                console.print(f"  {i}. {step_id}")

            return

        # Echte Ausführung
        console.print("[bold]Starte Ausführung...[/bold]\n")
        console.print("[yellow]Execution Engine noch nicht vollständig implementiert[/yellow]")

    except Exception as e:
        console.print(f"\n[red bold]Fehler:[/red bold] {e}\n")
        logger.exception("Execution failed")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Zeigt Systeminformationen."""
    import sys
    import platform

    table = Table(title="SCIO System Info")
    table.add_column("Komponente", style="cyan")
    table.add_column("Version/Wert", style="green")

    table.add_row("SCIO", __version__)
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Platform", platform.system())
    table.add_row("Architecture", platform.machine())

    config = get_config()
    table.add_row("Environment", config.environment)
    table.add_row("Debug", str(config.debug))

    console.print()
    console.print(table)
    console.print()


@app.command()
def init(
    name: str = typer.Argument(..., help="Name des Experiments"),
    output: Path = typer.Option(
        Path("."),
        "--output",
        "-o",
        help="Ausgabeverzeichnis",
    ),
) -> None:
    """Erstellt ein neues Experiment-Template."""
    template = f'''# SCIO Experiment: {name}
# Erstellt mit SCIO v{__version__}

name: {name}
version: "1.0"

metadata:
  author: ""
  description: |
    Beschreibung des Experiments hier einfügen.
  tags:
    - experiment

parameters:
  - name: random_seed
    type: int
    default: 42
    description: Seed für Reproduzierbarkeit

agents:
  - id: main_agent
    type: base
    name: Hauptagent
    description: Beschreibung des Agenten

steps:
  - id: step_1
    type: agent
    agent: main_agent
    inputs:
      seed: "${{random_seed}}"
    outputs:
      - result

outputs:
  result: Ergebnis des Experiments
'''

    output_file = output / f"{name}.yaml"
    output_file.write_text(template, encoding="utf-8")

    console.print(f"\n{OK} Experiment erstellt: {output_file}\n")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host-Adresse"),
    port: int = typer.Option(8000, "--port", "-p", help="Port"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Auto-Reload bei Aenderungen"),
) -> None:
    """Startet den SCIO API Server."""
    console.print(f"\n[bold]SCIO API Server[/bold]")
    console.print(f"URL: http://{host}:{port}")
    console.print(f"Docs: http://{host}:{port}/docs\n")

    from scio.api.server import run_server
    run_server(host=host, port=port, reload=reload)


@app.command()
def agents() -> None:
    """Listet alle registrierten Agenten."""
    # Importiere Builtins
    import scio.agents.builtin  # noqa

    from scio.agents.registry import AgentRegistry

    table = Table(title="Registrierte Agenten")
    table.add_column("Typ", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Beschreibung")

    for agent_type in AgentRegistry.list_types():
        agent_class = AgentRegistry.get(agent_type)
        version = getattr(agent_class, "version", "1.0")
        doc = (agent_class.__doc__ or "").split("\n")[0].strip()
        table.add_row(agent_type, version, doc[:50])

    console.print()
    console.print(table)
    console.print()


@app.command()
def tools() -> None:
    """Listet alle registrierten Tools."""
    # Importiere Builtins
    import scio.tools.builtin  # noqa

    from scio.tools.registry import ToolRegistry

    table = Table(title="Registrierte Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Beschreibung")

    for tool_name in ToolRegistry.list_tools():
        tool = ToolRegistry.create(tool_name)
        version = getattr(tool, "version", "1.0")
        desc = tool.config.description or ""
        table.add_row(tool_name, version, desc[:50])

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
