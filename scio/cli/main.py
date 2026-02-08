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


@app.command()
def python(
    action: str = typer.Argument("explain", help="Aktion: explain, analyze, generate, debug, improve, search, modules"),
    query: str = typer.Argument("", help="Suchanfrage oder Code"),
    code_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Python-Datei zum Analysieren"),
) -> None:
    """Python-Experte: Erklaert, analysiert und generiert Python-Code."""
    import asyncio
    from scio.agents.builtin import PythonExpertAgent
    from scio.agents.base import AgentContext

    agent = PythonExpertAgent({"name": "python_expert"})
    ctx = AgentContext(agent_id="cli", execution_id="cli")

    # Code aus Datei laden
    code = ""
    if code_file and code_file.exists():
        code = code_file.read_text(encoding="utf-8")

    async def run_action():
        if action == "explain":
            if not query:
                console.print("[yellow]Gib ein Konzept an: scio python explain list[/yellow]")
                return
            result = await agent.execute({"action": "explain", "query": query}, ctx)
            console.print(f"\n[bold]Python: {query}[/bold]\n")
            for item in result.get("results", [])[:10]:
                console.print(f"  [cyan]{item['category']}[/cyan]: [green]{item['name']}[/green]")
                if "description" in item:
                    console.print(f"    {item['description']}")
                if "code" in item:
                    console.print(f"    [dim]{item['code'][:100]}...[/dim]")

        elif action == "analyze":
            if not code and not query:
                console.print("[yellow]Gib Code an: scio python analyze 'def foo(): pass' oder --file script.py[/yellow]")
                return
            code_to_analyze = code or query
            result = await agent.execute({"action": "analyze", "code": code_to_analyze}, ctx)
            console.print(f"\n[bold]Code-Analyse[/bold]\n")
            info = result.get("info", {})
            console.print(f"  Zeilen: {info.get('lines', 0)}, Funktionen: {info.get('functions', 0)}, Klassen: {info.get('classes', 0)}")
            console.print(f"  Gueltig: {OK if result.get('valid') else FAIL}")
            for issue in result.get("issues", []):
                color = "red" if issue.get("type") == "error" else "yellow"
                console.print(f"  [{color}]{issue.get('type', 'issue')}[/{color}]: {issue.get('message')}")
            for sug in result.get("suggestions", [])[:5]:
                console.print(f"  [dim]Vorschlag: {sug.get('message')}[/dim]")

        elif action == "generate":
            if not query:
                console.print("[yellow]Verfuegbare Templates: class, function, dataclass, decorator, async_function, unittest, cli, api_client, singleton, factory, observer[/yellow]")
                return
            result = await agent.execute({"action": "generate", "query": query}, ctx)
            if "code" in result:
                console.print(f"\n[bold]Template: {result.get('template_type')}[/bold]\n")
                console.print(result["code"])
            else:
                console.print(f"\n[yellow]{result.get('message')}[/yellow]")

        elif action == "debug":
            if not code and not query:
                console.print("[yellow]Gib Code und Fehler an: scio python debug 'code' --file error.py[/yellow]")
                return
            result = await agent.execute({
                "action": "debug",
                "code": code or "",
                "error": query or "",
            }, ctx)
            console.print(f"\n[bold]Debug-Hilfe[/bold]\n")
            for sug in result.get("suggestions", []):
                console.print(f"  {BULLET} {sug}")

        elif action == "improve":
            if not code and not query:
                console.print("[yellow]Gib Code an: scio python improve --file script.py[/yellow]")
                return
            result = await agent.execute({"action": "improve", "code": code or query}, ctx)
            console.print(f"\n[bold]Verbesserungsvorschlaege[/bold]\n")
            for imp in result.get("improvements", [])[:10]:
                console.print(f"  Zeile {imp.get('line', '?')}: {imp.get('suggestion')}")
                if "example" in imp:
                    console.print(f"    [dim]Beispiel: {imp['example']}[/dim]")

        elif action == "search":
            if not query:
                console.print("[yellow]Gib einen Suchbegriff an: scio python search async[/yellow]")
                return
            result = await agent.execute({"action": "search", "query": query}, ctx)
            console.print(f"\n[bold]Suche: {query} ({result.get('total', 0)} Ergebnisse)[/bold]\n")
            for item in result.get("results", [])[:15]:
                console.print(f"  [cyan]{item['category']}[/cyan]: [green]{item['key']}[/green]")
                if item.get("value"):
                    console.print(f"    {item['value'][:80]}")

        elif action == "modules":
            result = await agent.execute({"action": "list_modules"}, ctx)
            console.print(f"\n[bold]Python Stdlib Module ({result.get('total', 0)})[/bold]\n")
            modules = result.get("modules", [])
            # In Spalten anzeigen
            cols = 4
            for i in range(0, len(modules), cols):
                row = modules[i:i+cols]
                console.print("  " + "  ".join(f"[cyan]{m:15}[/cyan]" for m in row))

        else:
            console.print(f"[red]Unbekannte Aktion: {action}[/red]")
            console.print("Verfuegbar: explain, analyze, generate, debug, improve, search, modules")

    asyncio.run(run_action())
    console.print()


if __name__ == "__main__":
    app()
