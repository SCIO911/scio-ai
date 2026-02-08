# SCIO Architektur

## Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI / API                            │
├─────────────────────────────────────────────────────────────┤
│                    Execution Engine                          │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Scheduler│  │  Sandbox  │  │ Context  │  │ Checkpoint│  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Agent System                            │
│  ┌──────────────────────┐  ┌────────────────────────────┐  │
│  │    Agent Registry    │  │     Builtin Agents         │  │
│  │  - register_agent()  │  │  - DataLoader              │  │
│  │  - create()          │  │  - Analyzer                │  │
│  │  - list_types()      │  │  - Reporter                │  │
│  └──────────────────────┘  │  - Transformer             │  │
│                            │  - LLM                     │  │
│                            └────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                       Tool System                            │
│  ┌──────────────────────┐  ┌────────────────────────────┐  │
│  │    Tool Registry     │  │      Builtin Tools         │  │
│  │  - register_tool()   │  │  - FileReader/Writer       │  │
│  │  - create()          │  │  - HttpClient              │  │
│  │  - get_schemas()     │  │  - PythonExecutor          │  │
│  └──────────────────────┘  │  - Shell, Math             │  │
│                            └────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Validation Layer                         │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │ YAML Parser  │  │ Schema Validator│  │ Security Check│  │
│  └──────────────┘  └─────────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Core / Config                           │
│  ┌────────┐  ┌────────────┐  ┌─────────┐  ┌─────────────┐  │
│  │ Config │  │ Exceptions │  │ Logging │  │   Utils     │  │
│  └────────┘  └────────────┘  └─────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Kernkomponenten

### 1. Core (`scio/core/`)

- **config.py**: Pydantic-basierte Konfiguration mit Umgebungsvariablen
- **exceptions.py**: Hierarchische Exception-Klassen
- **logging.py**: Strukturiertes Logging mit structlog
- **utils.py**: Hilfsfunktionen (ID-Generierung, Hashing, etc.)

### 2. Parser (`scio/parser/`)

- **yaml_parser.py**: Sicherer YAML-Parser mit Custom Tags
- **schema.py**: Pydantic-Schemas für Experimente
- **validators.py**: Erweiterte Validierungslogik

### 3. Validation (`scio/validation/`)

- **base.py**: Validator-Basisklassen und ValidationChain
- **scientific.py**: Wissenschaftliche Validierung
- **security.py**: Sicherheitsvalidierung

### 4. Execution (`scio/execution/`)

- **engine.py**: Haupt-Ausführungsengine
- **sandbox.py**: Sichere Ausführungsumgebung
- **scheduler.py**: Task-Scheduling mit Prioritäten

### 5. Agents (`scio/agents/`)

- **base.py**: Agent-Basisklasse mit Lifecycle
- **registry.py**: Agent-Registrierung
- **builtin/**: Mitgelieferte Agenten

### 6. Tools (`scio/tools/`)

- **base.py**: Tool-Basisklasse
- **registry.py**: Tool-Registrierung
- **builtin/**: Mitgelieferte Tools

### 7. Memory (`scio/memory/`)

- **store.py**: Key-Value Store mit TTL
- **context.py**: Ausführungskontext

### 8. Plugins (`scio/plugins/`)

- **base.py**: Plugin-Basisklasse
- **loader.py**: Plugin-Laden und -Verwaltung

### 9. API (`scio/api/`)

- **app.py**: FastAPI-Anwendung
- **routes.py**: REST-Endpunkte

### 10. CLI (`scio/cli/`)

- **main.py**: Typer-basierte CLI

## Datenfluss

1. **Experiment laden**: YAML → Parser → Schema
2. **Validierung**: Schema → ValidationChain → Report
3. **Ausführung**: Schema → Engine → Scheduler → Agents/Tools
4. **Ergebnisse**: Context → Memory → Output

## Erweiterbarkeit

### Eigene Agenten

```python
from scio.agents import Agent, register_agent

@register_agent("my_agent")
class MyAgent(Agent):
    async def execute(self, input_data, context):
        return {"result": "done"}
```

### Eigene Tools

```python
from scio.tools import Tool, register_tool

@register_tool("my_tool")
class MyTool(Tool):
    async def execute(self, input_data):
        return {"output": input_data}
```

### Plugins

```python
from scio.plugins import Plugin, PluginMetadata

class MyPlugin(Plugin):
    metadata = PluginMetadata(name="my_plugin", version="1.0")

    def on_load(self):
        self.register_agent("plugin_agent", MyAgentClass)
        self.register_tool("plugin_tool", MyToolClass)
```

## Sicherheit

- **Sandbox**: Eingeschränkte Ausführungsumgebung
- **Blocked Modules**: Gefährliche Python-Module blockiert
- **Path Validation**: Nur erlaubte Pfade zugänglich
- **Network Control**: Netzwerkzugriff kontrollierbar
- **Resource Limits**: Speicher- und CPU-Limits
