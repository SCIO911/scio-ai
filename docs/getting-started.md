# SCIO - Getting Started

## Installation

### Basis-Installation

```bash
pip install -e .
```

### Mit optionalen Abhängigkeiten

```bash
# Mit API-Server
pip install -e ".[api]"

# Mit LLM-Unterstützung
pip install -e ".[llm]"

# Mit Daten-Tools (Parquet, Pandas)
pip install -e ".[data]"

# Alles
pip install -e ".[all]"

# Entwicklung
pip install -e ".[dev]"
```

## Erste Schritte

### 1. Experiment erstellen

```bash
scio init my_first_experiment
```

Dies erstellt eine `my_first_experiment.yaml` Datei.

### 2. Experiment validieren

```bash
scio validate my_first_experiment.yaml
```

### 3. Experiment ausführen (Dry-Run)

```bash
scio run my_first_experiment.yaml --dry-run
```

### 4. Verfügbare Agenten und Tools anzeigen

```bash
scio agents
scio tools
```

## Experiment-Struktur

Ein SCIO-Experiment besteht aus:

```yaml
name: mein_experiment
version: "1.0"

metadata:
  author: "Dein Name"
  description: "Beschreibung"
  tags: [experiment, test]

parameters:
  - name: seed
    type: int
    default: 42

agents:
  - id: loader
    type: data_loader
    config:
      encoding: utf-8

steps:
  - id: load_data
    type: agent
    agent: loader
    inputs:
      path: "./data/input.csv"
    outputs:
      - data

outputs:
  data: "Geladene Daten"
```

## API Server

Starte den API-Server:

```bash
scio serve --port 8000
```

API-Dokumentation: http://localhost:8000/docs

## Builtin Agents

| Agent | Beschreibung |
|-------|-------------|
| `data_loader` | Lädt Daten aus CSV, JSON, YAML, Parquet |
| `analyzer` | Statistische Analysen |
| `reporter` | Generiert Reports (JSON, Markdown, HTML) |
| `transformer` | Datentransformationen |
| `llm` | LLM-Integration (OpenAI, Anthropic) |

## Builtin Tools

| Tool | Beschreibung |
|------|-------------|
| `file_reader` | Liest Dateien |
| `file_writer` | Schreibt Dateien |
| `http_client` | HTTP-Anfragen |
| `python_executor` | Sichere Python-Ausführung |
| `shell` | Shell-Kommandos (eingeschränkt) |
| `math` | Mathematische Operationen |

## Konfiguration

SCIO kann über Umgebungsvariablen oder `.env` konfiguriert werden:

```bash
SCIO_ENVIRONMENT=production
SCIO_DEBUG=false
SCIO_LOGGING__LEVEL=INFO
SCIO_EXECUTION__MAX_CONCURRENT_AGENTS=4
SCIO_SECURITY__NETWORK_ENABLED=false
```

## Nächste Schritte

- [Architektur](architecture.md)
- [API Referenz](api/)
- [Beispiele](../examples/)
