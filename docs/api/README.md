# SCIO API Referenz

## REST API

Die SCIO REST API ist unter `/api/v1` verfügbar.

### Basis-URL

```
http://localhost:8000/api/v1
```

### Endpunkte

#### System

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| GET | `/health` | Health Check |
| GET | `/info` | System-Informationen |

#### Experiments

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| POST | `/experiments/validate` | Experiment validieren |
| POST | `/experiments/run` | Experiment starten |
| POST | `/experiments/upload` | YAML hochladen |

#### Agents

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| GET | `/agents` | Alle Agenten auflisten |

#### Tools

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| GET | `/tools` | Alle Tools auflisten |
| POST | `/tools/{name}/execute` | Tool direkt ausführen |

## Beispiele

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development"
}
```

### Experiment validieren

```bash
curl -X POST http://localhost:8000/api/v1/experiments/validate \
  -H "Content-Type: application/json" \
  -d '{
    "experiment": {
      "name": "test",
      "version": "1.0",
      "steps": [{"id": "s1", "type": "tool"}]
    }
  }'
```

### Tool ausführen

```bash
curl -X POST http://localhost:8000/api/v1/tools/math/execute \
  -H "Content-Type: application/json" \
  -d '{"operation": "add", "a": 5, "b": 3}'
```

Response:
```json
{
  "success": true,
  "output": {
    "result": 8,
    "operation": "add"
  }
}
```

## OpenAPI

Die vollständige OpenAPI-Spezifikation ist verfügbar unter:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
