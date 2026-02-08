#!/usr/bin/env python3
"""
SCIO - OpenAPI Documentation
Generiert OpenAPI 3.0 Spezifikation und Swagger UI
"""

from flask import Blueprint, jsonify, render_template_string
from typing import Dict, Any

openapi_bp = Blueprint('openapi', __name__)

# OpenAPI 3.0 Specification
OPENAPI_SPEC: Dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {
        "title": "SCIO AI-Workstation API",
        "description": """
## SCIO - Self-Coding Intelligence Orchestrator

Vollautomatische AI-Service-Plattform mit:
- **LLM Inference** - OpenAI-kompatible API
- **Image/Video Generation** - SDXL, Flux, CogVideoX
- **Decision Engine** - Intelligente Entscheidungsfindung
- **Learning** - Reinforcement Learning & Continuous Learning
- **Orchestration** - Event Bus, Workflows, Multi-Agent
- **100.000+ Tools** - Capability Engine

### Authentifizierung
API-Key im Header: `X-API-Key: your-api-key`
        """,
        "version": "2.0.0",
        "contact": {
            "name": "SCIO Support",
            "url": "https://scio.ai"
        },
        "license": {
            "name": "MIT"
        }
    },
    "servers": [
        {"url": "http://localhost:5000", "description": "Local Development"},
    ],
    "tags": [
        {"name": "Health", "description": "Health Checks und System-Status"},
        {"name": "Jobs", "description": "Job-Verwaltung und Ausführung"},
        {"name": "LLM", "description": "Large Language Model Inference"},
        {"name": "Images", "description": "Bildgenerierung und -verarbeitung"},
        {"name": "Decision", "description": "Decision Engine"},
        {"name": "Learning", "description": "RL Agent und Continuous Learning"},
        {"name": "Planning", "description": "AI Planner"},
        {"name": "Knowledge", "description": "Knowledge Graph"},
        {"name": "Agents", "description": "Multi-Agent System"},
        {"name": "Orchestration", "description": "Event Bus und Workflows"},
        {"name": "Capabilities", "description": "100.000+ Tools"},
        {"name": "Stats", "description": "Dashboard und Statistiken"},
        {"name": "Config", "description": "Konfiguration"},
    ],
    "paths": {
        # ─── Health Endpoints ─────────────────────────────────────────
        "/health": {
            "get": {
                "tags": ["Health"],
                "summary": "Health Check",
                "description": "Gibt Systemstatus zurück",
                "responses": {
                    "200": {
                        "description": "System healthy",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/healthz": {
            "get": {
                "tags": ["Health"],
                "summary": "Liveness Probe",
                "description": "Kubernetes Liveness Probe",
                "responses": {
                    "200": {"description": "Alive"},
                    "503": {"description": "Not alive"}
                }
            }
        },
        "/readyz": {
            "get": {
                "tags": ["Health"],
                "summary": "Readiness Probe",
                "description": "Kubernetes Readiness Probe",
                "responses": {
                    "200": {"description": "Ready"},
                    "503": {"description": "Not ready"}
                }
            }
        },
        "/metrics": {
            "get": {
                "tags": ["Health"],
                "summary": "Prometheus Metrics",
                "description": "Prometheus-kompatible Metriken",
                "responses": {
                    "200": {
                        "description": "Metrics im Prometheus-Format",
                        "content": {"text/plain": {}}
                    }
                }
            }
        },

        # ─── Jobs Endpoints ───────────────────────────────────────────
        "/api/jobs": {
            "post": {
                "tags": ["Jobs"],
                "summary": "Job erstellen",
                "description": "Erstellt einen neuen Job zur Ausführung",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/JobCreateRequest"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Job erstellt",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/JobResponse"}
                            }
                        }
                    }
                }
            },
            "get": {
                "tags": ["Jobs"],
                "summary": "Jobs auflisten",
                "description": "Gibt Liste aller Jobs zurück",
                "parameters": [
                    {"name": "status", "in": "query", "schema": {"type": "string"}},
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}}
                ],
                "responses": {
                    "200": {"description": "Job-Liste"}
                }
            }
        },
        "/api/jobs/{job_id}": {
            "get": {
                "tags": ["Jobs"],
                "summary": "Job-Status abrufen",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {"description": "Job-Details"},
                    "404": {"description": "Job nicht gefunden"}
                }
            }
        },

        # ─── LLM Endpoints ────────────────────────────────────────────
        "/v1/chat/completions": {
            "post": {
                "tags": ["LLM"],
                "summary": "Chat Completion",
                "description": "OpenAI-kompatible Chat Completion API",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ChatCompletionRequest"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Chat Response",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ChatCompletionResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/v1/completions": {
            "post": {
                "tags": ["LLM"],
                "summary": "Text Completion",
                "description": "OpenAI-kompatible Completion API",
                "responses": {"200": {"description": "Completion Response"}}
            }
        },

        # ─── Image Endpoints ──────────────────────────────────────────
        "/v1/images/generations": {
            "post": {
                "tags": ["Images"],
                "summary": "Bild generieren",
                "description": "Generiert Bild aus Text-Prompt",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ImageGenerationRequest"}
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Generiertes Bild"}
                }
            }
        },

        # ─── Decision Engine ──────────────────────────────────────────
        "/api/ai/decision/decide": {
            "post": {
                "tags": ["Decision"],
                "summary": "Entscheidung treffen",
                "description": "Trifft Entscheidung basierend auf Kontext",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DecisionRequest"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Entscheidung",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DecisionResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/ai/decision/best-action": {
            "post": {
                "tags": ["Decision"],
                "summary": "Beste Aktion ermitteln",
                "description": "Kombiniert alle Entscheidungsquellen",
                "responses": {"200": {"description": "Beste Aktion"}}
            }
        },

        # ─── Learning ─────────────────────────────────────────────────
        "/api/ai/learning/record": {
            "post": {
                "tags": ["Learning"],
                "summary": "Observation aufzeichnen",
                "description": "Zeichnet eine Beobachtung für das Lernen auf",
                "responses": {"200": {"description": "Erfolgreich"}}
            }
        },
        "/api/ai/learning/feedback": {
            "post": {
                "tags": ["Learning"],
                "summary": "Feedback geben",
                "description": "Gibt Feedback zu einer Aktion",
                "responses": {"200": {"description": "Feedback verarbeitet"}}
            }
        },

        # ─── Orchestration ────────────────────────────────────────────
        "/api/orchestration/status": {
            "get": {
                "tags": ["Orchestration"],
                "summary": "Orchestrator Status",
                "description": "Gibt Status aller Module zurück",
                "responses": {"200": {"description": "Status"}}
            }
        },
        "/api/orchestration/process": {
            "post": {
                "tags": ["Orchestration"],
                "summary": "Request orchestrieren",
                "description": "Verarbeitet einen Request durch alle Module",
                "responses": {"200": {"description": "Ergebnis"}}
            }
        },
        "/api/orchestration/workflow": {
            "post": {
                "tags": ["Orchestration"],
                "summary": "Workflow erstellen",
                "description": "Erstellt und startet einen Workflow",
                "responses": {"200": {"description": "Workflow gestartet"}}
            }
        },

        # ─── Capabilities ─────────────────────────────────────────────
        "/api/capabilities/tools": {
            "get": {
                "tags": ["Capabilities"],
                "summary": "Tools auflisten",
                "description": "Gibt alle verfügbaren Tools zurück",
                "parameters": [
                    {"name": "category", "in": "query", "schema": {"type": "string"}},
                    {"name": "search", "in": "query", "schema": {"type": "string"}}
                ],
                "responses": {"200": {"description": "Tool-Liste"}}
            }
        },
        "/api/capabilities/tools/{tool_id}": {
            "get": {
                "tags": ["Capabilities"],
                "summary": "Tool-Details",
                "parameters": [
                    {"name": "tool_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {"200": {"description": "Tool-Info"}}
            }
        },
        "/api/capabilities/execute/{tool_id}": {
            "post": {
                "tags": ["Capabilities"],
                "summary": "Tool ausführen",
                "parameters": [
                    {"name": "tool_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {"200": {"description": "Ergebnis"}}
            }
        },

        # ─── Stats ────────────────────────────────────────────────────
        "/api/stats": {
            "get": {
                "tags": ["Stats"],
                "summary": "Dashboard Stats",
                "description": "Aggregierte System-Statistiken",
                "responses": {
                    "200": {
                        "description": "Stats",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DashboardStats"}
                            }
                        }
                    }
                }
            }
        },
        "/api/stats/queue": {
            "get": {
                "tags": ["Stats"],
                "summary": "Queue Stats",
                "responses": {"200": {"description": "Queue-Statistiken"}}
            }
        },
        "/api/stats/hardware": {
            "get": {
                "tags": ["Stats"],
                "summary": "Hardware Stats",
                "responses": {"200": {"description": "Hardware-Statistiken"}}
            }
        },
    },

    "components": {
        "schemas": {
            "HealthResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "healthy"},
                    "version": {"type": "string", "example": "2.0.0"},
                    "gpu_count": {"type": "integer"},
                    "is_busy": {"type": "boolean"}
                }
            },
            "JobCreateRequest": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"type": "string", "enum": ["llm_inference", "image_generation", "video_generation"]},
                    "priority": {"type": "integer", "default": 0},
                    "input": {"type": "object"}
                }
            },
            "JobResponse": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "status": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"}
                }
            },
            "ChatCompletionRequest": {
                "type": "object",
                "required": ["messages"],
                "properties": {
                    "model": {"type": "string", "default": "mistral-7b"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                "content": {"type": "string"}
                            }
                        }
                    },
                    "max_tokens": {"type": "integer", "default": 2048},
                    "temperature": {"type": "number", "default": 0.7},
                    "stream": {"type": "boolean", "default": False}
                }
            },
            "ChatCompletionResponse": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "object": {"type": "string", "example": "chat.completion"},
                    "choices": {"type": "array"},
                    "usage": {"type": "object"}
                }
            },
            "ImageGenerationRequest": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string", "default": "flux-schnell"},
                    "size": {"type": "string", "default": "1024x1024"},
                    "n": {"type": "integer", "default": 1}
                }
            },
            "DecisionRequest": {
                "type": "object",
                "properties": {
                    "tree": {"type": "string", "default": "default"},
                    "context": {"type": "object"}
                }
            },
            "DecisionResponse": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "array", "items": {"type": "string"}}
                }
            },
            "DashboardStats": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "uptime_seconds": {"type": "integer"},
                    "queue": {"type": "object"},
                    "hardware": {"type": "object"},
                    "api": {"type": "object"},
                    "ai_modules": {"type": "object"},
                    "health": {"type": "object"}
                }
            }
        },
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            }
        }
    },
    "security": [
        {"ApiKeyAuth": []}
    ]
}


@openapi_bp.route('/api/openapi.json')
def openapi_spec():
    """OpenAPI 3.0 Spezifikation"""
    return jsonify(OPENAPI_SPEC)


@openapi_bp.route('/api/docs')
@openapi_bp.route('/api/docs/')
def swagger_ui():
    """Swagger UI"""
    return render_template_string(SWAGGER_UI_HTML)


# Swagger UI HTML Template
SWAGGER_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCIO API Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css">
    <style>
        body { margin: 0; padding: 0; }
        .swagger-ui .topbar { display: none; }
        .swagger-ui .info .title { color: #1e88e5; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: "/api/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                layout: "StandaloneLayout",
                persistAuthorization: true
            });
        };
    </script>
</body>
</html>
"""
