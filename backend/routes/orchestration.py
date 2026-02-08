#!/usr/bin/env python3
"""
SCIO - Orchestration API Routes
Routes für Orchestrator, Event Bus und Workflow Engine
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any

orch_bp = Blueprint('orchestration', __name__, url_prefix='/api/orchestration')


@orch_bp.route('/status', methods=['GET'])
def orchestration_status():
    """Gibt Orchestrator-Status zurück"""
    try:
        from backend.orchestration import get_orchestrator
        orch = get_orchestrator()

        stats = orch.get_statistics()

        return jsonify({
            "running": stats["running"],
            "initialized": stats["initialized"],
            "total_modules": stats["total_modules"],
            "modules_by_type": stats["modules_by_type"],
            "health": stats["health"]["overall_status"]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/health', methods=['GET'])
def orchestration_health():
    """Gibt detaillierte Health-Informationen zurück"""
    try:
        from backend.orchestration import get_orchestrator
        orch = get_orchestrator()

        health = orch.get_health_summary()

        return jsonify(health)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/modules', methods=['GET'])
def orchestration_modules():
    """Gibt alle registrierten Module zurück"""
    try:
        from backend.orchestration import get_orchestrator
        orch = get_orchestrator()

        modules = []
        for name, info in orch.modules.items():
            modules.append({
                "name": name,
                "type": info.module_type,
                "status": info.status.value,
                "errors": info.error_count,
                "successes": info.success_count,
                "last_health_check": info.last_health_check.isoformat() if info.last_health_check else None
            })

        return jsonify({
            "modules": modules,
            "count": len(modules)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/process', methods=['POST'])
def orchestration_process():
    """Verarbeitet eine Anfrage über den Orchestrator"""
    try:
        from backend.orchestration import get_orchestrator
        orch = get_orchestrator()

        data = request.json or {}
        request_type = data.get('type', 'analyze')
        request_data = data.get('data', {})

        result = orch.process_request(request_type, request_data)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/events/stats', methods=['GET'])
def events_stats():
    """Gibt Event Bus Statistiken zurück"""
    try:
        from backend.orchestration import get_event_bus
        bus = get_event_bus()

        stats = bus.get_statistics()

        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/events/history', methods=['GET'])
def events_history():
    """Gibt Event-History zurück"""
    try:
        from backend.orchestration import get_event_bus
        bus = get_event_bus()

        limit = request.args.get('limit', 100, type=int)
        source = request.args.get('source', None)

        events = bus.get_history(source=source, limit=limit)

        return jsonify({
            "events": [e.to_dict() for e in events],
            "count": len(events)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/events/emit', methods=['POST'])
def events_emit():
    """Emittiert ein benutzerdefiniertes Event"""
    try:
        from backend.orchestration import get_event_bus
        from backend.orchestration.event_bus import EventType
        bus = get_event_bus()

        data = request.json or {}
        event_type = data.get('type', 'custom')
        source = data.get('source', 'api')
        event_data = data.get('data', {})
        priority = data.get('priority', 0)

        # Event-Typ bestimmen
        try:
            evt_type = EventType(event_type)
        except ValueError:
            evt_type = EventType.CUSTOM

        event_id = bus.emit(evt_type, source, event_data, priority=priority)

        return jsonify({
            "event_id": event_id,
            "type": evt_type.value,
            "source": source
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/workflows', methods=['GET'])
def workflows_list():
    """Gibt verfügbare Workflows zurück"""
    try:
        from backend.orchestration import get_workflow_engine
        engine = get_workflow_engine()

        workflows = []
        for wf_id, wf in engine.workflows.items():
            workflows.append({
                "id": wf_id,
                "name": wf.name,
                "status": wf.status.value,
                "steps": len(wf.steps),
                "created_at": wf.created_at.isoformat()
            })

        # Predefined Workflows
        predefined = list(engine.predefined_workflows.keys())

        return jsonify({
            "workflows": workflows,
            "predefined": predefined,
            "active_count": len(workflows)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/workflows/execute', methods=['POST'])
def workflows_execute():
    """Führt einen Workflow aus"""
    try:
        from backend.orchestration import get_workflow_engine
        engine = get_workflow_engine()

        data = request.json or {}
        workflow_name = data.get('name', '')
        context = data.get('context', {})

        if not workflow_name:
            return jsonify({'error': 'Workflow name required'}), 400

        # Prüfen ob predefined
        if workflow_name in engine.predefined_workflows:
            result = engine.execute_predefined(workflow_name, context)
        else:
            # Versuche als Workflow-ID zu interpretieren
            if workflow_name in engine.workflows:
                result = engine.execute_workflow(workflow_name, context)
            else:
                return jsonify({'error': f'Workflow {workflow_name} not found'}), 404

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/workflows/create', methods=['POST'])
def workflows_create():
    """Erstellt einen neuen Workflow"""
    try:
        from backend.orchestration import get_workflow_engine
        engine = get_workflow_engine()

        data = request.json or {}
        name = data.get('name', 'custom_workflow')
        description = data.get('description', '')
        steps_data = data.get('steps', [])

        if not steps_data:
            return jsonify({'error': 'Workflow steps required'}), 400

        # Workflow erstellen (steps werden intern konvertiert)
        workflow = engine.create_workflow(
            name=name,
            description=description,
            steps=steps_data
        )

        return jsonify({
            "workflow_id": workflow.id,
            "name": workflow.name,
            "steps": len(workflow.steps)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/workflows/<workflow_id>/status', methods=['GET'])
def workflow_status(workflow_id: str):
    """Gibt Workflow-Status zurück"""
    try:
        from backend.orchestration import get_workflow_engine
        engine = get_workflow_engine()

        if workflow_id not in engine.workflows:
            return jsonify({'error': 'Workflow not found'}), 404

        wf = engine.workflows[workflow_id]

        return jsonify({
            "id": wf.id,
            "name": wf.name,
            "status": wf.status.value,
            "current_step": wf.current_step,
            "steps": len(wf.steps),
            "context": wf.context,
            "error": wf.error
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/stats', methods=['GET'])
def orchestration_stats():
    """Gibt umfassende Orchestration-Statistiken zurück"""
    try:
        from backend.orchestration import get_orchestrator, get_event_bus, get_workflow_engine

        orch = get_orchestrator()
        bus = get_event_bus()
        engine = get_workflow_engine()

        return jsonify({
            "orchestrator": {
                "running": orch._running,
                "modules": len(orch.modules),
                "modules_by_type": orch._count_modules_by_type()
            },
            "event_bus": bus.get_statistics(),
            "workflow_engine": engine.get_statistics(),
            "health": orch.get_health_summary()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orch_bp.route('/coordinate', methods=['POST'])
def orchestration_coordinate():
    """Koordiniert mehrere Module für eine Aufgabe"""
    try:
        from backend.orchestration import get_orchestrator
        orch = get_orchestrator()

        data = request.json or {}
        task = data.get('task', '')
        modules = data.get('modules', [])

        if not task:
            return jsonify({'error': 'Task description required'}), 400

        results = {}

        # Wenn keine Module spezifiziert, alle nutzen
        if not modules:
            modules = list(orch.modules.keys())

        for module_name in modules:
            if module_name in orch.modules:
                module = orch.modules[module_name]
                instance = module.instance

                try:
                    # Verschiedene Module-Typen behandeln
                    if module.module_type == "decision":
                        if hasattr(instance, 'make_decision'):
                            decision = instance.make_decision(task, data)
                            results[module_name] = {
                                "action": decision.action,
                                "confidence": decision.confidence
                            }

                    elif module.module_type == "planning":
                        if hasattr(instance, 'create_plan'):
                            plan = instance.create_plan(goal=task)
                            results[module_name] = {
                                "steps": len(plan.steps) if hasattr(plan, 'steps') else 0
                            }

                    elif module.module_type == "capabilities":
                        if hasattr(instance, 'find_tools'):
                            tools = instance.find_tools(task, limit=3)
                            results[module_name] = {
                                "tools": [t.tool.id for t in tools]
                            }

                    elif module.module_type == "knowledge":
                        if hasattr(instance, 'query'):
                            query_result = instance.query(task)
                            results[module_name] = {
                                "entities": len(query_result) if query_result else 0
                            }

                    else:
                        results[module_name] = {"status": "ready"}

                except Exception as e:
                    results[module_name] = {"error": str(e)}

        return jsonify({
            "task": task,
            "modules_used": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
