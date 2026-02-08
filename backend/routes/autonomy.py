#!/usr/bin/env python3
"""
SCIO - Autonomy Routes
API-Endpunkte für das Autonomie-System

Features:
- Status abrufen
- Selbst-Analyse
- Evolution triggern
- Fähigkeiten abfragen
- Memory-Zugriff
"""

from flask import Blueprint, request, jsonify
from datetime import datetime

autonomy_bp = Blueprint('autonomy', __name__)


@autonomy_bp.route('/status', methods=['GET'])
def get_status():
    """Gibt Status des Autonomie-Systems zurück"""
    try:
        from backend.autonomy import get_autonomy_engine
        autonomy = get_autonomy_engine()

        return jsonify({
            "status": "ok",
            "autonomy": autonomy.get_status(),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/analyze', methods=['GET'])
def analyze_self():
    """Führt Selbst-Analyse durch"""
    try:
        from backend.autonomy import get_autonomy_engine
        autonomy = get_autonomy_engine()

        analysis = autonomy.analyze_self()

        return jsonify({
            "status": "ok",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/capabilities', methods=['GET'])
def get_capabilities():
    """Gibt alle Fähigkeiten zurück"""
    try:
        from backend.autonomy import get_capability_analyzer
        analyzer = get_capability_analyzer()

        return jsonify({
            "status": "ok",
            "capabilities": analyzer.analyze_all(),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/gaps', methods=['GET'])
def get_gaps():
    """Gibt fehlende Fähigkeiten zurück"""
    try:
        from backend.autonomy import get_capability_analyzer
        analyzer = get_capability_analyzer()

        limit = request.args.get('limit', 10, type=int)
        gaps = analyzer.find_gaps()[:limit]

        return jsonify({
            "status": "ok",
            "gaps": gaps,
            "total": len(analyzer.find_gaps()),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/evolve', methods=['POST'])
def trigger_evolution():
    """Triggert eine Evolution"""
    try:
        from backend.autonomy import get_autonomy_engine
        autonomy = get_autonomy_engine()

        data = request.get_json() or {}
        target = data.get('target')  # Optionales spezifisches Ziel

        result = autonomy.evolve(target)

        return jsonify({
            "status": "ok" if result.success else "failed",
            "result": {
                "plan": {
                    "id": result.plan.id,
                    "name": result.plan.name,
                    "capability": result.plan.capability,
                    "status": result.plan.status.value,
                },
                "success": result.success,
                "files_created": result.files_created,
                "files_modified": result.files_modified,
                "message": result.message,
            },
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Gibt Evolutions-Vorschläge zurück"""
    try:
        from backend.autonomy import get_evolution_planner
        planner = get_evolution_planner()

        limit = request.args.get('limit', 5, type=int)
        suggestions = planner.suggest_next_steps(limit)

        return jsonify({
            "status": "ok",
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/codebase', methods=['GET'])
def get_codebase():
    """Gibt Codebase-Struktur zurück"""
    try:
        from backend.autonomy import get_self_awareness
        awareness = get_self_awareness()

        return jsonify({
            "status": "ok",
            "summary": awareness.get_summary(),
            "structure": awareness.get_structure(),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/workers', methods=['GET'])
def get_workers():
    """Gibt erkannte Worker zurück"""
    try:
        from backend.autonomy import get_self_awareness
        awareness = get_self_awareness()

        workers = awareness.get_workers()
        workers_list = []

        for name, info in workers.items():
            workers_list.append({
                "name": name,
                "file": info.file_path,
                "class": info.class_name,
                "methods": info.methods,
                "capabilities": info.capabilities,
                "models": info.models,
            })

        return jsonify({
            "status": "ok",
            "workers": workers_list,
            "total": len(workers_list),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/memory', methods=['GET'])
def get_memory():
    """Gibt Memory-Status zurück"""
    try:
        from backend.autonomy import get_memory
        memory = get_memory()

        return jsonify({
            "status": "ok",
            "memory": memory.get_summary(),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/memory/events', methods=['GET'])
def get_events():
    """Gibt Events aus dem Memory zurück"""
    try:
        from backend.autonomy import get_memory
        memory = get_memory()

        event_type = request.args.get('type')
        limit = request.args.get('limit', 50, type=int)

        events = memory.get_events(event_type=event_type, limit=limit)

        return jsonify({
            "status": "ok",
            "events": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type,
                    "message": e.message,
                    "data": e.data,
                    "tags": e.tags,
                }
                for e in events
            ],
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/memory/learnings', methods=['GET'])
def get_learnings():
    """Gibt Learnings aus dem Memory zurück"""
    try:
        from backend.autonomy import get_memory
        memory = get_memory()

        category = request.args.get('category')
        limit = request.args.get('limit', 20, type=int)

        learnings = memory.get_learnings(category=category, limit=limit)

        return jsonify({
            "status": "ok",
            "learnings": [
                {
                    "id": l.id,
                    "category": l.category,
                    "pattern": l.pattern,
                    "outcome": l.outcome,
                    "confidence": l.confidence,
                    "occurrences": l.occurrences,
                    "last_seen": l.last_seen.isoformat(),
                }
                for l in learnings
            ],
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/evolution/history', methods=['GET'])
def get_evolution_history():
    """Gibt Evolutions-Historie zurück"""
    try:
        from backend.autonomy import get_evolution_planner
        planner = get_evolution_planner()

        history = planner.get_history()

        return jsonify({
            "status": "ok",
            "history": [
                {
                    "plan": {
                        "id": r.plan.id,
                        "name": r.plan.name,
                        "capability": r.plan.capability,
                        "status": r.plan.status.value,
                    },
                    "success": r.success,
                    "files_created": r.files_created,
                    "files_modified": r.files_modified,
                    "message": r.message,
                }
                for r in history
            ],
            "stats": planner.get_status(),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/auto-evolve', methods=['POST'])
def auto_evolve():
    """Führt automatische Evolution durch"""
    try:
        from backend.autonomy import get_evolution_planner
        planner = get_evolution_planner()

        data = request.get_json() or {}
        max_evolutions = data.get('max', 1)

        results = planner.auto_evolve(max_evolutions)

        return jsonify({
            "status": "ok",
            "evolutions": [
                {
                    "capability": r.plan.capability,
                    "success": r.success,
                    "message": r.message,
                }
                for r in results
            ],
            "total": len(results),
            "successful": sum(1 for r in results if r.success),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/verify', methods=['POST'])
def verify_task():
    """
    Verifiziert dass ein Task zu 100% erfüllt wurde

    Body:
    {
        "description": "Task-Beschreibung",
        "expected": {
            "files": ["path/to/file.py"],
            "code_files": ["path/to/file.py"],
            "classes": [{"file": "path", "name": "ClassName", "methods": ["method1"]}],
            "functions": [{"file": "path", "name": "func_name"}],
            "content_contains": [{"file": "path", "text": "expected text"}],
            "no_errors": true
        }
    }
    """
    try:
        from backend.autonomy import get_autonomy_engine
        autonomy = get_autonomy_engine()

        data = request.get_json() or {}
        description = data.get('description', 'Custom task')
        expected = data.get('expected', {})

        result = autonomy.verify_task(description, expected)

        return jsonify({
            "status": "ok",
            "verification": result,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/verify/stats', methods=['GET'])
def get_verification_stats():
    """Gibt Verifizierungs-Statistiken zurück"""
    try:
        from backend.autonomy import get_task_verifier
        verifier = get_task_verifier()

        return jsonify({
            "status": "ok",
            "statistics": verifier.get_statistics(),
            "guarantee": "100% Task-Erfüllung",
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/verify/history', methods=['GET'])
def get_verification_history():
    """Gibt Verifizierungs-Historie zurück"""
    try:
        from backend.autonomy import get_task_verifier
        verifier = get_task_verifier()

        history = verifier.get_history()
        limit = request.args.get('limit', 20, type=int)

        return jsonify({
            "status": "ok",
            "history": [
                {
                    "status": v.status.value,
                    "completion_percent": v.completion_percent,
                    "checks_passed": v.checks_passed,
                    "checks_total": v.checks_total,
                    "missing": v.missing,
                    "timestamp": v.timestamp.isoformat(),
                }
                for v in history[-limit:]
            ],
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
