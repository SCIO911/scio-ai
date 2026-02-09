#!/usr/bin/env python3
"""
SCIO - Capabilities API Routes
Routes für Tool-Registry und Capability Engine
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any

caps_bp = Blueprint('capabilities', __name__, url_prefix='/api/capabilities')


@caps_bp.route('/stats', methods=['GET'])
def capabilities_stats():
    """Gibt Capability-Statistiken zurück"""
    try:
        from backend.capabilities import get_capability_engine
        engine = get_capability_engine()

        stats = engine.get_statistics()
        summary = engine.get_all_capabilities_summary()

        return jsonify({
            "total_tools": summary["total_tools"],
            "categories": summary["total_categories"],
            "category_distribution": summary["categories"],
            "top_tags": summary["top_tags"],
            "chains_available": len(summary["chains_available"]),
            "gpu_tools": summary["gpu_tools"],
            "network_tools": summary["network_tools"]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/tools', methods=['GET'])
def list_tools():
    """Gibt alle verfuegbaren Tools zurueck"""
    try:
        from backend.capabilities import get_capability_engine
        engine = get_capability_engine()

        # Query-Parameter
        category = request.args.get('category')
        search = request.args.get('search')
        limit = request.args.get('limit', 100, type=int)

        tools = engine.get_all_tools()

        # Filter nach Kategorie
        if category:
            tools = [t for t in tools if t.get('category') == category]

        # Suche
        if search:
            search = search.lower()
            tools = [t for t in tools if search in t.get('name', '').lower() or search in t.get('description', '').lower()]

        return jsonify({
            'tools': tools[:limit],
            'total': len(tools),
            'limit': limit
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/search', methods=['POST'])
def capabilities_search():
    """Sucht nach passenden Tools"""
    try:
        from backend.capabilities import get_capability_engine
        engine = get_capability_engine()

        data = request.json or {}
        query = data.get('query', '')
        category = data.get('category')
        tags = data.get('tags', [])
        limit = data.get('limit', 20)

        if query:
            matches = engine.find_tools(query, limit=limit)
            return jsonify({
                'results': [
                    {
                        'tool_id': m.tool.id,
                        'name': m.tool.name,
                        'description': m.tool.description,
                        'category': m.tool.category.value,
                        'confidence': m.confidence,
                        'reasoning': m.reasoning
                    }
                    for m in matches
                ],
                'count': len(matches)
            })
        else:
            from backend.capabilities.tool_registry import ToolCategory
            cat = ToolCategory(category) if category else None
            tools = engine.registry.search(category=cat, tags=tags, limit=limit)
            return jsonify({
                'results': [t.to_dict() for t in tools],
                'count': len(tools)
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/plan', methods=['POST'])
def capabilities_plan():
    """Erstellt Ausführungsplan für Aufgabe"""
    try:
        from backend.capabilities import get_capability_engine
        engine = get_capability_engine()

        data = request.json or {}
        task = data.get('task', '')

        if not task:
            return jsonify({'error': 'Task description required'}), 400

        plan = engine.create_plan(task)

        return jsonify({
            'plan_id': plan.id,
            'description': plan.description,
            'complexity': plan.complexity.value,
            'steps': plan.steps,
            'estimated_time_ms': plan.estimated_time_ms,
            'requires_gpu': plan.requires_gpu,
            'confidence': plan.confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/tool/<tool_id>', methods=['GET'])
def capabilities_tool(tool_id: str):
    """Gibt Tool-Details zurück"""
    try:
        from backend.capabilities import get_tool_registry
        registry = get_tool_registry()

        tool = registry.get_tool(tool_id)
        if not tool:
            return jsonify({'error': 'Tool not found'}), 404

        return jsonify({
            'id': tool.id,
            'name': tool.name,
            'description': tool.description,
            'category': tool.category.value,
            'subcategory': tool.subcategory,
            'input_types': tool.input_types,
            'output_types': tool.output_types,
            'parameters': tool.parameters,
            'tags': tool.tags,
            'requires_gpu': tool.requires_gpu,
            'requires_network': tool.requires_network,
            'enabled': tool.enabled,
            'usage_count': tool.usage_count,
            'avg_execution_time_ms': tool.avg_execution_time_ms,
            'success_rate': tool.success_rate
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/categories', methods=['GET'])
def capabilities_categories():
    """Gibt alle Kategorien zurück"""
    try:
        from backend.capabilities import get_tool_registry
        registry = get_tool_registry()

        categories = registry.get_categories()

        return jsonify({
            'categories': categories,
            'total': sum(categories.values())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/domain/<domain>', methods=['GET'])
def capabilities_domain(domain: str):
    """Gibt Capabilities einer Domain zurück"""
    try:
        from backend.capabilities import get_capability_engine
        engine = get_capability_engine()

        capabilities = engine.get_capabilities_by_domain(domain)

        return jsonify({
            'domain': domain,
            'capabilities': capabilities,
            'count': len(capabilities)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/chains', methods=['GET'])
def capabilities_chains():
    """Gibt verfügbare Tool-Chains zurück"""
    try:
        from backend.capabilities import get_capability_engine
        engine = get_capability_engine()

        chains_info = []
        for name, tools in engine.chains.items():
            chains_info.append({
                'name': name,
                'tools': tools,
                'steps': len(tools)
            })

        return jsonify({
            'chains': chains_info,
            'count': len(chains_info)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/execute', methods=['POST'])
def capabilities_execute():
    """Führt ein Tool aus"""
    try:
        from backend.capabilities import get_tool_registry
        registry = get_tool_registry()

        data = request.json or {}
        tool_id = data.get('tool_id', '')
        params = data.get('params', {})

        if not tool_id:
            return jsonify({'error': 'Tool ID required'}), 400

        result = registry.execute(tool_id, params)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/tags', methods=['GET'])
def capabilities_tags():
    """Gibt alle Tags zurück"""
    try:
        from backend.capabilities import get_tool_registry
        registry = get_tool_registry()

        tags = {tag: len(tools) for tag, tools in registry.tags_index.items()}
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            'tags': dict(sorted_tags[:100]),
            'total_unique': len(tags)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@caps_bp.route('/export', methods=['GET'])
def capabilities_export():
    """Exportiert Tool-Katalog"""
    try:
        from backend.capabilities import get_tool_registry
        registry = get_tool_registry()

        catalog = registry.export_catalog()

        return jsonify({
            'catalog': catalog,
            'format': 'json'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
