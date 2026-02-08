#!/usr/bin/env python3
"""
SCIO - AI Modules API Routes
Routes für Decision, Learning, Planning, Knowledge, Agents, Monitoring
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any

ai_bp = Blueprint('ai', __name__, url_prefix='/api/ai')


# ═══════════════════════════════════════════════════════════════
# DECISION ENGINE ROUTES
# ═══════════════════════════════════════════════════════════════

@ai_bp.route('/decision/decide', methods=['POST'])
def decision_decide():
    """Trifft eine Entscheidung basierend auf Context"""
    try:
        from backend.decision import get_decision_engine
        import uuid
        engine = get_decision_engine()

        data = request.json or {}
        tree_name = data.get('tree', 'worker_selection')
        context = data.get('context', {})
        use_heuristics = data.get('use_heuristics', True)

        decision = engine.decide(tree_name, context, use_heuristics)

        return jsonify({
            'decision_id': str(uuid.uuid4())[:8],
            'action': decision.action,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'decision_type': decision.decision_type.value,
            'processing_time_ms': decision.processing_time_ms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/decision/best-action', methods=['POST'])
def decision_best_action():
    """Findet beste Aktion aus Liste"""
    try:
        from backend.decision import get_decision_engine
        engine = get_decision_engine()

        data = request.json or {}
        actions = data.get('actions', [])
        context = data.get('context', {})

        if not actions:
            return jsonify({'error': 'No actions provided'}), 400

        action, score = engine.decide_best_action(actions, context)

        return jsonify({
            'best_action': action,
            'score': score,
            'all_actions': actions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/decision/stats', methods=['GET'])
def decision_stats():
    """Gibt Decision Engine Statistiken zurück"""
    try:
        from backend.decision import get_decision_engine
        engine = get_decision_engine()
        return jsonify(engine.get_statistics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# RULE ENGINE ROUTES
# ═══════════════════════════════════════════════════════════════

@ai_bp.route('/rules/evaluate', methods=['POST'])
def rules_evaluate():
    """Evaluiert Regeln gegen Context"""
    try:
        from backend.decision import get_rule_engine
        engine = get_rule_engine()

        data = request.json or {}
        rule_set = data.get('rule_set', 'api_rules')
        context = data.get('context', {})
        execute = data.get('execute', False)

        result = engine.evaluate(rule_set, context, execute)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/rules/list', methods=['GET'])
def rules_list():
    """Listet alle Regelsets"""
    try:
        from backend.decision import get_rule_engine
        engine = get_rule_engine()

        return jsonify({
            'rule_sets': list(engine.rule_sets.keys()),
            'total_rules': sum(len(rs.rules) for rs in engine.rule_sets.values())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# LEARNING ROUTES (RL & Continuous)
# ═══════════════════════════════════════════════════════════════

@ai_bp.route('/learning/action', methods=['POST'])
def learning_select_action():
    """Wählt beste Aktion via RL"""
    try:
        from backend.learning import get_rl_agent
        import uuid
        agent = get_rl_agent()

        data = request.json or {}
        context = data.get('context', {})
        action_space = data.get('action_space', 'worker_selection')

        action = agent.select_action(context, action_space)

        return jsonify({
            'action_id': str(uuid.uuid4())[:8],
            'action': action.name,
            'params': action.params,
            'action_space': action_space,
            'exploration_rate': agent.q_agent.epsilon
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/learning/feedback', methods=['POST'])
def learning_feedback():
    """Zeichnet Feedback auf"""
    try:
        from backend.learning import get_continuous_learner
        from backend.learning.continuous_learning import Feedback, FeedbackType
        learner = get_continuous_learner()

        data = request.json or {}

        feedback = Feedback(
            feedback_type=FeedbackType(data.get('type', 'explicit')),
            value=data.get('value', 0.5),
            context=data.get('context', {}),
            action=data.get('action', ''),
            user_id=data.get('user_id')
        )

        learner.record_feedback(feedback)

        return jsonify({'status': 'recorded', 'feedback_id': feedback.id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/learning/recommend', methods=['POST'])
def learning_recommend():
    """Gibt Empfehlung basierend auf gelernten Patterns"""
    try:
        from backend.learning import get_continuous_learner
        learner = get_continuous_learner()

        data = request.json or {}
        context = data.get('context', {})

        recommendation = learner.get_recommendation(context)

        return jsonify({
            'recommendation': recommendation,
            'confidence': 0.8 if recommendation else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/learning/stats', methods=['GET'])
def learning_stats():
    """Gibt Learning Statistiken zurück"""
    try:
        from backend.learning import get_rl_agent, get_continuous_learner
        rl = get_rl_agent()
        cl = get_continuous_learner()

        return jsonify({
            'rl_agent': rl.get_statistics(),
            'continuous_learner': cl.get_statistics()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# PLANNING ROUTES
# ═══════════════════════════════════════════════════════════════

@ai_bp.route('/planning/plan', methods=['POST'])
def planning_create():
    """Erstellt hierarchischen Plan"""
    try:
        from backend.planning import get_planner
        planner = get_planner()

        data = request.json or {}
        goal = data.get('goal', '')
        decomposition = data.get('decomposition', {})

        if not goal:
            return jsonify({'error': 'Goal required'}), 400

        plan = planner.hierarchical_plan(goal, decomposition)

        return jsonify({
            'plan_id': plan.id,
            'goal': plan.goal,
            'steps': plan.steps,
            'status': plan.status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/planning/execute/<plan_id>', methods=['POST'])
def planning_execute(plan_id: str):
    """Führt Plan aus"""
    try:
        from backend.planning import get_planner
        planner = get_planner()

        result = planner.execute_plan(plan_id)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/planning/search', methods=['POST'])
def planning_search():
    """Führt A* oder MCTS Suche durch"""
    try:
        from backend.planning import get_planner
        from backend.planning.planner import PlanningState
        planner = get_planner()

        data = request.json or {}
        algorithm = data.get('algorithm', 'astar')
        initial_state = data.get('state', {})
        goal_state = data.get('goal', {})

        state = PlanningState(
            id="search_start",
            data=initial_state,
            parent=None,
            action=None,
            cost=0
        )

        # Simple goal check
        def goal_check(s):
            return all(s.data.get(k) == v for k, v in goal_state.items())

        state.is_goal = goal_check

        if algorithm == 'mcts':
            actions = planner.mcts_search(state, iterations=data.get('iterations', 100))
        else:
            actions = planner.astar_search(state, max_iterations=data.get('iterations', 1000))

        return jsonify({
            'algorithm': algorithm,
            'actions': actions or [],
            'success': actions is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/planning/stats', methods=['GET'])
def planning_stats():
    """Gibt Planner Statistiken zurück"""
    try:
        from backend.planning import get_planner
        planner = get_planner()
        return jsonify(planner.get_statistics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH ROUTES
# ═══════════════════════════════════════════════════════════════

@ai_bp.route('/knowledge/entity', methods=['POST'])
def knowledge_add_entity():
    """Fügt Entity hinzu"""
    try:
        from backend.knowledge import get_knowledge_graph
        from backend.knowledge.knowledge_graph import Entity
        kg = get_knowledge_graph()

        data = request.json or {}

        entity = Entity(
            id=data.get('id', ''),
            type=data.get('type', 'unknown'),
            name=data.get('name', ''),
            properties=data.get('properties', {})
        )

        success = kg.add_entity(entity)

        return jsonify({
            'success': success,
            'entity_id': entity.id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/knowledge/entity/<entity_id>', methods=['GET'])
def knowledge_get_entity(entity_id: str):
    """Gibt Entity zurück"""
    try:
        from backend.knowledge import get_knowledge_graph
        kg = get_knowledge_graph()

        entity = kg.get_entity(entity_id)
        if not entity:
            return jsonify({'error': 'Entity not found'}), 404

        return jsonify({
            'id': entity.id,
            'type': entity.type,
            'name': entity.name,
            'properties': entity.properties
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/knowledge/relation', methods=['POST'])
def knowledge_add_relation():
    """Fügt Relation hinzu"""
    try:
        from backend.knowledge import get_knowledge_graph
        from backend.knowledge.knowledge_graph import Relation
        import uuid
        kg = get_knowledge_graph()

        data = request.json or {}

        relation = Relation(
            id=data.get('id', str(uuid.uuid4())[:8]),
            type=data.get('type', 'related_to'),
            source_id=data.get('source_id', ''),
            target_id=data.get('target_id', ''),
            properties=data.get('properties', {})
        )

        success = kg.add_relation(relation)

        return jsonify({
            'success': success,
            'source': relation.source_id,
            'target': relation.target_id,
            'type': relation.type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/knowledge/query', methods=['POST'])
def knowledge_query():
    """Sucht im Knowledge Graph"""
    try:
        from backend.knowledge import get_knowledge_graph
        kg = get_knowledge_graph()

        data = request.json or {}

        results = kg.query(
            entity_type=data.get('entity_type'),
            relation_type=data.get('relation_type'),
            properties=data.get('properties', {})
        )

        return jsonify({
            'results': [
                {'id': e.id, 'type': e.type, 'name': e.name}
                for e in results
            ],
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/knowledge/path', methods=['POST'])
def knowledge_find_path():
    """Findet Pfad zwischen Entities"""
    try:
        from backend.knowledge import get_knowledge_graph
        kg = get_knowledge_graph()

        data = request.json or {}
        source = data.get('source_id', '')
        target = data.get('target_id', '')

        path = kg.find_path(source, target)

        return jsonify({
            'path': path,
            'found': path is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/knowledge/infer', methods=['POST'])
def knowledge_infer():
    """Führt Inferenz durch"""
    try:
        from backend.knowledge import get_knowledge_graph
        kg = get_knowledge_graph()

        new_relations = kg.infer()

        return jsonify({
            'new_relations': len(new_relations),
            'relations': [
                {'source': r.source_id, 'target': r.target_id, 'type': r.relation_type}
                for r in new_relations
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/knowledge/stats', methods=['GET'])
def knowledge_stats():
    """Gibt Knowledge Graph Statistiken zurück"""
    try:
        from backend.knowledge import get_knowledge_graph
        kg = get_knowledge_graph()
        return jsonify(kg.get_statistics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# MULTI-AGENT SYSTEM ROUTES
# ═══════════════════════════════════════════════════════════════

@ai_bp.route('/agents/list', methods=['GET'])
def agents_list():
    """Listet alle Agenten"""
    try:
        from backend.agents import get_multi_agent_system
        mas = get_multi_agent_system()

        agents = []
        for agent_id, agent in mas.agents.items():
            agents.append({
                'id': agent.agent_id,
                'name': agent.name,
                'status': agent.status.value,
                'capabilities': agent.capabilities
            })

        return jsonify({
            'agents': agents,
            'count': len(agents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/agents/<agent_id>/status', methods=['GET'])
def agents_status(agent_id: str):
    """Gibt Agent Status zurück"""
    try:
        from backend.agents import get_multi_agent_system
        mas = get_multi_agent_system()

        status = mas.get_agent_status(agent_id)
        if not status:
            return jsonify({'error': 'Agent not found'}), 404

        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/agents/delegate', methods=['POST'])
def agents_delegate():
    """Delegiert Task an passendes Agent"""
    try:
        from backend.agents import get_multi_agent_system
        mas = get_multi_agent_system()

        data = request.json or {}
        description = data.get('description', '')
        params = data.get('params', {})
        priority = data.get('priority', 0)

        if not description:
            return jsonify({'error': 'Description required'}), 400

        task = mas.delegate_task(description, params, priority)

        return jsonify({
            'task_id': task.id,
            'assigned_to': task.assigned_to,
            'status': task.status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/agents/broadcast', methods=['POST'])
def agents_broadcast():
    """Broadcast an alle Agenten"""
    try:
        from backend.agents import get_multi_agent_system
        mas = get_multi_agent_system()

        data = request.json or {}
        content = data.get('content', {})

        mas.broadcast(content)

        return jsonify({
            'status': 'broadcasted',
            'recipients': len(mas.agents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/agents/stats', methods=['GET'])
def agents_stats():
    """Gibt Multi-Agent System Statistiken zurück"""
    try:
        from backend.agents import get_multi_agent_system
        mas = get_multi_agent_system()
        return jsonify(mas.get_statistics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# MONITORING ROUTES (Drift & Performance)
# ═══════════════════════════════════════════════════════════════

@ai_bp.route('/monitoring/record', methods=['POST'])
def monitoring_record():
    """Zeichnet Metrik auf"""
    try:
        from backend.monitoring import get_drift_detector, get_performance_tracker
        drift = get_drift_detector()
        perf = get_performance_tracker()

        data = request.json or {}
        metric_name = data.get('metric', '')
        value = data.get('value', 0)

        if not metric_name:
            return jsonify({'error': 'Metric name required'}), 400

        drift.record(metric_name, value)

        return jsonify({
            'status': 'recorded',
            'metric': metric_name,
            'value': value
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/monitoring/drift/alerts', methods=['GET'])
def monitoring_drift_alerts():
    """Gibt Drift Alerts zurück"""
    try:
        from backend.monitoring import get_drift_detector
        drift = get_drift_detector()

        severity = request.args.get('severity')
        drift_type = request.args.get('type')

        alerts = drift.get_alerts()

        return jsonify({
            'alerts': [
                {
                    'id': a.id,
                    'type': a.drift_type.value,
                    'severity': a.severity.value,
                    'metric': a.metric_name,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in alerts
            ],
            'count': len(alerts)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/monitoring/drift/health/<metric>', methods=['GET'])
def monitoring_metric_health(metric: str):
    """Gibt Metrik-Gesundheit zurück"""
    try:
        from backend.monitoring import get_drift_detector
        drift = get_drift_detector()

        health = drift.get_metric_health(metric)

        return jsonify(health)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/monitoring/performance/current', methods=['GET'])
def monitoring_perf_current():
    """Gibt aktuelle Performance Statistiken zurück"""
    try:
        from backend.monitoring import get_performance_tracker
        perf = get_performance_tracker()

        return jsonify(perf.get_current_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/monitoring/performance/hourly', methods=['GET'])
def monitoring_perf_hourly():
    """Gibt stündliche Performance Reports zurück"""
    try:
        from backend.monitoring import get_performance_tracker
        perf = get_performance_tracker()

        hours = int(request.args.get('hours', 24))

        return jsonify({
            'report': perf.get_hourly_report(hours),
            'hours': hours
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/monitoring/performance/daily', methods=['GET'])
def monitoring_perf_daily():
    """Gibt tägliche Performance Reports zurück"""
    try:
        from backend.monitoring import get_performance_tracker
        perf = get_performance_tracker()

        days = int(request.args.get('days', 30))

        return jsonify({
            'report': perf.get_daily_report(days),
            'days': days
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/monitoring/performance/model/<model>', methods=['GET'])
def monitoring_perf_model(model: str):
    """Gibt Model-spezifische Statistiken zurück"""
    try:
        from backend.monitoring import get_performance_tracker
        perf = get_performance_tracker()

        return jsonify(perf.get_model_stats(model))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_bp.route('/monitoring/stats', methods=['GET'])
def monitoring_stats():
    """Gibt Monitoring Gesamtstatistiken zurück"""
    try:
        from backend.monitoring import get_drift_detector, get_performance_tracker
        drift = get_drift_detector()
        perf = get_performance_tracker()

        return jsonify({
            'drift_detector': drift.get_statistics(),
            'performance_tracker': perf.get_statistics()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# COMBINED STATS
# ═══════════════════════════════════════════════════════════════

@ai_bp.route('/stats', methods=['GET'])
def ai_combined_stats():
    """Gibt kombinierte Statistiken aller AI-Module zurück"""
    try:
        stats = {}

        try:
            from backend.decision import get_decision_engine
            stats['decision'] = get_decision_engine().get_statistics()
        except:
            stats['decision'] = {'status': 'unavailable'}

        try:
            from backend.learning import get_rl_agent
            stats['learning'] = get_rl_agent().get_statistics()
        except:
            stats['learning'] = {'status': 'unavailable'}

        try:
            from backend.planning import get_planner
            stats['planning'] = get_planner().get_statistics()
        except:
            stats['planning'] = {'status': 'unavailable'}

        try:
            from backend.knowledge import get_knowledge_graph
            stats['knowledge'] = get_knowledge_graph().get_statistics()
        except:
            stats['knowledge'] = {'status': 'unavailable'}

        try:
            from backend.agents import get_multi_agent_system
            stats['agents'] = get_multi_agent_system().get_statistics()
        except:
            stats['agents'] = {'status': 'unavailable'}

        try:
            from backend.monitoring import get_drift_detector, get_performance_tracker
            stats['monitoring'] = {
                'drift': get_drift_detector().get_statistics(),
                'performance': get_performance_tracker().get_statistics()
            }
        except:
            stats['monitoring'] = {'status': 'unavailable'}

        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
