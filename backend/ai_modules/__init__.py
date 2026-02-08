#!/usr/bin/env python3
"""
SCIO - AI Modules Central Import
Zentrale Import-Stelle für alle AI-Module
"""

# Decision Engine
try:
    from backend.decision import (
        get_decision_engine,
        get_rule_engine,
        DecisionEngine,
        RuleEngine
    )
except ImportError:
    get_decision_engine = None
    get_rule_engine = None
    DecisionEngine = None
    RuleEngine = None

# Learning Module
try:
    from backend.learning import (
        get_rl_agent,
        get_continuous_learner,
        RLAgent,
        ContinuousLearner
    )
except ImportError:
    get_rl_agent = None
    get_continuous_learner = None
    RLAgent = None
    ContinuousLearner = None

# Planning Module
try:
    from backend.planning import (
        get_planner,
        Planner
    )
except ImportError:
    get_planner = None
    Planner = None

# Knowledge Graph
try:
    from backend.knowledge import (
        get_knowledge_graph,
        KnowledgeGraph
    )
except ImportError:
    get_knowledge_graph = None
    KnowledgeGraph = None

# Multi-Agent System
try:
    from backend.agents import (
        get_multi_agent_system,
        MultiAgentSystem
    )
except ImportError:
    get_multi_agent_system = None
    MultiAgentSystem = None

# Monitoring
try:
    from backend.monitoring import (
        get_drift_detector,
        get_performance_tracker,
        DriftDetector,
        PerformanceTracker
    )
except ImportError:
    get_drift_detector = None
    get_performance_tracker = None
    DriftDetector = None
    PerformanceTracker = None

__all__ = [
    # Decision
    'get_decision_engine',
    'get_rule_engine',
    'DecisionEngine',
    'RuleEngine',
    # Learning
    'get_rl_agent',
    'get_continuous_learner',
    'RLAgent',
    'ContinuousLearner',
    # Planning
    'get_planner',
    'Planner',
    # Knowledge
    'get_knowledge_graph',
    'KnowledgeGraph',
    # Agents
    'get_multi_agent_system',
    'MultiAgentSystem',
    # Monitoring
    'get_drift_detector',
    'get_performance_tracker',
    'DriftDetector',
    'PerformanceTracker',
]
