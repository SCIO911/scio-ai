#!/usr/bin/env python3
"""
SCIO Constants
Alle Magic Numbers und Konfigurationswerte als benannte Konstanten
"""

# ═══════════════════════════════════════════════════════════════════════════
# MEMORY LIMITS
# ═══════════════════════════════════════════════════════════════════════════

# Event Bus
EVENT_HISTORY_MAX_SIZE = 1000
EVENT_QUEUE_MAX_SIZE = 10000

# Workflow Engine
WORKFLOW_HISTORY_MAX_SIZE = 100
WORKFLOW_STEP_MAX_RETRIES = 3
WORKFLOW_STEP_RETRY_DELAY_SECONDS = 1.0

# Knowledge Graph
KNOWLEDGE_ENTITY_MAX_COUNT = 100000
KNOWLEDGE_RELATION_MAX_COUNT = 500000
KNOWLEDGE_QUERY_MAX_RESULTS = 1000

# Learning
RL_EXPERIENCE_BUFFER_SIZE = 10000
RL_Q_TABLE_MAX_STATES = 100000
CONTINUOUS_LEARNER_PATTERN_BUFFER_SIZE = 5000

# Monitoring
METRICS_HISTORY_MAX_SIZE = 10000
ALERTS_MAX_COUNT = 1000
LATENCY_HISTORY_SIZE = 1000

# Workers
MODEL_CACHE_MAX_SIZE = 5
JOB_RESULT_MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100MB


# ═══════════════════════════════════════════════════════════════════════════
# TIMEOUTS (Sekunden)
# ═══════════════════════════════════════════════════════════════════════════

# Job Processing
JOB_DEFAULT_TIMEOUT = 300
JOB_MAX_TIMEOUT = 3600
JOB_CLEANUP_INTERVAL = 60

# Health Checks
HEALTH_CHECK_INTERVAL = 30
HEALTH_CHECK_TIMEOUT = 10
MODULE_HEALTH_CHECK_INTERVAL = 30

# Event Processing
EVENT_PROCESSING_TIMEOUT = 5.0
EVENT_CALLBACK_TIMEOUT = 2.0

# Workflow
WORKFLOW_STEP_TIMEOUT = 300
WORKFLOW_MAX_EXECUTION_TIME = 3600

# Model Loading
MODEL_LOAD_TIMEOUT = 120
MODEL_INFERENCE_TIMEOUT = 60


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE TUNING
# ═══════════════════════════════════════════════════════════════════════════

# Thread Pools
EVENT_CALLBACK_THREAD_POOL_SIZE = 10
WORKFLOW_EXECUTOR_POOL_SIZE = 5
AGENT_TASK_POOL_SIZE = 10

# Batch Sizes
EMBEDDING_BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 8
KNOWLEDGE_QUERY_BATCH_SIZE = 100

# Cache
LRU_CACHE_SIZE = 1000
TOOL_SEARCH_CACHE_SIZE = 500
DECISION_CACHE_TTL_SECONDS = 60


# ═══════════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING
# ═══════════════════════════════════════════════════════════════════════════

# Q-Learning Hyperparameters
RL_LEARNING_RATE = 0.1
RL_DISCOUNT_FACTOR = 0.95
RL_EXPLORATION_RATE_INITIAL = 1.0
RL_EXPLORATION_RATE_MIN = 0.01
RL_EXPLORATION_DECAY = 0.995

# UCB (Upper Confidence Bound)
UCB_EXPLORATION_CONSTANT = 1.414  # sqrt(2)

# MCTS
MCTS_MAX_ITERATIONS = 1000
MCTS_MAX_DEPTH = 50
MCTS_SIMULATION_LIMIT = 100


# ═══════════════════════════════════════════════════════════════════════════
# SECURITY
# ═══════════════════════════════════════════════════════════════════════════

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_SECOND = 10
RATE_LIMIT_REQUESTS_PER_MINUTE = 100
RATE_LIMIT_REQUESTS_PER_HOUR = 1000
RATE_LIMIT_BURST_SIZE = 20
RATE_LIMIT_BLOCK_DURATION = 60

# Input Validation
MAX_INPUT_LENGTH = 100000
MAX_PAYLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_FILE_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100MB

# Expression Evaluation
SAFE_EVAL_MAX_DEPTH = 20
SAFE_EVAL_MAX_NODES = 100
SAFE_EVAL_MAX_LENGTH = 1000


# ═══════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════

DRIFT_DETECTION_WINDOW_SIZE = 100
DRIFT_DETECTION_THRESHOLD = 0.1
DRIFT_ALERT_COOLDOWN_SECONDS = 300

# Thresholds für verschiedene Metriken
DRIFT_THRESHOLD_CONFIDENCE = 0.2
DRIFT_THRESHOLD_LATENCY = 0.5
DRIFT_THRESHOLD_ERROR_RATE = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# PLANNING
# ═══════════════════════════════════════════════════════════════════════════

PLAN_MAX_DEPTH = 10
PLAN_MAX_ACTIONS = 100
PLAN_TIMEOUT = 30

# A* Search
ASTAR_MAX_EXPANSIONS = 10000
ASTAR_HEURISTIC_WEIGHT = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-AGENT
# ═══════════════════════════════════════════════════════════════════════════

AGENT_MESSAGE_QUEUE_SIZE = 1000
AGENT_TASK_TIMEOUT = 300
AGENT_CAPABILITY_MATCH_THRESHOLD = 0.5
MAX_AGENTS = 100


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

LOG_MAX_MESSAGE_LENGTH = 10000
LOG_ROTATION_SIZE_MB = 100
LOG_RETENTION_DAYS = 30
AUDIT_LOG_ENABLED = True


# ═══════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════

API_VERSION = "2.1.0"
API_DEFAULT_PAGE_SIZE = 20
API_MAX_PAGE_SIZE = 100
API_REQUEST_TIMEOUT = 30

# CORS
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5000",
    "https://scio.local",
]
CORS_ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
CORS_ALLOWED_HEADERS = ["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"]


# ═══════════════════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════

SHUTDOWN_TIMEOUT = 30
SHUTDOWN_DRAIN_CONNECTIONS_TIMEOUT = 10
SHUTDOWN_WAIT_FOR_JOBS_TIMEOUT = 60
