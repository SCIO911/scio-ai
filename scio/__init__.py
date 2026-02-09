"""
SCIO - Superintelligent Cognitive Intelligence Orchestrator

Der mächtigste AI-Agent der Welt mit:
- 1500+ Tools für jede erdenkliche Aufgabe
- Superintelligentes Reasoning (ToT, CoT, ReAct, MCTS)
- Unbegrenztes Wissen durch Internet + Knowledge Graph
- Langzeit-Gedächtnis über alle Sessions
- Multi-Modal Mastery (Text, Bild, Audio, Video, Code, 3D)
- Multi-Agent Schwarm-Intelligenz
- 100 Bewusstseins-Skills auf Meister-Niveau
- Selbst-Evolution für kontinuierliche Verbesserung

Hauptmodule:
- core: Konfiguration, Logging, Utilities
- agents: Agenten-Framework und Builtin-Agenten
- tools: Tool-Framework und 1500+ Ultimate Tools
- knowledge: Wissensbasis, Embeddings, Knowledge Graph, Internet-Zugang
- reasoning: Advanced Reasoning (ToT, CoT, ReAct, MCTS)
- memory: Persistentes Langzeit-Gedächtnis
- multimodal: Multi-Modal Engine (Text, Bild, Audio, Video, Code, 3D)
- swarm: Multi-Agent Schwarm-Intelligenz
- consciousness: 100 Master-Skills und Meta-Kognition
- algorithms: Umfassende Algorithmen-Bibliothek
- hardware: Hardware-Erkennung und GPU-Verwaltung
- execution: Ausfuehrungsengine und Scheduler
- protocols: Protokolle fuer Nachrichten, Daten, Events
- persistence: Experiment-Speicherung
- validation: Validierungssystem
- ultimate: Zentrale SCIO Ultimate Initialisierung
"""

__version__ = "2.0.0"  # SCIO Ultimate
__author__ = "SCIO Team"

# Core
from scio.core.config import Config, get_config
from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

# Agents
from scio.agents.base import Agent, AgentConfig
from scio.agents.registry import AgentRegistry, register_agent

# Tools
from scio.tools.base import Tool
from scio.tools.registry import ToolRegistry, register_tool

# Knowledge
from scio.knowledge.base import KnowledgeBase, KnowledgeEntry
from scio.knowledge.graph import KnowledgeGraph, Entity, Relation
from scio.knowledge.reasoning import ReasoningEngine
from scio.knowledge.embeddings import TextEmbedder

# Hardware
from scio.hardware.config import SYSTEM_PROFILE, OptimalSettings
from scio.hardware.detector import HardwareDetector
from scio.hardware.monitor import HardwareMonitor, get_monitor
from scio.hardware.gpu import GPUManager, get_gpu_manager

# Execution
from scio.execution.engine import ExecutionEngine
from scio.execution.checkpoint import CheckpointManager

# Protocols
from scio.protocols.messages import Message, Request, Response
from scio.protocols.events import Event, EventBus
from scio.protocols.execution import ExecutionContext, ExecutionState

# ============================================================================
# SCIO ULTIMATE COMPONENTS
# ============================================================================

# Advanced Reasoning (ToT, CoT, ReAct, MCTS)
from scio.reasoning import (
    AdvancedReasoning,
    ReasoningStrategy,
    ReasoningResult,
    TreeOfThought,
    ChainOfThought,
    ReActReasoner,
    MCTSPlanner,
    SelfReflection,
)

# Persistent Memory
from scio.memory import (
    PersistentMemory,
    Memory,
    MemoryType,
    get_memory,
)

# Internet Knowledge
from scio.knowledge import (
    InternetKnowledge,
    SearchResult,
    WebPage,
    Paper,
    RealTimeData,
    get_internet,
    search,
    fetch,
    search_papers,
)

# Multi-Modal Engine
from scio.multimodal import (
    UnifiedMultiModal,
    ModalityType,
    MultiModalResult,
    get_multimodal,
)

# Agent Swarm
from scio.swarm import (
    AgentSwarm,
    SwarmAgent,
    AgentRole,
    SwarmResult,
    get_swarm,
)

# Meta-Cognition
from scio.consciousness.meta_cognition import MetaCognitionEngine

# SCIO Ultimate (Central Orchestrator)
from scio.ultimate import (
    SCIOUltimate,
    SCIOCapabilities,
    get_scio,
    initialize,
    solve,
    understand,
    create,
)

__all__ = [
    # Meta
    "__version__",
    "__author__",
    # Core
    "Config",
    "get_config",
    "get_logger",
    "generate_id",
    "now_utc",
    # Agents
    "Agent",
    "AgentConfig",
    "AgentRegistry",
    "register_agent",
    # Tools
    "Tool",
    "ToolRegistry",
    "register_tool",
    # Knowledge
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeGraph",
    "Entity",
    "Relation",
    "ReasoningEngine",
    "TextEmbedder",
    # Hardware
    "SYSTEM_PROFILE",
    "OptimalSettings",
    "HardwareDetector",
    "HardwareMonitor",
    "get_monitor",
    "GPUManager",
    "get_gpu_manager",
    # Execution
    "ExecutionEngine",
    "CheckpointManager",
    # Protocols
    "Message",
    "Request",
    "Response",
    "Event",
    "EventBus",
    "ExecutionContext",
    "ExecutionState",
    # ========================================
    # SCIO ULTIMATE EXPORTS
    # ========================================
    # Advanced Reasoning
    "AdvancedReasoning",
    "ReasoningStrategy",
    "ReasoningResult",
    "TreeOfThought",
    "ChainOfThought",
    "ReActReasoner",
    "MCTSPlanner",
    "SelfReflection",
    # Persistent Memory
    "PersistentMemory",
    "Memory",
    "MemoryType",
    "get_memory",
    # Internet Knowledge
    "InternetKnowledge",
    "SearchResult",
    "WebPage",
    "Paper",
    "RealTimeData",
    "get_internet",
    "search",
    "fetch",
    "search_papers",
    # Multi-Modal Engine
    "UnifiedMultiModal",
    "ModalityType",
    "MultiModalResult",
    "get_multimodal",
    # Agent Swarm
    "AgentSwarm",
    "SwarmAgent",
    "AgentRole",
    "SwarmResult",
    "get_swarm",
    # Meta-Cognition
    "MetaCognitionEngine",
    # SCIO Ultimate
    "SCIOUltimate",
    "SCIOCapabilities",
    "get_scio",
    "initialize",
    "solve",
    "understand",
    "create",
]
