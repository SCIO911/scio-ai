"""
SCIO Agent Swarm Module

Multi-Agent Schwarm-Intelligenz mit:
- Spezialisierte Agenten (Researcher, Analyst, Coder, etc.)
- Koordinierte Zusammenarbeit
- Emergentes Verhalten
- Selbst-Organisation
"""

from scio.swarm.agent_swarm import (
    AgentSwarm,
    SwarmAgent,
    AgentRole,
    SwarmTask,
    SwarmResult,
    SwarmConfig,
    get_swarm,
)

__all__ = [
    "AgentSwarm",
    "SwarmAgent",
    "AgentRole",
    "SwarmTask",
    "SwarmResult",
    "SwarmConfig",
    "get_swarm",
]
