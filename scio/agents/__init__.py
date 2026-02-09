"""SCIO Agents - Agenten-System für autonome Aufgabenausführung."""

from scio.agents.base import Agent, AgentConfig, AgentState
from scio.agents.registry import AgentRegistry, register_agent

# Import builtin agents to trigger registration
from scio.agents.builtin import (
    DataLoaderAgent,
    AnalyzerAgent,
    ReporterAgent,
    TransformerAgent,
    LLMAgent,
    PythonExpertAgent,
)
from scio.agents.finance_advisor import (
    FinanceAdvisor,
    FinanceAdvice,
    AdvisorTopic,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentState",
    "AgentRegistry",
    "register_agent",
    "DataLoaderAgent",
    "AnalyzerAgent",
    "ReporterAgent",
    "TransformerAgent",
    "LLMAgent",
    "PythonExpertAgent",
    # Finance
    "FinanceAdvisor",
    "FinanceAdvice",
    "AdvisorTopic",
]
