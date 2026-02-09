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
from scio.agents.master_agent import (
    MasterAgent,
    AgentCapability,
    AgentTask,
    AgentThought,
    AgentMemory,
    create_master_agent,
    create_research_agent,
    create_trading_agent,
    create_content_agent,
    create_automation_agent,
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
    # Master Agent
    "MasterAgent",
    "AgentCapability",
    "AgentTask",
    "AgentThought",
    "AgentMemory",
    "create_master_agent",
    "create_research_agent",
    "create_trading_agent",
    "create_content_agent",
    "create_automation_agent",
]
