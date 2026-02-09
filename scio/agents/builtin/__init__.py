"""
SCIO Builtin Agents

Standard-Agenten die mit SCIO mitgeliefert werden.
"""

from scio.agents.builtin.data_loader import DataLoaderAgent, DataLoaderConfig
from scio.agents.builtin.analyzer import AnalyzerAgent, AnalyzerConfig
from scio.agents.builtin.reporter import ReporterAgent, ReporterConfig
from scio.agents.builtin.llm_agent import LLMAgent, LLMConfig
from scio.agents.builtin.transformer import TransformerAgent, TransformerConfig
from scio.agents.builtin.python_expert import PythonExpertAgent, PythonExpertConfig
from scio.agents.builtin.database_agent import DatabaseAgent, DatabaseConfig
from scio.agents.builtin.api_agent import APIAgent, APIConfig, WebhookAgent

__all__ = [
    "DataLoaderAgent",
    "DataLoaderConfig",
    "AnalyzerAgent",
    "AnalyzerConfig",
    "ReporterAgent",
    "ReporterConfig",
    "LLMAgent",
    "LLMConfig",
    "TransformerAgent",
    "TransformerConfig",
    "PythonExpertAgent",
    "PythonExpertConfig",
    "DatabaseAgent",
    "DatabaseConfig",
    "APIAgent",
    "APIConfig",
    "WebhookAgent",
]
