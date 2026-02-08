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
]
