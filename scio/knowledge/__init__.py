"""
SCIO Knowledge System

Mächtiges Wissensmanagementsystem mit Vektoreinbettungen,
semantischer Suche, Knowledge Graph und Echtzeit-Internet-Zugang.
"""

from scio.knowledge.base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeResult,
)
from scio.knowledge.embeddings import (
    EmbeddingEngine,
    TextEmbedder,
    DocumentEmbedder,
)
from scio.knowledge.graph import (
    KnowledgeGraph,
    Entity,
    Relation,
    Triple,
)
from scio.knowledge.retrieval import (
    SemanticRetriever,
    HybridRetriever,
    RetrievalResult,
)
from scio.knowledge.reasoning import (
    ReasoningEngine,
    InferenceResult,
    LogicRule,
)
from scio.knowledge.internet_access import (
    InternetKnowledge,
    InternetConfig,
    SearchResult,
    WebPage,
    Paper,
    RealTimeData,
    SearchEngine,
    get_internet,
    search,
    fetch,
    search_papers,
)

__all__ = [
    # Base
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeQuery",
    "KnowledgeResult",
    # Embeddings
    "EmbeddingEngine",
    "TextEmbedder",
    "DocumentEmbedder",
    # Graph
    "KnowledgeGraph",
    "Entity",
    "Relation",
    "Triple",
    # Retrieval
    "SemanticRetriever",
    "HybridRetriever",
    "RetrievalResult",
    # Reasoning
    "ReasoningEngine",
    "InferenceResult",
    "LogicRule",
    # Internet Access
    "InternetKnowledge",
    "InternetConfig",
    "SearchResult",
    "WebPage",
    "Paper",
    "RealTimeData",
    "SearchEngine",
    "get_internet",
    "search",
    "fetch",
    "search_papers",
]
