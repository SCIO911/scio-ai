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
from scio.knowledge.ultimate_knowledge import (
    UltimateKnowledgeBase,
    KnowledgeDomain,
    KnowledgeEntry as UltimateKnowledgeEntry,
    get_knowledge,
)
from scio.knowledge.mega_knowledge import (
    MegaKnowledge,
    MoneyMethod,
    TechStack,
    ExpertiseLevel,
    get_mega_knowledge,
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
    # Ultimate Knowledge
    "UltimateKnowledgeBase",
    "KnowledgeDomain",
    "UltimateKnowledgeEntry",
    "get_knowledge",
    # Mega Knowledge
    "MegaKnowledge",
    "MoneyMethod",
    "TechStack",
    "ExpertiseLevel",
    "get_mega_knowledge",
]
