"""
SCIO Retrieval System

Semantische und hybride Wissensabfrage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from scio.core.logging import get_logger
from scio.knowledge.base import KnowledgeBase, KnowledgeEntry, KnowledgeQuery, KnowledgeResult
from scio.knowledge.embeddings import EmbeddingEngine, TextEmbedder

logger = get_logger(__name__)


class RetrievalStrategy(str, Enum):
    """Retrieval-Strategien"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    """Konfiguration fuer Retrieval"""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    min_score: float = 0.0
    rerank_top_k: int = 50
    use_mmr: bool = False
    mmr_lambda: float = 0.5


@dataclass
class RetrievalResult:
    """Ergebnis einer Retrieval-Anfrage"""
    entries: List[KnowledgeResult]
    query: str
    strategy: RetrievalStrategy
    total_candidates: int
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def top_score(self) -> float:
        return self.entries[0].score if self.entries else 0.0


class SemanticRetriever:
    """Semantischer Retriever basierend auf Vektor-Aehnlichkeit"""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedder: Optional[EmbeddingEngine] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.kb = knowledge_base
        self.embedder = embedder or TextEmbedder()
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[KnowledgeQuery] = None,
    ) -> RetrievalResult:
        """Fuehrt semantische Suche durch"""
        import time
        start = time.perf_counter()

        k = top_k or self.config.top_k
        query_embedding = self.embedder.embed(query)

        candidates: List[Tuple[str, float]] = []
        for entry_id, embedding in self.kb._embeddings.items():
            entry = self.kb._entries.get(entry_id)
            if not entry:
                continue
            if filters and not filters.matches_entry(entry):
                continue

            similarity = float(np.dot(query_embedding, embedding))
            if similarity >= self.config.min_score:
                candidates.append((entry_id, similarity))

        total_candidates = len(candidates)
        candidates.sort(key=lambda x: x[1], reverse=True)

        if self.config.use_mmr:
            selected = self._apply_mmr(query_embedding, candidates, k)
        else:
            selected = candidates[:k]

        results = []
        for entry_id, score in selected:
            entry = self.kb._entries[entry_id]
            results.append(KnowledgeResult(entry=entry, score=score))

        elapsed = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            entries=results,
            query=query,
            strategy=RetrievalStrategy.SEMANTIC,
            total_candidates=total_candidates,
            processing_time_ms=elapsed,
        )

    def _apply_mmr(
        self,
        query_embedding: np.ndarray,
        candidates: List[Tuple[str, float]],
        k: int,
    ) -> List[Tuple[str, float]]:
        """Maximal Marginal Relevance fuer diverse Ergebnisse"""
        if len(candidates) <= k:
            return candidates

        selected = []
        remaining = dict(candidates[:self.config.rerank_top_k])
        lam = self.config.mmr_lambda

        while len(selected) < k and remaining:
            best_id = None
            best_score = float('-inf')

            for entry_id, relevance in remaining.items():
                relevance_score = lam * relevance

                if selected:
                    entry_emb = self.kb._embeddings[entry_id]
                    max_sim = max(
                        float(np.dot(entry_emb, self.kb._embeddings[sel_id]))
                        for sel_id, _ in selected
                    )
                    diversity_score = (1 - lam) * max_sim
                else:
                    diversity_score = 0

                mmr_score = relevance_score - diversity_score
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_id = entry_id

            if best_id:
                selected.append((best_id, remaining[best_id]))
                del remaining[best_id]

        return selected


class HybridRetriever:
    """Hybrider Retriever kombiniert semantische und Keyword-Suche"""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedder: Optional[EmbeddingEngine] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.kb = knowledge_base
        self.embedder = embedder or TextEmbedder()
        self.config = config or RetrievalConfig()
        self.semantic = SemanticRetriever(knowledge_base, embedder, config)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[KnowledgeQuery] = None,
    ) -> RetrievalResult:
        """Fuehrt hybride Suche durch"""
        import time
        start = time.perf_counter()

        k = top_k or self.config.top_k

        # Semantische Suche
        sem_result = self.semantic.retrieve(query, self.config.rerank_top_k, filters)
        sem_scores = {r.entry.id: r.score for r in sem_result.entries}

        # Keyword-Suche
        kw_entries = self.kb.keyword_search(query, self.config.rerank_top_k)
        kw_scores = {e.id: 1.0 / (i + 1) for i, e in enumerate(kw_entries)}

        # Kombiniere
        all_ids = set(sem_scores.keys()) | set(kw_scores.keys())
        combined = {}
        for entry_id in all_ids:
            sem = sem_scores.get(entry_id, 0)
            kw = kw_scores.get(entry_id, 0)
            combined[entry_id] = (
                self.config.semantic_weight * sem +
                self.config.keyword_weight * kw
            )

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for entry_id, score in sorted_results:
            entry = self.kb._entries.get(entry_id)
            if entry:
                results.append(KnowledgeResult(entry=entry, score=score))

        elapsed = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            entries=results,
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            total_candidates=len(all_ids),
            processing_time_ms=elapsed,
        )


__all__ = [
    'RetrievalStrategy',
    'RetrievalConfig',
    'RetrievalResult',
    'SemanticRetriever',
    'HybridRetriever',
]
