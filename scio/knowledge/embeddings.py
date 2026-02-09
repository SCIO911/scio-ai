"""
SCIO Embedding System

Vektoreinbettungen für semantische Ähnlichkeitssuche.
"""

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from scio.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Konfiguration für Embedding-Engines."""

    model_name: str = "default"
    embedding_dim: int = 384
    max_length: int = 512
    normalize: bool = True
    batch_size: int = 32
    cache_embeddings: bool = True
    cache_path: Optional[Path] = None


class EmbeddingEngine(ABC):
    """Abstrakte Basis für Embedding-Engines."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._cache: dict[str, np.ndarray] = {}

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Erzeugt ein Embedding für einen Text."""
        pass

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Erzeugt Embeddings für mehrere Texte."""
        return [self.embed(t) for t in texts]

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Berechnet die Cosine-Ähnlichkeit zwischen zwei Embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))

    def _get_cache_key(self, text: str) -> str:
        """Generiert einen Cache-Key für einen Text."""
        return hashlib.md5(text.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Leert den Embedding-Cache."""
        self._cache.clear()


class TextEmbedder(EmbeddingEngine):
    """
    Leistungsstarker Text-Embedder mit mehreren Strategien.

    Verwendet eine Kombination aus:
    - TF-IDF ähnlichen Gewichtungen
    - N-Gram Features
    - Positionscodierung
    - Semantische Cluster
    """

    # Vordefinierte semantische Cluster für bessere Embeddings
    SEMANTIC_CLUSTERS = {
        "positiv": ["gut", "super", "toll", "wunderbar", "exzellent", "perfekt", "great", "good", "excellent"],
        "negativ": ["schlecht", "falsch", "fehler", "problem", "bad", "wrong", "error", "fail"],
        "zeit": ["zeit", "datum", "tag", "stunde", "minute", "time", "date", "hour", "day"],
        "menge": ["viel", "wenig", "mehr", "weniger", "alle", "keine", "many", "few", "all", "none"],
        "aktion": ["machen", "tun", "erstellen", "löschen", "do", "make", "create", "delete", "run"],
        "daten": ["daten", "datei", "speicher", "laden", "data", "file", "storage", "load", "save"],
        "code": ["funktion", "klasse", "variable", "methode", "function", "class", "method", "code"],
        "math": ["berechnen", "addieren", "multiplizieren", "calculate", "add", "multiply", "sum"],
        "logik": ["wenn", "dann", "sonst", "und", "oder", "if", "then", "else", "and", "or"],
        "system": ["system", "prozess", "thread", "speicher", "process", "memory", "cpu", "gpu"],
    }

    # Stoppwörter für TF-IDF
    STOPWORDS = {
        "der", "die", "das", "und", "oder", "aber", "in", "an", "auf", "mit", "für", "von",
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "with", "for", "of", "to",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "ein", "eine", "einer", "eines", "ist", "sind", "war", "waren", "hat", "haben",
    }

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        super().__init__(config)
        self._idf_cache: dict[str, float] = {}
        self._doc_count = 0

        # Initialisiere Cluster-Vektoren
        self._cluster_vectors = self._init_cluster_vectors()

    def _init_cluster_vectors(self) -> dict[str, np.ndarray]:
        """Initialisiert Vektoren für semantische Cluster."""
        vectors = {}
        for cluster_name, words in self.SEMANTIC_CLUSTERS.items():
            # Erzeuge deterministischen Vektor für jeden Cluster
            vec = np.zeros(self.config.embedding_dim // 4, dtype=np.float32)
            for word in words:
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                for i in range(len(vec)):
                    vec[i] += ((h >> i) & 1) - 0.5
            vec /= (np.linalg.norm(vec) + 1e-8)
            vectors[cluster_name] = vec
        return vectors

    def _tokenize(self, text: str) -> list[str]:
        """Tokenisiert einen Text."""
        text = text.lower()
        # Entferne Sonderzeichen, behalte Buchstaben und Zahlen
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def _get_ngrams(self, tokens: list[str], n: int = 2) -> list[str]:
        """Generiert N-Gramme aus Tokens."""
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def embed(self, text: str) -> np.ndarray:
        """
        Erzeugt ein reichhaltiges Embedding für einen Text.

        Das Embedding besteht aus vier Teilen:
        1. Unigram-Features (TF-IDF-ähnlich)
        2. Bigram-Features
        3. Semantische Cluster-Aktivierungen
        4. Positions- und Längenfeatures
        """
        cache_key = self._get_cache_key(text)
        if self.config.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]

        dim = self.config.embedding_dim
        quarter = dim // 4
        embedding = np.zeros(dim, dtype=np.float32)

        tokens = self._tokenize(text)
        if not tokens:
            if self.config.cache_embeddings:
                self._cache[cache_key] = embedding
            return embedding

        # Teil 1: Unigram-Features
        unigram_vec = np.zeros(quarter, dtype=np.float32)
        token_counts: dict[str, int] = {}
        for token in tokens:
            if token not in self.STOPWORDS:
                token_counts[token] = token_counts.get(token, 0) + 1

        for token, count in token_counts.items():
            # Deterministischer Hash -> Position
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            positions = [(h >> (i * 4)) % quarter for i in range(4)]
            tf = math.log(1 + count)  # TF-Gewichtung
            for pos in positions:
                unigram_vec[pos] += tf * (1 if (h >> pos) & 1 else -1)

        embedding[:quarter] = unigram_vec

        # Teil 2: Bigram-Features
        bigram_vec = np.zeros(quarter, dtype=np.float32)
        bigrams = self._get_ngrams(tokens, 2)
        for bigram in bigrams:
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            pos = h % quarter
            bigram_vec[pos] += 1 if (h >> 16) & 1 else -1

        embedding[quarter:2*quarter] = bigram_vec

        # Teil 3: Semantische Cluster-Aktivierungen
        cluster_vec = np.zeros(quarter, dtype=np.float32)
        token_set = set(tokens)
        for i, (cluster_name, words) in enumerate(self.SEMANTIC_CLUSTERS.items()):
            overlap = len(token_set & set(words))
            if overlap > 0:
                # Aktiviere Cluster-Vektor
                cluster_v = self._cluster_vectors[cluster_name]
                activation = math.log(1 + overlap)
                start = (i * len(cluster_v)) % quarter
                end = min(start + len(cluster_v), quarter)
                cluster_vec[start:end] += cluster_v[:end-start] * activation

        embedding[2*quarter:3*quarter] = cluster_vec

        # Teil 4: Meta-Features (Länge, Komplexität, etc.)
        meta_vec = np.zeros(quarter, dtype=np.float32)

        # Textlänge-Feature
        meta_vec[0] = math.log(1 + len(text))
        # Token-Count-Feature
        meta_vec[1] = math.log(1 + len(tokens))
        # Unique-Token-Ratio
        meta_vec[2] = len(set(tokens)) / max(len(tokens), 1)
        # Durchschnittliche Wortlänge
        meta_vec[3] = sum(len(t) for t in tokens) / max(len(tokens), 1) / 10
        # Anzahl Zahlen
        meta_vec[4] = sum(1 for t in tokens if any(c.isdigit() for c in t)) / max(len(tokens), 1)
        # Großbuchstaben-Ratio
        meta_vec[5] = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        embedding[3*quarter:] = meta_vec

        # Normalisiere falls gewünscht
        if self.config.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

        if self.config.cache_embeddings:
            self._cache[cache_key] = embedding

        return embedding


class DocumentEmbedder(EmbeddingEngine):
    """
    Dokumenten-Embedder für lange Texte.

    Teilt Dokumente in Chunks und aggregiert deren Embeddings.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ):
        super().__init__(config)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_embedder = TextEmbedder(config)

    def _chunk_text(self, text: str) -> list[str]:
        """Teilt Text in überlappende Chunks."""
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk = " ".join(words[i:end])
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap

        return chunks if chunks else [text]

    def embed(self, text: str) -> np.ndarray:
        """Erzeugt ein Embedding für ein langes Dokument."""
        cache_key = self._get_cache_key(text)
        if self.config.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]

        chunks = self._chunk_text(text)

        if len(chunks) == 1:
            embedding = self._text_embedder.embed(chunks[0])
        else:
            # Aggregiere Chunk-Embeddings mit Positionsgewichtung
            embeddings = []
            weights = []
            for i, chunk in enumerate(chunks):
                emb = self._text_embedder.embed(chunk)
                embeddings.append(emb)
                # Gewichtung: Anfang und Ende wichtiger
                if i == 0 or i == len(chunks) - 1:
                    weights.append(1.5)
                else:
                    weights.append(1.0)

            # Gewichteter Durchschnitt
            embedding = np.average(embeddings, axis=0, weights=weights)

            if self.config.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm

        if self.config.cache_embeddings:
            self._cache[cache_key] = embedding

        return embedding

    def embed_with_chunks(self, text: str) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
        """Gibt sowohl das Dokument-Embedding als auch die Chunk-Embeddings zurück."""
        chunks = self._chunk_text(text)
        chunk_embeddings = [(chunk, self._text_embedder.embed(chunk)) for chunk in chunks]

        if len(chunk_embeddings) == 1:
            doc_embedding = chunk_embeddings[0][1]
        else:
            doc_embedding = np.mean([ce[1] for ce in chunk_embeddings], axis=0)
            if self.config.normalize:
                norm = np.linalg.norm(doc_embedding)
                if norm > 0:
                    doc_embedding /= norm

        return doc_embedding, chunk_embeddings


class CodeEmbedder(EmbeddingEngine):
    """
    Spezialisierter Embedder für Quellcode.

    Berücksichtigt:
    - Syntaxstrukturen
    - Identifier-Namen
    - Kontrollfluss-Muster
    - Kommentare
    """

    # Code-spezifische Keywords
    CODE_PATTERNS = {
        "control": ["if", "else", "elif", "for", "while", "try", "except", "with", "match", "case"],
        "define": ["def", "class", "lambda", "async", "await", "function", "const", "let", "var"],
        "return": ["return", "yield", "raise", "throw", "break", "continue", "pass"],
        "import": ["import", "from", "include", "require", "use", "using"],
        "type": ["int", "str", "float", "bool", "list", "dict", "tuple", "set", "None", "null"],
        "operator": ["and", "or", "not", "in", "is", "isinstance", "type", "len"],
    }

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        super().__init__(config)
        self._text_embedder = TextEmbedder(config)

    def _extract_identifiers(self, code: str) -> list[str]:
        """Extrahiert Identifier aus Code."""
        # Einfache Regex für Python/JS-ähnliche Identifier
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        return re.findall(pattern, code)

    def _extract_strings(self, code: str) -> list[str]:
        """Extrahiert String-Literale."""
        patterns = [
            r'"([^"\\]*(\\.[^"\\]*)*)"',
            r"'([^'\\]*(\\.[^'\\]*)*)'",
            r'"""(.*?)"""',
            r"'''(.*?)'''",
        ]
        strings = []
        for pattern in patterns:
            strings.extend(re.findall(pattern, code, re.DOTALL))
        return [s[0] if isinstance(s, tuple) else s for s in strings]

    def embed(self, code: str) -> np.ndarray:
        """Erzeugt ein Embedding für Quellcode."""
        cache_key = self._get_cache_key(code)
        if self.config.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]

        dim = self.config.embedding_dim
        third = dim // 3
        embedding = np.zeros(dim, dtype=np.float32)

        # Teil 1: Text-Embedding des gesamten Codes
        text_emb = self._text_embedder.embed(code)
        embedding[:third] = text_emb[:third]

        # Teil 2: Code-Pattern-Features
        pattern_vec = np.zeros(third, dtype=np.float32)
        identifiers = self._extract_identifiers(code)

        for i, (pattern_name, keywords) in enumerate(self.CODE_PATTERNS.items()):
            count = sum(1 for ident in identifiers if ident.lower() in keywords)
            pattern_vec[i % third] += math.log(1 + count)

        embedding[third:2*third] = pattern_vec

        # Teil 3: Struktur-Features
        struct_vec = np.zeros(third, dtype=np.float32)

        lines = code.split('\n')
        struct_vec[0] = math.log(1 + len(lines))  # Zeilenzahl
        struct_vec[1] = math.log(1 + len(identifiers))  # Identifier-Anzahl
        struct_vec[2] = len(set(identifiers)) / max(len(identifiers), 1)  # Unique Ratio
        struct_vec[3] = code.count('(') + code.count('[') + code.count('{')  # Klammern
        struct_vec[4] = code.count('def ') + code.count('class ')  # Definitionen
        struct_vec[5] = code.count('#') + code.count('//')  # Kommentare

        # Einrückungstiefe
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        struct_vec[6] = max(indents) / 4 if indents else 0
        struct_vec[7] = sum(indents) / max(len(indents), 1) / 4 if indents else 0

        embedding[2*third:] = struct_vec[:third]

        if self.config.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

        if self.config.cache_embeddings:
            self._cache[cache_key] = embedding

        return embedding


def create_embedder(
    embedder_type: str = "text",
    config: Optional[EmbeddingConfig] = None,
) -> EmbeddingEngine:
    """Factory-Funktion für Embedder."""
    embedders = {
        "text": TextEmbedder,
        "document": DocumentEmbedder,
        "code": CodeEmbedder,
    }

    if embedder_type not in embedders:
        raise ValueError(f"Unknown embedder type: {embedder_type}")

    return embedders[embedder_type](config)
