"""
SCIO Internet Access - Echtzeit-Wissen aus dem Internet

Ermöglicht SCIO den Zugriff auf:
- Web-Suche (Google, Bing, DuckDuckGo)
- Webseiten-Inhalte
- Wissenschaftliche Datenbanken (ArXiv, PubMed, Semantic Scholar)
- Echtzeit-Daten (Wetter, Börse, News)
- APIs und externe Services
"""

import asyncio
import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import quote_plus, urljoin, urlparse
import html

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class SearchEngine(str, Enum):
    """Verfügbare Suchmaschinen."""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    GITHUB = "github"
    WIKIPEDIA = "wikipedia"


class DataSource(str, Enum):
    """Verfügbare Datenquellen."""
    WEB = "web"
    NEWS = "news"
    ACADEMIC = "academic"
    FINANCIAL = "financial"
    WEATHER = "weather"
    SOCIAL = "social"
    CODE = "code"


@dataclass
class SearchResult:
    """Ein Suchergebnis."""

    title: str
    url: str
    snippet: str
    source: SearchEngine
    rank: int = 0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source.value,
            "rank": self.rank,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
            "fetched_at": self.fetched_at.isoformat(),
        }


@dataclass
class WebPage:
    """Eine abgerufene Webseite."""

    url: str
    title: str
    content: str  # Bereinigter Text
    html: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=now_utc)
    status_code: int = 200

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class Paper:
    """Ein wissenschaftliches Paper."""

    id: str
    title: str
    abstract: str
    authors: List[str]
    published_date: Optional[datetime] = None
    url: str = ""
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    citations: int = 0
    source: SearchEngine = SearchEngine.ARXIV
    categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "citations": self.citations,
            "source": self.source.value,
            "categories": self.categories,
        }


@dataclass
class RealTimeData:
    """Echtzeit-Daten."""

    data_type: str  # weather, stock, crypto, news
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=now_utc)
    source: str = ""
    is_fresh: bool = True

    @property
    def age_seconds(self) -> float:
        return (now_utc() - self.timestamp).total_seconds()


@dataclass
class InternetConfig:
    """Konfiguration für Internet-Zugang."""

    # HTTP
    timeout_seconds: float = 30.0
    max_retries: int = 3
    user_agent: str = "SCIO/1.0 (AI Agent; +https://scio.ai)"

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 Stunde
    max_cache_size: int = 1000

    # Rate Limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60

    # Content
    max_content_length: int = 1_000_000  # 1MB
    extract_text: bool = True
    follow_redirects: bool = True


# ============================================================================
# CACHE
# ============================================================================

class InternetCache:
    """Cache für Internet-Anfragen."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

    def _make_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            value, timestamp = self._cache[cache_key]
            if (now_utc() - timestamp).total_seconds() < self.ttl_seconds:
                return value
            else:
                del self._cache[cache_key]
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self._cache) >= self.max_size:
            # Remove oldest entries
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1][1]
            )
            for k, _ in sorted_items[:len(sorted_items) // 4]:
                del self._cache[k]

        cache_key = self._make_key(key)
        self._cache[cache_key] = (value, now_utc())

    def clear(self) -> None:
        self._cache.clear()


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Rate Limiter für API-Anfragen."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self._timestamps: List[datetime] = []

    async def acquire(self) -> None:
        """Wartet bis eine Anfrage erlaubt ist."""
        now = now_utc()
        minute_ago = now - timedelta(minutes=1)

        # Remove old timestamps
        self._timestamps = [ts for ts in self._timestamps if ts > minute_ago]

        if len(self._timestamps) >= self.requests_per_minute:
            # Wait until oldest request is more than a minute old
            wait_time = (self._timestamps[0] - minute_ago).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self._timestamps.append(now)


# ============================================================================
# HTTP CLIENT (ABSTRACT)
# ============================================================================

class HTTPClient(ABC):
    """Abstrakte HTTP Client Klasse."""

    @abstractmethod
    async def get(self, url: str, headers: Optional[Dict] = None) -> Tuple[int, str]:
        """GET Request."""
        pass

    @abstractmethod
    async def post(self, url: str, data: Dict, headers: Optional[Dict] = None) -> Tuple[int, str]:
        """POST Request."""
        pass


class SimpleHTTPClient(HTTPClient):
    """
    Einfacher HTTP Client.

    In der echten Implementierung würde man aiohttp oder httpx verwenden.
    Hier ist ein Platzhalter, der das Interface definiert.
    """

    def __init__(self, config: InternetConfig):
        self.config = config

    async def get(self, url: str, headers: Optional[Dict] = None) -> Tuple[int, str]:
        """GET Request (Platzhalter-Implementierung)."""
        # In echter Implementierung:
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(url, headers=headers) as response:
        #         return response.status, await response.text()

        # Platzhalter
        return 200, f"<html><body>Content from {url}</body></html>"

    async def post(self, url: str, data: Dict, headers: Optional[Dict] = None) -> Tuple[int, str]:
        """POST Request (Platzhalter-Implementierung)."""
        return 200, json.dumps({"result": "success"})


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

class TextExtractor:
    """Extrahiert Text aus HTML."""

    # Tags die entfernt werden sollen
    REMOVE_TAGS = {'script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript'}

    @classmethod
    def extract(cls, html_content: str) -> str:
        """Extrahiert lesbaren Text aus HTML."""
        # Entferne Script und Style Tags
        for tag in cls.REMOVE_TAGS:
            html_content = re.sub(
                f'<{tag}[^>]*>.*?</{tag}>',
                '',
                html_content,
                flags=re.DOTALL | re.IGNORECASE
            )

        # Entferne alle HTML Tags
        text = re.sub(r'<[^>]+>', ' ', html_content)

        # Dekodiere HTML Entities
        text = html.unescape(text)

        # Normalisiere Whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    @classmethod
    def extract_links(cls, html_content: str, base_url: str) -> List[str]:
        """Extrahiert alle Links aus HTML."""
        links = []
        pattern = r'href=["\']([^"\']+)["\']'

        for match in re.finditer(pattern, html_content):
            href = match.group(1)
            if href.startswith('http'):
                links.append(href)
            elif href.startswith('/'):
                links.append(urljoin(base_url, href))

        return list(set(links))

    @classmethod
    def extract_title(cls, html_content: str) -> str:
        """Extrahiert den Titel."""
        match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1).strip())
        return ""


# ============================================================================
# INTERNET KNOWLEDGE
# ============================================================================

class InternetKnowledge:
    """
    Hauptklasse für Internet-Zugang.

    Bietet einheitlichen Zugang zu:
    - Web-Suche
    - Webseiten-Inhalte
    - Wissenschaftliche Datenbanken
    - Echtzeit-Daten
    """

    def __init__(self, config: Optional[InternetConfig] = None):
        self.config = config or InternetConfig()
        self.http = SimpleHTTPClient(self.config)
        self.cache = InternetCache(
            max_size=self.config.max_cache_size,
            ttl_seconds=self.config.cache_ttl_seconds,
        )
        self.rate_limiter = RateLimiter(self.config.requests_per_minute)

        logger.info("InternetKnowledge initialized")

    # ========================================================================
    # WEB SEARCH
    # ========================================================================

    async def search_web(
        self,
        query: str,
        engine: SearchEngine = SearchEngine.DUCKDUCKGO,
        num_results: int = 10,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Führt eine Web-Suche durch.

        Args:
            query: Suchanfrage
            engine: Suchmaschine
            num_results: Anzahl Ergebnisse

        Returns:
            Liste von SearchResult
        """
        cache_key = f"search:{engine.value}:{query}:{num_results}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Search cache hit", query=query)
            return cached

        await self.rate_limiter.acquire()

        results = []

        if engine == SearchEngine.DUCKDUCKGO:
            results = await self._search_duckduckgo(query, num_results)
        elif engine == SearchEngine.GOOGLE:
            results = await self._search_google(query, num_results)
        elif engine == SearchEngine.BING:
            results = await self._search_bing(query, num_results)
        elif engine == SearchEngine.WIKIPEDIA:
            results = await self._search_wikipedia(query, num_results)
        elif engine == SearchEngine.GITHUB:
            results = await self._search_github(query, num_results)
        else:
            logger.warning(f"Unknown search engine: {engine}")

        if results:
            self.cache.set(cache_key, results)

        return results

    async def _search_duckduckgo(self, query: str, num_results: int) -> List[SearchResult]:
        """DuckDuckGo Suche."""
        # Platzhalter-Implementierung
        # In echt würde man die DuckDuckGo API oder HTML scrapen
        encoded_query = quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded_query}"

        try:
            status, html_content = await self.http.get(url)
            if status == 200:
                # Parse results (vereinfacht)
                results = []
                for i in range(num_results):
                    results.append(SearchResult(
                        title=f"Result {i+1} for: {query}",
                        url=f"https://example.com/result{i+1}",
                        snippet=f"This is a sample snippet for search query: {query}",
                        source=SearchEngine.DUCKDUCKGO,
                        rank=i + 1,
                        relevance_score=1.0 - (i * 0.1),
                    ))
                return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")

        return []

    async def _search_google(self, query: str, num_results: int) -> List[SearchResult]:
        """Google Suche (würde API-Key benötigen)."""
        # Platzhalter
        return []

    async def _search_bing(self, query: str, num_results: int) -> List[SearchResult]:
        """Bing Suche (würde API-Key benötigen)."""
        # Platzhalter
        return []

    async def _search_wikipedia(self, query: str, num_results: int) -> List[SearchResult]:
        """Wikipedia Suche."""
        encoded_query = quote_plus(query)
        url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded_query}&limit={num_results}&format=json"

        try:
            status, content = await self.http.get(url)
            if status == 200:
                data = json.loads(content)
                results = []
                if len(data) >= 4:
                    titles, descriptions, urls = data[1], data[2], data[3]
                    for i, (title, desc, url) in enumerate(zip(titles, descriptions, urls)):
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=desc or f"Wikipedia article about {title}",
                            source=SearchEngine.WIKIPEDIA,
                            rank=i + 1,
                            relevance_score=1.0 - (i * 0.1),
                        ))
                return results
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")

        return []

    async def _search_github(self, query: str, num_results: int) -> List[SearchResult]:
        """GitHub Code/Repository Suche."""
        # Platzhalter - würde GitHub API verwenden
        return []

    # ========================================================================
    # FETCH WEBPAGE
    # ========================================================================

    async def fetch_webpage(
        self,
        url: str,
        extract_text: bool = True,
    ) -> Optional[WebPage]:
        """
        Ruft eine Webseite ab.

        Args:
            url: URL der Seite
            extract_text: Text extrahieren?

        Returns:
            WebPage Objekt oder None
        """
        cache_key = f"page:{url}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Page cache hit", url=url)
            return cached

        await self.rate_limiter.acquire()

        try:
            headers = {"User-Agent": self.config.user_agent}
            status, html_content = await self.http.get(url, headers)

            if status != 200:
                logger.warning(f"Failed to fetch {url}: status {status}")
                return None

            if len(html_content) > self.config.max_content_length:
                html_content = html_content[:self.config.max_content_length]

            # Extract content
            title = TextExtractor.extract_title(html_content)
            content = TextExtractor.extract(html_content) if extract_text else ""
            links = TextExtractor.extract_links(html_content, url)

            page = WebPage(
                url=url,
                title=title,
                content=content,
                html=html_content if not extract_text else None,
                links=links[:100],  # Limit links
                status_code=status,
            )

            self.cache.set(cache_key, page)
            return page

        except Exception as e:
            logger.error(f"Failed to fetch webpage: {e}")
            return None

    # ========================================================================
    # ACADEMIC SEARCH
    # ========================================================================

    async def search_academic(
        self,
        query: str,
        source: SearchEngine = SearchEngine.ARXIV,
        max_results: int = 10,
    ) -> List[Paper]:
        """
        Sucht wissenschaftliche Paper.

        Args:
            query: Suchanfrage
            source: ArXiv, PubMed, Semantic Scholar
            max_results: Maximale Anzahl

        Returns:
            Liste von Paper
        """
        cache_key = f"academic:{source.value}:{query}:{max_results}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        await self.rate_limiter.acquire()

        papers = []

        if source == SearchEngine.ARXIV:
            papers = await self._search_arxiv(query, max_results)
        elif source == SearchEngine.SEMANTIC_SCHOLAR:
            papers = await self._search_semantic_scholar(query, max_results)
        elif source == SearchEngine.PUBMED:
            papers = await self._search_pubmed(query, max_results)

        if papers:
            self.cache.set(cache_key, papers)

        return papers

    async def _search_arxiv(self, query: str, max_results: int) -> List[Paper]:
        """ArXiv Suche."""
        encoded_query = quote_plus(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"

        try:
            status, content = await self.http.get(url)
            if status == 200:
                # Parse Atom/XML feed (vereinfacht)
                papers = []
                # In echter Implementierung würde man XML parsen
                for i in range(min(max_results, 5)):
                    papers.append(Paper(
                        id=f"arxiv:{generate_id('paper')}",
                        title=f"Paper about: {query} ({i+1})",
                        abstract=f"This paper investigates {query}...",
                        authors=["Author A", "Author B"],
                        url=f"https://arxiv.org/abs/2024.{i:05d}",
                        source=SearchEngine.ARXIV,
                        categories=["cs.AI", "cs.LG"],
                    ))
                return papers
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")

        return []

    async def _search_semantic_scholar(self, query: str, max_results: int) -> List[Paper]:
        """Semantic Scholar Suche."""
        # Platzhalter
        return []

    async def _search_pubmed(self, query: str, max_results: int) -> List[Paper]:
        """PubMed Suche."""
        # Platzhalter
        return []

    # ========================================================================
    # REALTIME DATA
    # ========================================================================

    async def get_realtime_data(
        self,
        data_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[RealTimeData]:
        """
        Holt Echtzeit-Daten.

        Args:
            data_type: weather, stock, crypto, news
            params: Zusätzliche Parameter

        Returns:
            RealTimeData oder None
        """
        params = params or {}

        if data_type == "weather":
            return await self._get_weather(params.get("location", "Berlin"))
        elif data_type == "stock":
            return await self._get_stock(params.get("symbol", "AAPL"))
        elif data_type == "crypto":
            return await self._get_crypto(params.get("coin", "bitcoin"))
        elif data_type == "news":
            return await self._get_news(params.get("topic", "technology"))

        return None

    async def _get_weather(self, location: str) -> Optional[RealTimeData]:
        """Holt Wetterdaten."""
        # Platzhalter - würde OpenWeatherMap oder ähnlich verwenden
        return RealTimeData(
            data_type="weather",
            data={
                "location": location,
                "temperature": 20,
                "condition": "partly cloudy",
                "humidity": 65,
                "wind_speed": 10,
            },
            source="weather_api",
        )

    async def _get_stock(self, symbol: str) -> Optional[RealTimeData]:
        """Holt Börsendaten."""
        # Platzhalter - würde Alpha Vantage oder ähnlich verwenden
        return RealTimeData(
            data_type="stock",
            data={
                "symbol": symbol,
                "price": 150.50,
                "change": 2.30,
                "change_percent": 1.55,
                "volume": 75000000,
            },
            source="stock_api",
        )

    async def _get_crypto(self, coin: str) -> Optional[RealTimeData]:
        """Holt Kryptowährungsdaten."""
        # Platzhalter - würde CoinGecko oder ähnlich verwenden
        return RealTimeData(
            data_type="crypto",
            data={
                "coin": coin,
                "price_usd": 65000.00,
                "change_24h": 2.5,
                "market_cap": 1200000000000,
            },
            source="crypto_api",
        )

    async def _get_news(self, topic: str) -> Optional[RealTimeData]:
        """Holt Nachrichten."""
        # Platzhalter - würde News API verwenden
        return RealTimeData(
            data_type="news",
            data={
                "topic": topic,
                "articles": [
                    {
                        "title": f"Latest news about {topic}",
                        "source": "Example News",
                        "url": "https://example.com/news/1",
                    }
                ],
            },
            source="news_api",
        )

    # ========================================================================
    # API ACCESS
    # ========================================================================

    async def call_api(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Any]:
        """
        Universeller API-Aufruf.

        Args:
            url: API Endpoint
            method: GET, POST, PUT, DELETE
            headers: HTTP Headers
            data: Request Body (für POST/PUT)
            params: Query Parameters

        Returns:
            (status_code, response_data)
        """
        await self.rate_limiter.acquire()

        # Add query params to URL
        if params:
            param_str = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
            url = f"{url}?{param_str}" if "?" not in url else f"{url}&{param_str}"

        try:
            if method.upper() == "GET":
                status, content = await self.http.get(url, headers)
            elif method.upper() == "POST":
                status, content = await self.http.post(url, data or {}, headers)
            else:
                logger.warning(f"Unsupported HTTP method: {method}")
                return 405, {"error": "Method not supported"}

            # Try to parse as JSON
            try:
                return status, json.loads(content)
            except json.JSONDecodeError:
                return status, content

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return 500, {"error": str(e)}

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Convenience-Methode für Standard-Suche."""
        return await self.search_web(query, SearchEngine.DUCKDUCKGO, num_results)

    async def get_page_content(self, url: str) -> str:
        """Convenience-Methode für Seiten-Inhalt."""
        page = await self.fetch_webpage(url)
        return page.content if page else ""

    async def get_wiki_summary(self, topic: str) -> str:
        """Holt Wikipedia-Zusammenfassung."""
        results = await self.search_web(topic, SearchEngine.WIKIPEDIA, 1)
        if results:
            page = await self.fetch_webpage(results[0].url)
            if page:
                # Erste paar Sätze
                sentences = page.content.split(". ")
                return ". ".join(sentences[:5]) + "."
        return ""

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        return {
            "cache_size": len(self.cache._cache),
            "rate_limit_remaining": max(0, self.config.requests_per_minute - len(self.rate_limiter._timestamps)),
        }


# ============================================================================
# SINGLETON & CONVENIENCE
# ============================================================================

_default_internet: Optional[InternetKnowledge] = None


def get_internet(config: Optional[InternetConfig] = None) -> InternetKnowledge:
    """Gibt eine Singleton-Instanz zurück."""
    global _default_internet
    if _default_internet is None:
        _default_internet = InternetKnowledge(config)
    return _default_internet


async def search(query: str, num_results: int = 10) -> List[SearchResult]:
    """Convenience-Funktion für Web-Suche."""
    return await get_internet().search(query, num_results)


async def fetch(url: str) -> Optional[WebPage]:
    """Convenience-Funktion für Seiten-Abruf."""
    return await get_internet().fetch_webpage(url)


async def search_papers(query: str, max_results: int = 10) -> List[Paper]:
    """Convenience-Funktion für Paper-Suche."""
    return await get_internet().search_academic(query, max_results=max_results)
