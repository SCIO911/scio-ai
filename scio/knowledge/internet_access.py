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
    HTTP Client mit httpx für echte HTTP-Requests.

    Unterstützt:
    - Async GET/POST Requests
    - Automatisches Retry bei Fehlern
    - Timeout-Handling
    - SSL-Verifizierung
    """

    def __init__(self, config: InternetConfig):
        self.config = config
        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Lazy initialization des HTTP Clients."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.config.timeout),
                    follow_redirects=True,
                    verify=True,
                    headers={
                        "User-Agent": self.config.user_agent,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
                    }
                )
            except ImportError:
                logger.warning("httpx nicht installiert - verwende urllib als Fallback")
                self._client = "fallback"
        return self._client

    async def get(self, url: str, headers: Optional[Dict] = None) -> Tuple[int, str]:
        """
        Führt einen GET Request aus.

        Args:
            url: Die URL für den Request
            headers: Optionale zusätzliche Headers

        Returns:
            Tuple aus (status_code, response_text)
        """
        client = await self._get_client()

        if client == "fallback":
            return await self._fallback_get(url, headers)

        try:
            import httpx
            merged_headers = headers or {}
            response = await client.get(url, headers=merged_headers)
            return response.status_code, response.text

        except httpx.TimeoutException:
            logger.warning(f"Timeout bei GET {url}")
            return 408, f'{{"error": "Request timeout for {url}"}}'

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP Fehler bei GET {url}: {e.response.status_code}")
            return e.response.status_code, e.response.text

        except Exception as e:
            logger.error(f"GET Request fehlgeschlagen für {url}: {e}")
            return 500, f'{{"error": "{str(e)}"}}'

    async def post(self, url: str, data: Dict, headers: Optional[Dict] = None) -> Tuple[int, str]:
        """
        Führt einen POST Request aus.

        Args:
            url: Die URL für den Request
            data: Die zu sendenden Daten
            headers: Optionale zusätzliche Headers

        Returns:
            Tuple aus (status_code, response_text)
        """
        client = await self._get_client()

        if client == "fallback":
            return await self._fallback_post(url, data, headers)

        try:
            import httpx
            merged_headers = {"Content-Type": "application/json"}
            if headers:
                merged_headers.update(headers)

            response = await client.post(url, json=data, headers=merged_headers)
            return response.status_code, response.text

        except httpx.TimeoutException:
            logger.warning(f"Timeout bei POST {url}")
            return 408, f'{{"error": "Request timeout for {url}"}}'

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP Fehler bei POST {url}: {e.response.status_code}")
            return e.response.status_code, e.response.text

        except Exception as e:
            logger.error(f"POST Request fehlgeschlagen für {url}: {e}")
            return 500, f'{{"error": "{str(e)}"}}'

    async def _fallback_get(self, url: str, headers: Optional[Dict] = None) -> Tuple[int, str]:
        """Fallback GET mit urllib für den Fall dass httpx nicht verfügbar ist."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(url, headers=headers or {})
            req.add_header("User-Agent", self.config.user_agent)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=self.config.timeout)
            )
            return response.status, response.read().decode('utf-8', errors='replace')

        except urllib.error.HTTPError as e:
            return e.code, e.read().decode('utf-8', errors='replace')
        except urllib.error.URLError as e:
            return 503, f'{{"error": "Connection failed: {str(e)}"}}'
        except Exception as e:
            return 500, f'{{"error": "{str(e)}"}}'

    async def _fallback_post(self, url: str, data: Dict, headers: Optional[Dict] = None) -> Tuple[int, str]:
        """Fallback POST mit urllib."""
        import urllib.request
        import urllib.error

        try:
            json_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=json_data, headers=headers or {})
            req.add_header("User-Agent", self.config.user_agent)
            req.add_header("Content-Type", "application/json")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=self.config.timeout)
            )
            return response.status, response.read().decode('utf-8', errors='replace')

        except urllib.error.HTTPError as e:
            return e.code, e.read().decode('utf-8', errors='replace')
        except urllib.error.URLError as e:
            return 503, f'{{"error": "Connection failed: {str(e)}"}}'
        except Exception as e:
            return 500, f'{{"error": "{str(e)}"}}'

    async def close(self):
        """Schließt den HTTP Client."""
        if self._client and self._client != "fallback":
            await self._client.aclose()
            self._client = None


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
        """DuckDuckGo Suche mit echtem HTML-Parsing."""
        encoded_query = quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded_query}"

        try:
            status, html_content = await self.http.get(url)
            if status == 200 and html_content:
                results = []
                # Parse echte DuckDuckGo HTML-Ergebnisse
                import re
                # DuckDuckGo result links pattern
                link_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
                snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+)</a>'

                links = re.findall(link_pattern, html_content)
                snippets = re.findall(snippet_pattern, html_content)

                for i, (link_url, title) in enumerate(links[:num_results]):
                    snippet = snippets[i] if i < len(snippets) else ""
                    # DuckDuckGo redirect URLs dekodieren
                    if link_url.startswith("//duckduckgo.com/l/?"):
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link_url).query)
                        link_url = parsed.get("uddg", [link_url])[0]

                    results.append(SearchResult(
                        title=title.strip(),
                        url=link_url,
                        snippet=snippet.strip(),
                        source=SearchEngine.DUCKDUCKGO,
                        rank=i + 1,
                        relevance_score=1.0 - (i * 0.05),
                    ))

                if results:
                    return results
                logger.warning(f"DuckDuckGo: Keine Ergebnisse gefunden für '{query}'")

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")

        return []

    async def _search_google(self, query: str, num_results: int) -> List[SearchResult]:
        """Google Suche - erfordert API-Key Konfiguration."""
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if not api_key or not search_engine_id:
            logger.warning("Google Search nicht konfiguriert: GOOGLE_API_KEY und GOOGLE_SEARCH_ENGINE_ID erforderlich")
            return []

        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={quote_plus(query)}&num={num_results}"
        try:
            status, content = await self.http.get(url)
            if status == 200:
                data = json.loads(content)
                results = []
                for i, item in enumerate(data.get("items", [])[:num_results]):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source=SearchEngine.GOOGLE,
                        rank=i + 1,
                        relevance_score=1.0 - (i * 0.05),
                    ))
                return results
        except Exception as e:
            logger.error(f"Google search failed: {e}")
        return []

    async def _search_bing(self, query: str, num_results: int) -> List[SearchResult]:
        """Bing Suche - erfordert API-Key Konfiguration."""
        api_key = os.getenv("BING_API_KEY")

        if not api_key:
            logger.warning("Bing Search nicht konfiguriert: BING_API_KEY erforderlich")
            return []

        url = f"https://api.bing.microsoft.com/v7.0/search?q={quote_plus(query)}&count={num_results}"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        try:
            status, content = await self.http.get(url, headers=headers)
            if status == 200:
                data = json.loads(content)
                results = []
                for i, item in enumerate(data.get("webPages", {}).get("value", [])[:num_results]):
                    results.append(SearchResult(
                        title=item.get("name", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        source=SearchEngine.BING,
                        rank=i + 1,
                        relevance_score=1.0 - (i * 0.05),
                    ))
                return results
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
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
        """
        GitHub Repository/Code Suche.

        Nutzt die GitHub Search API für Code und Repositories.
        """
        results = []

        # GitHub API für Repository-Suche
        api_url = f"https://api.github.com/search/repositories?q={quote_plus(query)}&per_page={num_results}"

        try:
            headers = {"Accept": "application/vnd.github.v3+json"}
            status, content = await self.http.get(api_url, headers=headers)

            if status == 200:
                data = json.loads(content)

                for i, item in enumerate(data.get("items", [])[:num_results]):
                    results.append(SearchResult(
                        title=item.get("full_name", "Unknown"),
                        url=item.get("html_url", ""),
                        snippet=item.get("description", "")[:300] if item.get("description") else "",
                        source=SearchEngine.GITHUB,
                        rank=i + 1,
                        relevance_score=item.get("score", 0.0),
                        metadata={
                            "stars": item.get("stargazers_count", 0),
                            "forks": item.get("forks_count", 0),
                            "language": item.get("language"),
                            "updated_at": item.get("updated_at"),
                            "topics": item.get("topics", []),
                        }
                    ))

            elif status == 403:
                logger.warning("GitHub API rate limit erreicht")

        except json.JSONDecodeError as e:
            logger.warning(f"GitHub JSON parse error: {e}")
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")

        return results

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
        """
        ArXiv Suche mit echtem XML-Parsing.

        Nutzt die ArXiv API und parst das Atom/XML Response Format.
        """
        encoded_query = quote_plus(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"

        try:
            status, content = await self.http.get(url)
            if status == 200:
                papers = self._parse_arxiv_xml(content)
                return papers[:max_results]
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")

        return []

    def _parse_arxiv_xml(self, xml_content: str) -> List[Paper]:
        """Parst ArXiv Atom/XML Response."""
        papers = []

        # Extrahiere Entry-Blöcke
        entry_pattern = r'<entry>(.*?)</entry>'
        entries = re.findall(entry_pattern, xml_content, re.DOTALL)

        for entry in entries:
            try:
                # ID extrahieren
                id_match = re.search(r'<id>([^<]+)</id>', entry)
                arxiv_id = id_match.group(1) if id_match else ""
                # Konvertiere URL zu ID
                arxiv_id = arxiv_id.replace("http://arxiv.org/abs/", "").strip()

                # Titel extrahieren
                title_match = re.search(r'<title>([^<]+)</title>', entry)
                title = title_match.group(1).strip() if title_match else "Unknown"
                title = re.sub(r'\s+', ' ', title)  # Normalisiere Whitespace

                # Abstract extrahieren
                abstract_match = re.search(r'<summary>([^<]+)</summary>', entry)
                abstract = abstract_match.group(1).strip() if abstract_match else ""
                abstract = re.sub(r'\s+', ' ', abstract)

                # Autoren extrahieren
                author_pattern = r'<author>.*?<name>([^<]+)</name>.*?</author>'
                authors = re.findall(author_pattern, entry, re.DOTALL)

                # Kategorien extrahieren
                category_pattern = r'<category[^>]*term="([^"]+)"'
                categories = re.findall(category_pattern, entry)

                # URL
                link_match = re.search(r'<link[^>]*href="([^"]*arxiv[^"]*)"[^>]*/>', entry)
                paper_url = link_match.group(1) if link_match else f"https://arxiv.org/abs/{arxiv_id}"

                papers.append(Paper(
                    id=f"arxiv:{arxiv_id}",
                    title=title,
                    abstract=abstract[:1000],  # Begrenzen
                    authors=authors[:10],
                    url=paper_url,
                    source=SearchEngine.ARXIV,
                    categories=categories[:5],
                ))

            except Exception as e:
                logger.debug(f"Error parsing ArXiv entry: {e}")
                continue

        return papers

    async def _search_semantic_scholar(self, query: str, max_results: int) -> List[Paper]:
        """
        Semantic Scholar API Suche.

        Nutzt die öffentliche API von Semantic Scholar.
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = f"?query={quote_plus(query)}&limit={min(max_results, 100)}&fields=paperId,title,abstract,authors,url,venue,year"

        try:
            status, content = await self.http.get(url + params)
            if status == 200:
                data = json.loads(content)
                papers = []

                for item in data.get("data", []):
                    authors = [
                        a.get("name", "Unknown")
                        for a in item.get("authors", [])
                    ]

                    papers.append(Paper(
                        id=f"s2:{item.get('paperId', '')}",
                        title=item.get("title", "Unknown"),
                        abstract=item.get("abstract", "")[:1000] if item.get("abstract") else "",
                        authors=authors[:10],
                        url=item.get("url", ""),
                        source=SearchEngine.SEMANTIC_SCHOLAR,
                        categories=[item.get("venue", "")] if item.get("venue") else [],
                        year=item.get("year"),
                    ))

                return papers

        except json.JSONDecodeError as e:
            logger.warning(f"Semantic Scholar JSON parse error: {e}")
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")

        return []

    async def _search_pubmed(self, query: str, max_results: int) -> List[Paper]:
        """
        PubMed API Suche.

        Nutzt NCBI E-utilities für die Suche.
        """
        # Schritt 1: ESearch für IDs
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={quote_plus(query)}&retmax={max_results}&retmode=json"
        )

        try:
            status, content = await self.http.get(search_url)
            if status != 200:
                return []

            search_data = json.loads(content)
            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            # Schritt 2: EFetch für Details
            ids_param = ",".join(id_list)
            fetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&id={ids_param}&retmode=json"
            )

            status, content = await self.http.get(fetch_url)
            if status != 200:
                return []

            fetch_data = json.loads(content)
            result = fetch_data.get("result", {})

            papers = []
            for pmid in id_list:
                item = result.get(pmid, {})
                if not item or pmid == "uids":
                    continue

                # Autoren aus AuthorList extrahieren
                author_list = item.get("authors", [])
                authors = [a.get("name", "") for a in author_list if a.get("name")]

                papers.append(Paper(
                    id=f"pmid:{pmid}",
                    title=item.get("title", "Unknown"),
                    abstract="",  # ESummary gibt kein Abstract zurück
                    authors=authors[:10],
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    source=SearchEngine.PUBMED,
                    categories=[item.get("source", "")],
                    year=int(item.get("pubdate", "")[:4]) if item.get("pubdate") else None,
                ))

            return papers

        except json.JSONDecodeError as e:
            logger.warning(f"PubMed JSON parse error: {e}")
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")

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
        """Holt Wetterdaten von Open-Meteo (kostenlos, kein API-Key erforderlich)."""
        try:
            # Geocoding für Koordinaten
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote_plus(location)}&count=1"
            status, content = await self.http.get(geo_url)
            if status != 200:
                logger.error(f"Geocoding failed for {location}")
                return None

            geo_data = json.loads(content)
            if not geo_data.get("results"):
                logger.warning(f"Location not found: {location}")
                return None

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]

            # Wetterdaten abrufen
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            status, content = await self.http.get(weather_url)
            if status == 200:
                data = json.loads(content)
                current = data.get("current", {})
                return RealTimeData(
                    data_type="weather",
                    data={
                        "location": location,
                        "temperature": current.get("temperature_2m"),
                        "humidity": current.get("relative_humidity_2m"),
                        "wind_speed": current.get("wind_speed_10m"),
                        "weather_code": current.get("weather_code"),
                    },
                    source="open-meteo.com",
                )
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
        return None

    async def _get_stock(self, symbol: str) -> Optional[RealTimeData]:
        """Holt Börsendaten von Alpha Vantage API."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            logger.warning("Stock data nicht verfügbar: ALPHAVANTAGE_API_KEY nicht gesetzt")
            return None

        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            status, content = await self.http.get(url)
            if status == 200:
                data = json.loads(content)
                quote = data.get("Global Quote", {})
                if quote:
                    return RealTimeData(
                        data_type="stock",
                        data={
                            "symbol": quote.get("01. symbol", symbol),
                            "price": float(quote.get("05. price", 0)),
                            "change": float(quote.get("09. change", 0)),
                            "change_percent": quote.get("10. change percent", "0%"),
                            "volume": int(quote.get("06. volume", 0)),
                        },
                        source="alphavantage.co",
                    )
        except Exception as e:
            logger.error(f"Stock fetch failed: {e}")
        return None

    async def _get_crypto(self, coin: str) -> Optional[RealTimeData]:
        """Holt Kryptowährungsdaten von CoinGecko (kostenlos)."""
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true"
            status, content = await self.http.get(url)
            if status == 200:
                data = json.loads(content)
                if coin in data:
                    coin_data = data[coin]
                    return RealTimeData(
                        data_type="crypto",
                        data={
                            "coin": coin,
                            "price_usd": coin_data.get("usd"),
                            "change_24h": coin_data.get("usd_24h_change"),
                            "market_cap": coin_data.get("usd_market_cap"),
                        },
                        source="coingecko.com",
                    )
        except Exception as e:
            logger.error(f"Crypto fetch failed: {e}")
        return None

    async def _get_news(self, topic: str) -> Optional[RealTimeData]:
        """Holt Nachrichten von NewsAPI oder GNews."""
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            # Fallback: GNews API (kostenlos, limitiert)
            try:
                url = f"https://gnews.io/api/v4/search?q={quote_plus(topic)}&lang=en&max=5&apikey={os.getenv('GNEWS_API_KEY', '')}"
                if not os.getenv('GNEWS_API_KEY'):
                    logger.warning("News API nicht konfiguriert: NEWS_API_KEY oder GNEWS_API_KEY erforderlich")
                    return None
                status, content = await self.http.get(url)
                if status == 200:
                    data = json.loads(content)
                    articles = [
                        {"title": a["title"], "source": a["source"]["name"], "url": a["url"]}
                        for a in data.get("articles", [])
                    ]
                    return RealTimeData(data_type="news", data={"topic": topic, "articles": articles}, source="gnews.io")
            except Exception as e:
                logger.error(f"GNews fetch failed: {e}")
            return None

        try:
            url = f"https://newsapi.org/v2/everything?q={quote_plus(topic)}&pageSize=5&apiKey={api_key}"
            status, content = await self.http.get(url)
            if status == 200:
                data = json.loads(content)
                articles = [
                    {"title": a["title"], "source": a["source"]["name"], "url": a["url"]}
                    for a in data.get("articles", [])
                ]
                return RealTimeData(
                    data_type="news",
                    data={"topic": topic, "articles": articles},
                    source="newsapi.org",
                )
        except Exception as e:
            logger.error(f"News fetch failed: {e}")
        return None

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
