"""
SCIO Web Search

Such-Funktionalitäten für verschiedene Quellen.
"""

import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc
from scio.web.client import WebClient
from scio.web.scraper import WebScraper

logger = get_logger(__name__)


class SearchType(str, Enum):
    """Suchtypen."""
    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    ACADEMIC = "academic"
    CODE = "code"


@dataclass
class SearchResult:
    """Ein Suchergebnis."""

    title: str
    url: str
    snippet: str
    source: str
    search_type: SearchType = SearchType.WEB
    rank: int = 0
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "search_type": self.search_type.value,
            "rank": self.rank,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class SearchResponse:
    """Antwort einer Suchanfrage."""

    query: str
    results: list[SearchResult]
    total_results: int = 0
    search_time_ms: float = 0
    page: int = 1
    has_more: bool = False
    source: str = ""


class SearchEngine:
    """
    Basis-Suchmaschine.

    Abstrahiert verschiedene Suchquellen.
    """

    def __init__(self, client: Optional[WebClient] = None):
        self.client = client or WebClient()
        self._scraper = WebScraper(self.client)
        logger.info("SearchEngine initialized")

    def search(
        self,
        query: str,
        search_type: SearchType = SearchType.WEB,
        num_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """
        Führt eine Suche durch.

        Args:
            query: Suchanfrage
            search_type: Art der Suche
            num_results: Anzahl der Ergebnisse

        Returns:
            SearchResponse mit Ergebnissen
        """
        raise NotImplementedError("Subclass must implement search()")


class WebSearch(SearchEngine):
    """
    Web-Suche über verschiedene Suchmaschinen.

    Features:
    - DuckDuckGo HTML-Suche
    - Bing-Suche
    - Custom Search Engines
    """

    def __init__(self, client: Optional[WebClient] = None):
        super().__init__(client)

    def search(
        self,
        query: str,
        search_type: SearchType = SearchType.WEB,
        num_results: int = 10,
        region: str = "wt-wt",
        safe_search: bool = True,
        **kwargs: Any,
    ) -> SearchResponse:
        """
        Führt eine Web-Suche durch.

        Args:
            query: Suchanfrage
            search_type: Art der Suche (web, news, images)
            num_results: Anzahl der Ergebnisse
            region: Regions-Code (z.B. "de-de", "wt-wt")
            safe_search: Safe Search aktivieren

        Returns:
            SearchResponse mit Ergebnissen
        """
        start_time = datetime.now()

        # DuckDuckGo HTML-Suche
        results = self._search_duckduckgo(query, num_results, region)

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time_ms=elapsed,
            source="DuckDuckGo",
            has_more=len(results) >= num_results,
        )

    def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        region: str,
    ) -> list[SearchResult]:
        """DuckDuckGo HTML-Suche."""
        results = []

        # DuckDuckGo HTML-Version
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query, "kl": region}

        try:
            response = self.client.post(url, data=params)
            if not response.ok:
                logger.warning("DuckDuckGo search failed", status=response.status_code)
                return results

            html = response.text

            # Parse Ergebnisse
            result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>'

            titles_urls = re.findall(result_pattern, html, re.DOTALL)
            snippets = re.findall(snippet_pattern, html, re.DOTALL)

            for i, (url, title) in enumerate(titles_urls[:num_results]):
                # URL dekodieren (DuckDuckGo nutzt Redirect-URLs)
                if "uddg=" in url:
                    match = re.search(r'uddg=([^&]+)', url)
                    if match:
                        url = urllib.parse.unquote(match.group(1))

                # HTML-Tags entfernen
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = ""
                if i < len(snippets):
                    snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()

                if title and url.startswith("http"):
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="DuckDuckGo",
                        search_type=SearchType.WEB,
                        rank=i + 1,
                        score=1.0 / (i + 1),  # Einfaches Ranking-Score
                    ))

        except Exception as e:
            logger.error("Search error", error=str(e))

        return results

    def search_news(self, query: str, num_results: int = 10) -> SearchResponse:
        """Sucht nach Nachrichten."""
        return self.search(query, SearchType.NEWS, num_results)

    def search_images(self, query: str, num_results: int = 10) -> SearchResponse:
        """Sucht nach Bildern."""
        return self.search(query, SearchType.IMAGES, num_results)

    def search_and_fetch(
        self,
        query: str,
        num_results: int = 5,
        fetch_content: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Sucht und lädt den Content der Ergebnisse.

        Returns:
            Liste von Dicts mit Suchergebnis und Content
        """
        search_response = self.search(query, num_results=num_results)
        results = []

        for result in search_response.results:
            item = result.to_dict()

            if fetch_content:
                try:
                    content = self._scraper.scrape(result.url)
                    item["content"] = content.text[:5000]  # Erste 5000 Zeichen
                    item["full_content"] = content.text
                except Exception as e:
                    item["content"] = f"Error fetching: {e}"
                    item["full_content"] = ""

            results.append(item)

        return results


class AcademicSearch(SearchEngine):
    """
    Akademische Suche.

    Features:
    - arXiv-Suche
    - Semantic Scholar
    - PubMed
    """

    def __init__(self, client: Optional[WebClient] = None):
        super().__init__(client)

    def search(
        self,
        query: str,
        search_type: SearchType = SearchType.ACADEMIC,
        num_results: int = 10,
        source: str = "arxiv",
        **kwargs: Any,
    ) -> SearchResponse:
        """
        Führt eine akademische Suche durch.

        Args:
            query: Suchanfrage
            num_results: Anzahl der Ergebnisse
            source: Quelle (arxiv, semanticscholar, pubmed)
        """
        start_time = datetime.now()

        if source == "arxiv":
            results = self._search_arxiv(query, num_results)
        elif source == "semanticscholar":
            results = self._search_semantic_scholar(query, num_results)
        else:
            results = self._search_arxiv(query, num_results)

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time_ms=elapsed,
            source=source,
            has_more=len(results) >= num_results,
        )

    def _search_arxiv(self, query: str, num_results: int) -> list[SearchResult]:
        """arXiv API-Suche."""
        results = []

        # arXiv API
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": num_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = self.client.get(url, params=params)
            if not response.ok:
                return results

            # Einfaches XML-Parsing
            xml = response.text

            # Entries extrahieren
            entries = re.findall(r'<entry>(.*?)</entry>', xml, re.DOTALL)

            for i, entry in enumerate(entries):
                # Titel
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                title = re.sub(r'\s+', ' ', title)

                # Abstract
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                summary = summary_match.group(1).strip() if summary_match else ""
                summary = re.sub(r'\s+', ' ', summary)[:500]

                # URL
                id_match = re.search(r'<id>(.*?)</id>', entry)
                arxiv_url = id_match.group(1) if id_match else ""

                # Autoren
                authors = re.findall(r'<name>(.*?)</name>', entry)

                # Kategorien
                categories = re.findall(r'<category[^>]*term="([^"]*)"', entry)

                # Datum
                published_match = re.search(r'<published>(.*?)</published>', entry)
                published = published_match.group(1)[:10] if published_match else ""

                if title and arxiv_url:
                    results.append(SearchResult(
                        title=title,
                        url=arxiv_url,
                        snippet=summary,
                        source="arXiv",
                        search_type=SearchType.ACADEMIC,
                        rank=i + 1,
                        score=1.0 / (i + 1),
                        metadata={
                            "authors": authors[:5],
                            "categories": categories,
                            "published": published,
                        },
                    ))

        except Exception as e:
            logger.error("arXiv search error", error=str(e))

        return results

    def _search_semantic_scholar(self, query: str, num_results: int) -> list[SearchResult]:
        """Semantic Scholar API-Suche."""
        results = []

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": num_results,
            "fields": "title,abstract,url,authors,year,citationCount",
        }

        try:
            response = self.client.get(url, params=params)
            if not response.ok:
                return results

            data = response.json()
            papers = data.get("data", [])

            for i, paper in enumerate(papers):
                authors = [a.get("name", "") for a in paper.get("authors", [])]

                results.append(SearchResult(
                    title=paper.get("title", ""),
                    url=paper.get("url", ""),
                    snippet=paper.get("abstract", "")[:500] if paper.get("abstract") else "",
                    source="Semantic Scholar",
                    search_type=SearchType.ACADEMIC,
                    rank=i + 1,
                    score=1.0 / (i + 1),
                    metadata={
                        "authors": authors[:5],
                        "year": paper.get("year"),
                        "citations": paper.get("citationCount", 0),
                    },
                ))

        except Exception as e:
            logger.error("Semantic Scholar search error", error=str(e))

        return results

    def search_arxiv(self, query: str, num_results: int = 10) -> SearchResponse:
        """Direkte arXiv-Suche."""
        return self.search(query, num_results=num_results, source="arxiv")

    def search_papers(self, query: str, num_results: int = 10) -> SearchResponse:
        """Suche nach wissenschaftlichen Papern."""
        return self.search(query, num_results=num_results, source="semanticscholar")

    def get_paper_details(self, arxiv_id: str) -> Optional[dict[str, Any]]:
        """
        Holt Details zu einem arXiv-Paper.

        Args:
            arxiv_id: arXiv-ID (z.B. "2301.07041")

        Returns:
            Paper-Details oder None
        """
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

        try:
            response = self.client.get(url)
            if not response.ok:
                return None

            xml = response.text
            entry_match = re.search(r'<entry>(.*?)</entry>', xml, re.DOTALL)
            if not entry_match:
                return None

            entry = entry_match.group(1)

            # Alle Felder extrahieren
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            published_match = re.search(r'<published>(.*?)</published>', entry)
            updated_match = re.search(r'<updated>(.*?)</updated>', entry)

            authors = re.findall(r'<name>(.*?)</name>', entry)
            categories = re.findall(r'<category[^>]*term="([^"]*)"', entry)
            links = re.findall(r'<link[^>]*href="([^"]*)"[^>]*/>', entry)

            pdf_link = None
            for link in links:
                if "pdf" in link:
                    pdf_link = link
                    break

            return {
                "id": arxiv_id,
                "title": re.sub(r'\s+', ' ', title_match.group(1).strip()) if title_match else "",
                "abstract": re.sub(r'\s+', ' ', summary_match.group(1).strip()) if summary_match else "",
                "authors": authors,
                "categories": categories,
                "published": published_match.group(1) if published_match else "",
                "updated": updated_match.group(1) if updated_match else "",
                "pdf_url": pdf_link,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }

        except Exception as e:
            logger.error("Paper details error", error=str(e))
            return None


class CodeSearch:
    """
    Code-Suche auf GitHub etc.
    """

    def __init__(self, client: Optional[WebClient] = None):
        self.client = client or WebClient()

    def search_github(
        self,
        query: str,
        language: Optional[str] = None,
        num_results: int = 10,
    ) -> list[SearchResult]:
        """
        Sucht Code auf GitHub.

        Note: Nutzt die GitHub Search API (rate limited ohne Token).
        """
        results = []

        url = "https://api.github.com/search/code"
        q = query
        if language:
            q += f" language:{language}"

        params = {"q": q, "per_page": num_results}
        headers = {"Accept": "application/vnd.github.v3+json"}

        try:
            response = self.client.get(url, params=params, headers=headers)
            if not response.ok:
                logger.warning("GitHub search failed", status=response.status_code)
                return results

            data = response.json()
            items = data.get("items", [])

            for i, item in enumerate(items):
                repo = item.get("repository", {})
                results.append(SearchResult(
                    title=item.get("name", ""),
                    url=item.get("html_url", ""),
                    snippet=item.get("path", ""),
                    source="GitHub",
                    search_type=SearchType.CODE,
                    rank=i + 1,
                    score=item.get("score", 0),
                    metadata={
                        "repository": repo.get("full_name", ""),
                        "language": language or "unknown",
                        "path": item.get("path", ""),
                    },
                ))

        except Exception as e:
            logger.error("GitHub search error", error=str(e))

        return results

    def search_repos(
        self,
        query: str,
        language: Optional[str] = None,
        num_results: int = 10,
    ) -> list[SearchResult]:
        """Sucht GitHub-Repositories."""
        results = []

        url = "https://api.github.com/search/repositories"
        q = query
        if language:
            q += f" language:{language}"

        params = {"q": q, "per_page": num_results, "sort": "stars"}
        headers = {"Accept": "application/vnd.github.v3+json"}

        try:
            response = self.client.get(url, params=params, headers=headers)
            if not response.ok:
                return results

            data = response.json()
            items = data.get("items", [])

            for i, item in enumerate(items):
                results.append(SearchResult(
                    title=item.get("full_name", ""),
                    url=item.get("html_url", ""),
                    snippet=item.get("description", "") or "",
                    source="GitHub",
                    search_type=SearchType.CODE,
                    rank=i + 1,
                    score=item.get("stargazers_count", 0),
                    metadata={
                        "stars": item.get("stargazers_count", 0),
                        "forks": item.get("forks_count", 0),
                        "language": item.get("language"),
                        "topics": item.get("topics", []),
                        "updated": item.get("updated_at"),
                    },
                ))

        except Exception as e:
            logger.error("GitHub repos search error", error=str(e))

        return results
