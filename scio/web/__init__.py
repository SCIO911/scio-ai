"""
SCIO Web Module

Umfassende Internet-Fähigkeiten für Datenbeschaffung,
Web-Scraping, API-Zugriff und mehr.
"""

from scio.web.client import (
    WebClient,
    AsyncWebClient,
    HttpResponse,
    HttpError,
)
from scio.web.scraper import (
    WebScraper,
    ContentExtractor,
    LinkExtractor,
    TableExtractor,
)
from scio.web.search import (
    SearchEngine,
    SearchResult,
    WebSearch,
    AcademicSearch,
)
from scio.web.api import (
    APIClient,
    RESTClient,
    GraphQLClient,
    WebhookServer,
)
from scio.web.feeds import (
    FeedParser,
    FeedEntry,
    RSSFeed,
    AtomFeed,
)

__all__ = [
    # Client
    "WebClient",
    "AsyncWebClient",
    "HttpResponse",
    "HttpError",
    # Scraper
    "WebScraper",
    "ContentExtractor",
    "LinkExtractor",
    "TableExtractor",
    # Search
    "SearchEngine",
    "SearchResult",
    "WebSearch",
    "AcademicSearch",
    # API
    "APIClient",
    "RESTClient",
    "GraphQLClient",
    "WebhookServer",
    # Feeds
    "FeedParser",
    "FeedEntry",
    "RSSFeed",
    "AtomFeed",
]
