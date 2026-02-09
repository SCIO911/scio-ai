"""
Tests for SCIO Web Modules

Tests for:
- WebClient and AsyncWebClient
- WebScraper and content extraction
- WebSearch and AcademicSearch
- FeedParser
- APIClient and GraphQLClient
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

from scio.web.client import (
    WebClient,
    AsyncWebClient,
    HttpResponse,
    HttpError,
    HttpMethod,
    ContentType,
    RequestConfig,
)
from scio.web.scraper import (
    WebScraper,
    ContentExtractor,
    LinkExtractor,
    TableExtractor,
    ExtractedContent,
    HTMLParser,
)
from scio.web.search import (
    SearchEngine,
    SearchResult,
    SearchResponse,
    WebSearch,
    AcademicSearch,
    CodeSearch,
    SearchType,
)
from scio.web.api import (
    APIClient,
    RESTClient,
    GraphQLClient,
    WebhookServer,
    APIConfig,
    GraphQLError,
)
from scio.web.feeds import (
    FeedParser,
    FeedEntry,
    RSSFeed,
    AtomFeed,
    FeedType,
)


# =============================================================================
# WEB CLIENT TESTS
# =============================================================================

class TestHttpMethod:
    """Tests for HttpMethod enum."""

    def test_methods_exist(self):
        """Test HTTP methods exist."""
        assert HttpMethod.GET.value == "GET"
        assert HttpMethod.POST.value == "POST"
        assert HttpMethod.PUT.value == "PUT"
        assert HttpMethod.DELETE.value == "DELETE"


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_types_exist(self):
        """Test content types exist."""
        assert ContentType.JSON.value == "application/json"
        assert ContentType.HTML.value == "text/html"


class TestHttpResponse:
    """Tests for HttpResponse."""

    def test_create_response(self):
        """Test creating response."""
        response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"key": "value"}',
            url="https://example.com",
            method="GET",
            elapsed_ms=100,
        )
        assert response.ok is True
        assert response.status_code == 200

    def test_response_text(self):
        """Test text property."""
        response = HttpResponse(
            status_code=200,
            headers={},
            body=b"Hello World",
            url="",
            method="GET",
            elapsed_ms=0,
        )
        assert response.text == "Hello World"

    def test_response_json(self):
        """Test JSON parsing."""
        response = HttpResponse(
            status_code=200,
            headers={},
            body=b'{"key": "value"}',
            url="",
            method="GET",
            elapsed_ms=0,
        )
        data = response.json()
        assert data == {"key": "value"}

    def test_ok_status_codes(self):
        """Test ok property for various status codes."""
        for code in [200, 201, 204, 299]:
            response = HttpResponse(code, {}, b"", "", "GET", 0)
            assert response.ok is True

        for code in [400, 404, 500, 503]:
            response = HttpResponse(code, {}, b"", "", "GET", 0)
            assert response.ok is False

    def test_raise_for_status(self):
        """Test raise_for_status."""
        response = HttpResponse(404, {}, b"Not Found", "", "GET", 0)
        with pytest.raises(HttpError):
            response.raise_for_status()


class TestRequestConfig:
    """Tests for RequestConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RequestConfig()
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.verify_ssl is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RequestConfig(
            timeout=60.0,
            max_retries=5,
            verify_ssl=False,
        )
        assert config.timeout == 60.0
        assert config.max_retries == 5


class TestWebClient:
    """Tests for WebClient."""

    def test_create_client(self):
        """Test creating client."""
        client = WebClient()
        assert client is not None

    def test_set_cookie(self):
        """Test setting cookies."""
        client = WebClient()
        client.set_cookie("session", "abc123")
        assert client._cookies["session"] == "abc123"

    def test_clear_cookies(self):
        """Test clearing cookies."""
        client = WebClient()
        client.set_cookie("session", "abc123")
        client.clear_cookies()
        assert len(client._cookies) == 0


# =============================================================================
# WEB SCRAPER TESTS
# =============================================================================

class TestHTMLParser:
    """Tests for HTMLParser."""

    def test_extract_text(self):
        """Test text extraction."""
        parser = HTMLParser()
        parser.feed("<html><body><p>Hello World</p></body></html>")
        assert "Hello World" in parser.get_text()

    def test_extract_links(self):
        """Test link extraction."""
        parser = HTMLParser()
        parser.feed('<a href="https://example.com">Link</a>')
        assert "https://example.com" in parser.links

    def test_extract_images(self):
        """Test image extraction."""
        parser = HTMLParser()
        parser.feed('<img src="image.jpg" alt="test">')
        assert "image.jpg" in parser.images

    def test_extract_title(self):
        """Test title extraction."""
        parser = HTMLParser()
        parser.feed("<html><head><title>My Title</title></head></html>")
        assert parser.title == "My Title"

    def test_extract_meta(self):
        """Test meta tag extraction."""
        parser = HTMLParser()
        parser.feed('<meta name="description" content="Test description">')
        assert parser.meta.get("description") == "Test description"

    def test_extract_headings(self):
        """Test heading extraction."""
        parser = HTMLParser()
        parser.feed("<h1>Main Title</h1><h2>Subtitle</h2>")
        assert (1, "Main Title") in parser.headings
        assert (2, "Subtitle") in parser.headings

    def test_extract_tables(self):
        """Test table extraction."""
        parser = HTMLParser()
        parser.feed("""
            <table>
                <tr><th>Name</th><th>Value</th></tr>
                <tr><td>A</td><td>1</td></tr>
            </table>
        """)
        assert len(parser.tables) == 1
        assert len(parser.tables[0]) == 2  # 2 rows

    def test_ignore_scripts(self):
        """Test that script content is ignored."""
        parser = HTMLParser()
        parser.feed("<script>var x = 'secret';</script><p>Visible</p>")
        text = parser.get_text()
        assert "secret" not in text
        assert "Visible" in text

    def test_ignore_styles(self):
        """Test that style content is ignored."""
        parser = HTMLParser()
        parser.feed("<style>.class { color: red; }</style><p>Visible</p>")
        text = parser.get_text()
        assert "color" not in text
        assert "Visible" in text


class TestExtractedContent:
    """Tests for ExtractedContent dataclass."""

    def test_create_content(self):
        """Test creating content."""
        content = ExtractedContent(
            url="https://example.com",
            title="Test",
            text="Content here",
        )
        assert content.url == "https://example.com"
        assert content.title == "Test"


class TestWebScraper:
    """Tests for WebScraper."""

    def test_parse_html(self):
        """Test HTML parsing."""
        scraper = WebScraper()
        content = scraper.parse_html(
            "<html><head><title>Test</title></head><body><p>Hello</p></body></html>",
            "https://example.com"
        )
        assert content.title == "Test"
        assert "Hello" in content.text

    def test_resolve_relative_links(self):
        """Test relative link resolution."""
        scraper = WebScraper()
        content = scraper.parse_html(
            '<a href="/page">Link</a>',
            "https://example.com"
        )
        assert "https://example.com/page" in content.links


class TestContentExtractor:
    """Tests for ContentExtractor."""

    def test_extract_main_content(self):
        """Test main content extraction."""
        extractor = ContentExtractor()
        html = """
            <nav>Navigation here</nav>
            <article>
                <p>This is a very long paragraph with lots of content that should be
                extracted as the main content because it has more than 100 characters
                and contains the actual article text that users want to read.</p>
            </article>
            <footer>Footer here</footer>
        """
        main = extractor.extract_main_content(html)
        assert len(main) > 0

    def test_extract_structured_data(self):
        """Test structured data extraction."""
        extractor = ContentExtractor()
        html = '''
            <script type="application/ld+json">
            {"@type": "Article", "headline": "Test"}
            </script>
        '''
        data = extractor.extract_structured_data(html)
        assert len(data) == 1
        assert data[0]["@type"] == "Article"


class TestTableExtractor:
    """Tests for TableExtractor."""

    def test_tables_to_dicts(self):
        """Test converting tables to dictionaries."""
        extractor = TableExtractor()
        tables = [[["Name", "Value"], ["A", "1"], ["B", "2"]]]
        dicts = extractor.tables_to_dicts(tables)

        assert len(dicts) == 1
        assert len(dicts[0]) == 2
        assert dicts[0][0]["Name"] == "A"
        assert dicts[0][0]["Value"] == "1"

    def test_find_table_by_header(self):
        """Test finding table by header."""
        extractor = TableExtractor()
        tables = [
            [["ID", "Name"], ["1", "A"]],
            [["Price", "Quantity"], ["10", "5"]],
        ]
        table = extractor.find_table_by_header(tables, "Price")
        assert table is not None
        assert "Price" in table[0]


# =============================================================================
# SEARCH TESTS
# =============================================================================

class TestSearchType:
    """Tests for SearchType enum."""

    def test_types_exist(self):
        """Test search types exist."""
        assert SearchType.WEB.value == "web"
        assert SearchType.NEWS.value == "news"
        assert SearchType.ACADEMIC.value == "academic"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_result(self):
        """Test creating result."""
        result = SearchResult(
            title="Test Result",
            url="https://example.com",
            snippet="This is a test",
            source="Test",
        )
        assert result.title == "Test Result"
        assert result.rank == 0

    def test_to_dict(self):
        """Test converting to dict."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet",
            source="Source",
        )
        d = result.to_dict()
        assert d["title"] == "Test"
        assert d["url"] == "https://example.com"


class TestSearchResponse:
    """Tests for SearchResponse."""

    def test_create_response(self):
        """Test creating response."""
        response = SearchResponse(
            query="test query",
            results=[],
            total_results=0,
        )
        assert response.query == "test query"
        assert response.page == 1


class TestCodeSearch:
    """Tests for CodeSearch."""

    def test_create_search(self):
        """Test creating code search."""
        search = CodeSearch()
        assert search is not None


# =============================================================================
# API CLIENT TESTS
# =============================================================================

class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = APIConfig(base_url="https://api.example.com")
        assert config.timeout == 30.0
        assert config.auth_type == "bearer"

    def test_auth_config(self):
        """Test authentication configuration."""
        config = APIConfig(
            base_url="https://api.example.com",
            api_key="test-key",
            auth_type="api_key",
        )
        assert config.api_key == "test-key"


class TestAPIClient:
    """Tests for APIClient."""

    def test_auth_headers_bearer(self):
        """Test Bearer authentication headers."""
        config = APIConfig(
            base_url="https://api.example.com",
            api_key="test-token",
            auth_type="bearer",
        )
        client = APIClient(config)
        headers = client._get_auth_headers()
        assert headers["Authorization"] == "Bearer test-token"

    def test_auth_headers_api_key(self):
        """Test API key authentication headers."""
        config = APIConfig(
            base_url="https://api.example.com",
            api_key="test-key",
            auth_type="api_key",
        )
        client = APIClient(config)
        headers = client._get_auth_headers()
        assert headers["X-API-Key"] == "test-key"

    def test_hmac_signature(self):
        """Test HMAC signature generation."""
        config = APIConfig(
            base_url="https://api.example.com",
            api_key="key",
            api_secret="secret",
            auth_type="hmac",
        )
        client = APIClient(config)
        signature = client._create_hmac_signature("POST", "/test", {"data": "value"})
        assert len(signature) == 64  # SHA256 hex digest

    def test_clear_cache(self):
        """Test cache clearing."""
        config = APIConfig(base_url="https://api.example.com")
        client = APIClient(config)
        client._cache["key"] = (datetime.now(), "value")
        client.clear_cache()
        assert len(client._cache) == 0


class TestGraphQLClient:
    """Tests for GraphQLClient."""

    def test_create_client(self):
        """Test creating client."""
        client = GraphQLClient("https://api.example.com/graphql")
        assert client is not None


class TestGraphQLError:
    """Tests for GraphQLError."""

    def test_error_with_errors(self):
        """Test error with error list."""
        error = GraphQLError("Test error", errors=[{"message": "detail"}])
        assert str(error) == "Test error"
        assert len(error.errors) == 1


# =============================================================================
# FEED PARSER TESTS
# =============================================================================

class TestFeedType:
    """Tests for FeedType enum."""

    def test_types_exist(self):
        """Test feed types exist."""
        assert FeedType.RSS.value == "rss"
        assert FeedType.ATOM.value == "atom"


class TestFeedEntry:
    """Tests for FeedEntry dataclass."""

    def test_create_entry(self):
        """Test creating entry."""
        entry = FeedEntry(
            id="entry-1",
            title="Test Entry",
            link="https://example.com/entry",
        )
        assert entry.id == "entry-1"
        assert entry.title == "Test Entry"

    def test_to_dict(self):
        """Test converting to dict."""
        entry = FeedEntry(
            id="entry-1",
            title="Test",
            link="https://example.com",
            published=datetime(2024, 1, 1),
        )
        d = entry.to_dict()
        assert d["id"] == "entry-1"
        assert d["published"] == "2024-01-01T00:00:00"


class TestFeedParser:
    """Tests for FeedParser."""

    def test_detect_rss(self):
        """Test RSS detection."""
        parser = FeedParser()
        feed_type = parser._detect_feed_type('<rss version="2.0"><channel></channel></rss>')
        assert feed_type == FeedType.RSS2

    def test_detect_atom(self):
        """Test Atom detection."""
        parser = FeedParser()
        feed_type = parser._detect_feed_type(
            '<feed xmlns="http://www.w3.org/2005/Atom"><title>Test</title></feed>'
        )
        assert feed_type == FeedType.ATOM

    def test_parse_rss(self):
        """Test RSS parsing."""
        parser = FeedParser()
        rss = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <link>https://example.com</link>
                <description>A test feed</description>
                <item>
                    <title>Entry 1</title>
                    <link>https://example.com/1</link>
                    <description>First entry</description>
                </item>
            </channel>
        </rss>'''

        feed = parser.parse(rss)
        assert isinstance(feed, RSSFeed)
        assert feed.title == "Test Feed"
        assert len(feed.entries) == 1
        assert feed.entries[0].title == "Entry 1"

    def test_parse_atom(self):
        """Test Atom parsing."""
        parser = FeedParser()
        atom = '''<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <title>Test Feed</title>
            <link href="https://example.com" rel="alternate"/>
            <subtitle>A test feed</subtitle>
            <entry>
                <title>Entry 1</title>
                <link href="https://example.com/1" rel="alternate"/>
                <id>entry-1</id>
                <summary>First entry</summary>
            </entry>
        </feed>'''

        feed = parser.parse(atom)
        assert isinstance(feed, AtomFeed)
        assert feed.title == "Test Feed"
        assert len(feed.entries) == 1
        assert feed.entries[0].title == "Entry 1"

    def test_parse_date_rfc822(self):
        """Test RFC 822 date parsing."""
        parser = FeedParser()
        date = parser._parse_date("Sat, 01 Jan 2024 12:00:00 +0000")
        assert date is not None
        assert date.year == 2024
        assert date.month == 1
        assert date.day == 1

    def test_parse_date_iso8601(self):
        """Test ISO 8601 date parsing."""
        parser = FeedParser()
        date = parser._parse_date("2024-01-01T12:00:00Z")
        assert date is not None
        assert date.year == 2024


class TestRSSFeed:
    """Tests for RSSFeed dataclass."""

    def test_create_feed(self):
        """Test creating feed."""
        feed = RSSFeed(
            title="Test Feed",
            link="https://example.com",
            description="A test",
            feed_type=FeedType.RSS2,
        )
        assert feed.title == "Test Feed"

    def test_to_dict(self):
        """Test converting to dict."""
        feed = RSSFeed(
            title="Test",
            link="https://example.com",
            description="A test",
            feed_type=FeedType.RSS2,
            entries=[
                FeedEntry(id="1", title="Entry", link="https://example.com/1")
            ],
        )
        d = feed.to_dict()
        assert d["title"] == "Test"
        assert len(d["entries"]) == 1


class TestAtomFeed:
    """Tests for AtomFeed dataclass."""

    def test_create_feed(self):
        """Test creating feed."""
        feed = AtomFeed(
            title="Test Feed",
            link="https://example.com",
        )
        assert feed.title == "Test Feed"
        assert feed.feed_type == FeedType.ATOM


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestWebModuleIntegration:
    """Integration tests for web modules."""

    def test_scraper_with_parser(self):
        """Test scraper creates correct content."""
        scraper = WebScraper()
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Welcome</h1>
                <p>This is a test paragraph.</p>
                <a href="/link1">Link 1</a>
                <a href="/link2">Link 2</a>
                <table>
                    <tr><th>A</th><th>B</th></tr>
                    <tr><td>1</td><td>2</td></tr>
                </table>
            </body>
        </html>
        """
        content = scraper.parse_html(html, "https://example.com")

        assert content.title == "Test Page"
        assert "Welcome" in content.text
        assert "test paragraph" in content.text
        assert len(content.links) == 2
        assert len(content.tables) == 1
        assert (1, "Welcome") in content.headings

    def test_search_result_lifecycle(self):
        """Test search result creation and serialization."""
        result = SearchResult(
            title="Test Result",
            url="https://example.com/result",
            snippet="This is a test result snippet",
            source="TestEngine",
            search_type=SearchType.WEB,
            rank=1,
            score=0.95,
            metadata={"extra": "data"},
        )

        d = result.to_dict()

        assert d["title"] == "Test Result"
        assert d["search_type"] == "web"
        assert d["metadata"]["extra"] == "data"

    def test_feed_entry_lifecycle(self):
        """Test feed entry creation and serialization."""
        entry = FeedEntry(
            id="unique-id",
            title="Test Article",
            link="https://example.com/article",
            summary="Short summary",
            content="Full content here...",
            author="Test Author",
            published=datetime(2024, 6, 15, 12, 0, 0),
            categories=["tech", "news"],
        )

        d = entry.to_dict()

        assert d["id"] == "unique-id"
        assert d["author"] == "Test Author"
        assert "tech" in d["categories"]
        assert d["published"] == "2024-06-15T12:00:00"

    def test_module_exports(self):
        """Test that all expected classes are exported."""
        from scio.web import (
            WebClient,
            AsyncWebClient,
            WebScraper,
            ContentExtractor,
            LinkExtractor,
            TableExtractor,
            SearchEngine,
            SearchResult,
            WebSearch,
            AcademicSearch,
            APIClient,
            RESTClient,
            GraphQLClient,
            WebhookServer,
            FeedParser,
            FeedEntry,
            RSSFeed,
            AtomFeed,
        )

        # All imports should work
        assert WebClient is not None
        assert FeedParser is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
