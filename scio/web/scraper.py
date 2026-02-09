"""
SCIO Web Scraper

Web-Scraping und Content-Extraktion.
"""

import html.parser
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from urllib.parse import urljoin, urlparse

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc
from scio.web.client import WebClient, HttpResponse

logger = get_logger(__name__)


@dataclass
class ExtractedContent:
    """Extrahierter Web-Content."""

    url: str
    title: Optional[str] = None
    text: str = ""
    html: str = ""
    links: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    headings: list[tuple[int, str]] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=now_utc)


class HTMLParser(html.parser.HTMLParser):
    """Einfacher HTML-Parser für Content-Extraktion."""

    def __init__(self):
        super().__init__()
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self.images: list[str] = []
        self.meta: dict[str, str] = {}
        self.headings: list[tuple[int, str]] = []
        self.title: Optional[str] = None
        self.tables: list[list[list[str]]] = []

        self._current_tag: Optional[str] = None
        self._in_script = False
        self._in_style = False
        self._in_title = False
        self._current_heading: Optional[int] = None
        self._heading_text: list[str] = []

        # Table parsing state
        self._in_table = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._in_cell = False
        self._cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self._current_tag = tag
        attrs_dict = {k: v for k, v in attrs if v}

        if tag == "script":
            self._in_script = True
        elif tag == "style":
            self._in_style = True
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            href = attrs_dict.get("href")
            if href:
                self.links.append(href)
        elif tag == "img":
            src = attrs_dict.get("src")
            if src:
                self.images.append(src)
        elif tag == "meta":
            name = attrs_dict.get("name", attrs_dict.get("property", ""))
            content = attrs_dict.get("content", "")
            if name and content:
                self.meta[name] = content
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._current_heading = int(tag[1])
            self._heading_text = []
        elif tag == "table":
            self._in_table = True
            self._current_table = []
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False
        elif tag == "style":
            self._in_style = False
        elif tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._current_heading:
            heading_text = " ".join(self._heading_text).strip()
            if heading_text:
                self.headings.append((self._current_heading, heading_text))
            self._current_heading = None
        elif tag == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False
            self._current_table = []
        elif tag == "tr" and self._current_row:
            self._current_table.append(self._current_row)
            self._current_row = []
        elif tag in ("td", "th") and self._in_cell:
            cell_content = " ".join(self._cell_text).strip()
            self._current_row.append(cell_content)
            self._in_cell = False
            self._cell_text = []
        elif tag in ("p", "div", "br", "li"):
            self.text_parts.append("\n")

        self._current_tag = None

    def handle_data(self, data: str) -> None:
        if self._in_script or self._in_style:
            return

        text = data.strip()
        if not text:
            return

        if self._in_title:
            self.title = text
        if self._current_heading is not None:
            self._heading_text.append(text)
        if self._in_cell:
            self._cell_text.append(text)

        self.text_parts.append(text)

    def get_text(self) -> str:
        """Gibt den extrahierten Text zurück."""
        text = " ".join(self.text_parts)
        # Normalisiere Whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()


class WebScraper:
    """
    Web-Scraper für Content-Extraktion.

    Features:
    - HTML-Parsing
    - Text-Extraktion
    - Link-Extraktion
    - Meta-Daten
    - Tabellen-Extraktion
    """

    def __init__(self, client: Optional[WebClient] = None):
        self.client = client or WebClient()
        logger.info("WebScraper initialized")

    def scrape(self, url: str, **kwargs: Any) -> ExtractedContent:
        """
        Scraped eine Webseite.

        Args:
            url: Die zu scrapende URL
            **kwargs: Zusätzliche Parameter für den Request

        Returns:
            ExtractedContent mit allen extrahierten Daten
        """
        response = self.client.get(url, **kwargs)
        response.raise_for_status()

        return self.parse_html(response.text, url)

    def parse_html(self, html: str, base_url: str = "") -> ExtractedContent:
        """
        Parst HTML und extrahiert Content.

        Args:
            html: HTML-String
            base_url: Basis-URL für relative Links

        Returns:
            ExtractedContent
        """
        parser = HTMLParser()
        try:
            parser.feed(html)
        except Exception as e:
            logger.warning("HTML parsing error", error=str(e))

        # Relative URLs auflösen
        links = []
        for link in parser.links:
            if base_url:
                links.append(urljoin(base_url, link))
            else:
                links.append(link)

        images = []
        for img in parser.images:
            if base_url:
                images.append(urljoin(base_url, img))
            else:
                images.append(img)

        return ExtractedContent(
            url=base_url,
            title=parser.title,
            text=parser.get_text(),
            html=html,
            links=links,
            images=images,
            meta=parser.meta,
            headings=parser.headings,
            tables=parser.tables,
        )

    def scrape_multiple(
        self,
        urls: list[str],
        **kwargs: Any,
    ) -> list[ExtractedContent]:
        """Scraped mehrere URLs."""
        results = []
        for url in urls:
            try:
                content = self.scrape(url, **kwargs)
                results.append(content)
            except Exception as e:
                logger.error("Scrape failed", url=url, error=str(e))
                results.append(ExtractedContent(url=url, text=f"Error: {e}"))
        return results


class ContentExtractor:
    """
    Fortgeschrittene Content-Extraktion.

    Features:
    - Hauptinhalt-Erkennung
    - Boilerplate-Entfernung
    - Strukturierte Daten
    """

    def __init__(self):
        self._scraper = WebScraper()

    def extract_main_content(self, html: str, url: str = "") -> str:
        """
        Extrahiert den Hauptinhalt einer Seite.

        Entfernt Navigation, Footer, Sidebar etc.
        """
        content = self._scraper.parse_html(html, url)

        # Einfache Heuristik: Längster zusammenhängender Textblock
        text = content.text
        paragraphs = text.split("\n\n")

        # Filtere kurze Absätze (wahrscheinlich Navigation etc.)
        main_paragraphs = [p for p in paragraphs if len(p) > 100]

        if main_paragraphs:
            return "\n\n".join(main_paragraphs)
        return text

    def extract_article(self, url: str) -> dict[str, Any]:
        """
        Extrahiert einen Artikel mit Metadaten.

        Returns:
            Dict mit title, author, date, content, etc.
        """
        content = self._scraper.scrape(url)

        # Versuche Metadaten zu extrahieren
        author = content.meta.get("author", content.meta.get("og:author"))
        date = content.meta.get("article:published_time",
                               content.meta.get("date",
                               content.meta.get("pubdate")))
        description = content.meta.get("description",
                                       content.meta.get("og:description"))

        return {
            "url": url,
            "title": content.title,
            "author": author,
            "date": date,
            "description": description,
            "content": self.extract_main_content(content.html, url),
            "headings": content.headings,
            "images": content.images[:10],  # Erste 10 Bilder
            "meta": content.meta,
        }

    def extract_structured_data(self, html: str) -> list[dict[str, Any]]:
        """
        Extrahiert strukturierte Daten (JSON-LD, Microdata).
        """
        structured = []

        # JSON-LD extrahieren
        pattern = r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>'
        matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                import json
                data = json.loads(match)
                structured.append(data)
            except json.JSONDecodeError:
                pass

        return structured


class LinkExtractor:
    """
    Link-Extraktion und -Analyse.

    Features:
    - Interne/externe Link-Klassifizierung
    - Broken Link Detection
    - Sitemap-Generierung
    """

    def __init__(self, client: Optional[WebClient] = None):
        self.client = client or WebClient()
        self._scraper = WebScraper(self.client)

    def extract_links(
        self,
        url: str,
        filter_internal: bool = False,
        filter_external: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Extrahiert alle Links von einer Seite.

        Args:
            url: Die zu analysierende URL
            filter_internal: Nur interne Links
            filter_external: Nur externe Links

        Returns:
            Liste von Link-Informationen
        """
        content = self._scraper.scrape(url)
        base_domain = urlparse(url).netloc

        links = []
        for link in content.links:
            parsed = urlparse(link)
            is_internal = parsed.netloc == "" or parsed.netloc == base_domain

            if filter_internal and not is_internal:
                continue
            if filter_external and is_internal:
                continue

            links.append({
                "url": link,
                "is_internal": is_internal,
                "domain": parsed.netloc or base_domain,
                "path": parsed.path,
            })

        return links

    def check_links(
        self,
        urls: list[str],
        timeout: float = 5.0,
    ) -> list[dict[str, Any]]:
        """
        Prüft ob Links erreichbar sind.

        Returns:
            Liste mit Link-Status
        """
        results = []
        for url in urls:
            try:
                response = self.client.head(url, timeout=timeout)
                results.append({
                    "url": url,
                    "status": response.status_code,
                    "ok": response.ok,
                    "redirect": response.status_code in (301, 302, 307, 308),
                })
            except Exception as e:
                results.append({
                    "url": url,
                    "status": 0,
                    "ok": False,
                    "error": str(e),
                })

        return results

    def crawl(
        self,
        start_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        same_domain_only: bool = True,
    ) -> list[ExtractedContent]:
        """
        Crawlt eine Website.

        Args:
            start_url: Start-URL
            max_pages: Maximale Seitenzahl
            max_depth: Maximale Tiefe
            same_domain_only: Nur gleiche Domain

        Returns:
            Liste aller gecrawlten Seiten
        """
        visited: set[str] = set()
        to_visit: list[tuple[str, int]] = [(start_url, 0)]
        results: list[ExtractedContent] = []
        base_domain = urlparse(start_url).netloc

        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)

            if url in visited:
                continue
            if depth > max_depth:
                continue

            visited.add(url)

            try:
                content = self._scraper.scrape(url)
                results.append(content)

                # Neue Links hinzufügen
                for link in content.links:
                    parsed = urlparse(link)

                    # Nur HTTP(S)
                    if parsed.scheme and parsed.scheme not in ("http", "https"):
                        continue

                    # Domain-Filter
                    if same_domain_only:
                        link_domain = parsed.netloc or base_domain
                        if link_domain != base_domain:
                            continue

                    if link not in visited:
                        to_visit.append((link, depth + 1))

                logger.debug("Crawled", url=url[:50], depth=depth, total=len(results))

            except Exception as e:
                logger.warning("Crawl failed", url=url[:50], error=str(e))

        return results


class TableExtractor:
    """
    Tabellen-Extraktion aus HTML.

    Features:
    - HTML-Tabellen parsen
    - In strukturierte Daten konvertieren
    - Header-Erkennung
    """

    def __init__(self):
        self._scraper = WebScraper()

    def extract_tables(self, url: str) -> list[list[list[str]]]:
        """
        Extrahiert alle Tabellen von einer URL.

        Returns:
            Liste von Tabellen (Zeilen x Spalten)
        """
        content = self._scraper.scrape(url)
        return content.tables

    def extract_tables_from_html(self, html: str) -> list[list[list[str]]]:
        """Extrahiert Tabellen aus HTML-String."""
        content = self._scraper.parse_html(html)
        return content.tables

    def tables_to_dicts(
        self,
        tables: list[list[list[str]]],
        header_row: int = 0,
    ) -> list[list[dict[str, str]]]:
        """
        Konvertiert Tabellen zu Listen von Dicts.

        Args:
            tables: Liste von Tabellen
            header_row: Zeile mit den Spaltenüberschriften

        Returns:
            Liste von Tabellen als Dict-Listen
        """
        result = []

        for table in tables:
            if len(table) <= header_row:
                continue

            headers = table[header_row]
            rows = []

            for row in table[header_row + 1:]:
                row_dict = {}
                for i, cell in enumerate(row):
                    if i < len(headers):
                        row_dict[headers[i]] = cell
                    else:
                        row_dict[f"column_{i}"] = cell
                rows.append(row_dict)

            result.append(rows)

        return result

    def find_table_by_header(
        self,
        tables: list[list[list[str]]],
        header_contains: str,
    ) -> Optional[list[list[str]]]:
        """
        Findet eine Tabelle anhand eines Header-Texts.
        """
        header_lower = header_contains.lower()

        for table in tables:
            if table:
                first_row = table[0]
                for cell in first_row:
                    if header_lower in cell.lower():
                        return table

        return None
