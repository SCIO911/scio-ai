"""
SCIO Feed Parser

RSS und Atom Feed Parsing.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from xml.etree import ElementTree as ET

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc
from scio.web.client import WebClient

logger = get_logger(__name__)


class FeedType(str, Enum):
    """Feed-Typen."""
    RSS = "rss"
    RSS2 = "rss2"
    ATOM = "atom"
    UNKNOWN = "unknown"


@dataclass
class FeedEntry:
    """Ein Feed-Eintrag."""

    id: str
    title: str
    link: str
    summary: str = ""
    content: str = ""
    author: Optional[str] = None
    published: Optional[datetime] = None
    updated: Optional[datetime] = None
    categories: list[str] = field(default_factory=list)
    enclosures: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "link": self.link,
            "summary": self.summary,
            "content": self.content,
            "author": self.author,
            "published": self.published.isoformat() if self.published else None,
            "updated": self.updated.isoformat() if self.updated else None,
            "categories": self.categories,
            "enclosures": self.enclosures,
        }


@dataclass
class RSSFeed:
    """RSS Feed Metadaten."""

    title: str
    link: str
    description: str
    feed_type: FeedType
    entries: list[FeedEntry] = field(default_factory=list)
    language: Optional[str] = None
    last_build_date: Optional[datetime] = None
    image_url: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "link": self.link,
            "description": self.description,
            "feed_type": self.feed_type.value,
            "language": self.language,
            "last_build_date": self.last_build_date.isoformat() if self.last_build_date else None,
            "image_url": self.image_url,
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass
class AtomFeed:
    """Atom Feed Metadaten."""

    title: str
    link: str
    subtitle: str = ""
    feed_type: FeedType = FeedType.ATOM
    entries: list[FeedEntry] = field(default_factory=list)
    author: Optional[str] = None
    updated: Optional[datetime] = None
    icon: Optional[str] = None
    logo: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "link": self.link,
            "subtitle": self.subtitle,
            "feed_type": self.feed_type.value,
            "author": self.author,
            "updated": self.updated.isoformat() if self.updated else None,
            "icon": self.icon,
            "logo": self.logo,
            "entries": [e.to_dict() for e in self.entries],
        }


class FeedParser:
    """
    Parser für RSS und Atom Feeds.

    Features:
    - RSS 1.0/2.0 Support
    - Atom Support
    - Automatische Typ-Erkennung
    - Datumskonvertierung
    """

    def __init__(self, client: Optional[WebClient] = None):
        self.client = client or WebClient()
        logger.info("FeedParser initialized")

    def parse_url(self, url: str) -> RSSFeed | AtomFeed:
        """
        Lädt und parst einen Feed von einer URL.

        Args:
            url: Feed-URL

        Returns:
            RSSFeed oder AtomFeed
        """
        response = self.client.get(url)
        response.raise_for_status()
        return self.parse(response.text, url)

    def parse(self, xml_string: str, feed_url: str = "") -> RSSFeed | AtomFeed:
        """
        Parst einen Feed aus einem XML-String.

        Args:
            xml_string: XML-Inhalt
            feed_url: URL des Feeds (für relative Links)

        Returns:
            RSSFeed oder AtomFeed
        """
        feed_type = self._detect_feed_type(xml_string)

        if feed_type == FeedType.ATOM:
            return self._parse_atom(xml_string, feed_url)
        else:
            return self._parse_rss(xml_string, feed_url, feed_type)

    def _detect_feed_type(self, xml_string: str) -> FeedType:
        """Erkennt den Feed-Typ."""
        xml_lower = xml_string.lower()

        if "<feed" in xml_lower and "xmlns" in xml_lower and "atom" in xml_lower:
            return FeedType.ATOM
        elif "<rss" in xml_lower:
            if 'version="2' in xml_lower or "version='2" in xml_lower:
                return FeedType.RSS2
            return FeedType.RSS
        elif "<rdf:" in xml_lower:
            return FeedType.RSS

        return FeedType.UNKNOWN

    def _parse_rss(
        self,
        xml_string: str,
        feed_url: str,
        feed_type: FeedType,
    ) -> RSSFeed:
        """Parst RSS 1.0/2.0 Feeds."""
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.error("RSS parse error", error=str(e))
            return RSSFeed(
                title="Parse Error",
                link=feed_url,
                description=str(e),
                feed_type=feed_type,
            )

        # Namespace-Handling
        ns = {}
        if root.tag.startswith("{"):
            ns_match = re.match(r"\{([^}]+)\}", root.tag)
            if ns_match:
                ns["rss"] = ns_match.group(1)

        # Channel finden
        channel = root.find(".//channel") or root.find(".//{%s}channel" % ns.get("rss", ""))

        if channel is None:
            # RSS 1.0 ohne Channel-Element
            channel = root

        # Feed-Metadaten
        title = self._get_text(channel, "title", ns)
        link = self._get_text(channel, "link", ns)
        description = self._get_text(channel, "description", ns)
        language = self._get_text(channel, "language", ns)

        last_build = self._get_text(channel, "lastBuildDate", ns)
        last_build_date = self._parse_date(last_build) if last_build else None

        image_elem = channel.find("image")
        image_url = None
        if image_elem is not None:
            image_url = self._get_text(image_elem, "url", ns)

        # Einträge parsen
        entries = []
        items = channel.findall("item") or root.findall(".//{%s}item" % ns.get("rss", ""))

        for item in items:
            entry = self._parse_rss_item(item, ns)
            entries.append(entry)

        return RSSFeed(
            title=title,
            link=link or feed_url,
            description=description,
            feed_type=feed_type,
            entries=entries,
            language=language,
            last_build_date=last_build_date,
            image_url=image_url,
        )

    def _parse_rss_item(self, item: ET.Element, ns: dict) -> FeedEntry:
        """Parst ein RSS-Item."""
        title = self._get_text(item, "title", ns)
        link = self._get_text(item, "link", ns)
        description = self._get_text(item, "description", ns)

        # content:encoded mit Namespace-Handling
        content = self._get_namespaced_text(item, "encoded",
            "http://purl.org/rss/1.0/modules/content/") or description

        # Author mit dc:creator Fallback
        author = self._get_text(item, "author", ns) or self._get_namespaced_text(
            item, "creator", "http://purl.org/dc/elements/1.1/")

        guid = self._get_text(item, "guid", ns)

        pub_date_str = self._get_text(item, "pubDate", ns)
        published = self._parse_date(pub_date_str) if pub_date_str else None

        # Kategorien
        categories = []
        for cat in item.findall("category"):
            if cat.text:
                categories.append(cat.text)

        # Enclosures (Podcasts etc.)
        enclosures = []
        for enc in item.findall("enclosure"):
            enclosures.append({
                "url": enc.get("url", ""),
                "type": enc.get("type", ""),
                "length": enc.get("length", ""),
            })

        return FeedEntry(
            id=guid or link or generate_id("entry"),
            title=title,
            link=link,
            summary=description[:500] if description else "",
            content=content,
            author=author,
            published=published,
            categories=categories,
            enclosures=enclosures,
        )

    def _parse_atom(self, xml_string: str, feed_url: str) -> AtomFeed:
        """Parst Atom Feeds."""
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.error("Atom parse error", error=str(e))
            return AtomFeed(
                title="Parse Error",
                link=feed_url,
                subtitle=str(e),
            )

        # Atom Namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Namespace aus Root-Element extrahieren
        if root.tag.startswith("{"):
            ns_match = re.match(r"\{([^}]+)\}", root.tag)
            if ns_match:
                ns["atom"] = ns_match.group(1)

        # Feed-Metadaten
        title = self._get_text(root, "title", ns, "atom") or ""
        subtitle = self._get_text(root, "subtitle", ns, "atom") or ""

        # Link finden (self oder alternate)
        link = ""
        for link_elem in root.findall(".//{%s}link" % ns.get("atom", "")):
            rel = link_elem.get("rel", "alternate")
            if rel in ("alternate", "self"):
                link = link_elem.get("href", "")
                if rel == "alternate":
                    break

        if not link:
            link = feed_url

        # Author
        author_elem = root.find(".//{%s}author" % ns.get("atom", ""))
        author = None
        if author_elem is not None:
            author = self._get_text(author_elem, "name", ns, "atom")

        # Updated
        updated_str = self._get_text(root, "updated", ns, "atom")
        updated = self._parse_date(updated_str) if updated_str else None

        icon = self._get_text(root, "icon", ns, "atom")
        logo = self._get_text(root, "logo", ns, "atom")

        # Einträge parsen
        entries = []
        for entry_elem in root.findall(".//{%s}entry" % ns.get("atom", "")):
            entry = self._parse_atom_entry(entry_elem, ns)
            entries.append(entry)

        return AtomFeed(
            title=title,
            link=link,
            subtitle=subtitle,
            entries=entries,
            author=author,
            updated=updated,
            icon=icon,
            logo=logo,
        )

    def _parse_atom_entry(self, entry_elem: ET.Element, ns: dict) -> FeedEntry:
        """Parst einen Atom-Entry."""
        ns_uri = ns.get("atom", "http://www.w3.org/2005/Atom")

        title = self._get_text(entry_elem, "title", ns, "atom") or ""

        # Link finden
        link = ""
        for link_elem in entry_elem.findall(".//{%s}link" % ns_uri):
            rel = link_elem.get("rel", "alternate")
            if rel == "alternate":
                link = link_elem.get("href", "")
                break
        if not link:
            link_elem = entry_elem.find(".//{%s}link" % ns_uri)
            if link_elem is not None:
                link = link_elem.get("href", "")

        entry_id = self._get_text(entry_elem, "id", ns, "atom") or link
        summary = self._get_text(entry_elem, "summary", ns, "atom") or ""

        # Content
        content_elem = entry_elem.find(".//{%s}content" % ns_uri)
        content = ""
        if content_elem is not None:
            if content_elem.text:
                content = content_elem.text
            else:
                # XHTML content
                content = ET.tostring(content_elem, encoding="unicode", method="text")

        # Author
        author_elem = entry_elem.find(".//{%s}author" % ns_uri)
        author = None
        if author_elem is not None:
            author = self._get_text(author_elem, "name", ns, "atom")

        # Dates
        published_str = self._get_text(entry_elem, "published", ns, "atom")
        updated_str = self._get_text(entry_elem, "updated", ns, "atom")
        published = self._parse_date(published_str) if published_str else None
        updated = self._parse_date(updated_str) if updated_str else None

        # Categories
        categories = []
        for cat in entry_elem.findall(".//{%s}category" % ns_uri):
            term = cat.get("term")
            if term:
                categories.append(term)

        return FeedEntry(
            id=entry_id,
            title=title,
            link=link,
            summary=summary[:500] if summary else "",
            content=content or summary,
            author=author,
            published=published,
            updated=updated,
            categories=categories,
        )

    def _get_text(
        self,
        element: ET.Element,
        tag: str,
        ns: dict,
        prefix: str = "",
    ) -> str:
        """Extrahiert Text aus einem Element."""
        # Versuche mit und ohne Namespace
        if prefix and prefix in ns:
            child = element.find(f".//{{{ns[prefix]}}}{tag}")
        else:
            child = element.find(tag)
            if child is None:
                child = element.find(f".//{tag}")

        if child is not None and child.text:
            return child.text.strip()
        return ""

    def _get_namespaced_text(
        self,
        element: ET.Element,
        local_name: str,
        namespace: str,
    ) -> str:
        """Extrahiert Text aus einem Element mit bekanntem Namespace."""
        try:
            child = element.find(f".//{{{namespace}}}{local_name}")
            if child is not None and child.text:
                return child.text.strip()
        except Exception:
            pass
        return ""

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parst verschiedene Datumsformate."""
        if not date_str:
            return None

        # Verschiedene Formate probieren
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 822
            "%a, %d %b %Y %H:%M:%S %Z",  # RFC 822 mit Zeitzone
            "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",        # ISO 8601 UTC
            "%Y-%m-%d %H:%M:%S",         # Einfach
            "%Y-%m-%d",                   # Nur Datum
        ]

        # Zeitzone-Suffix bereinigen
        date_str = date_str.strip()
        date_str = re.sub(r"\s+\([^)]+\)$", "", date_str)  # "(GMT)" etc. entfernen

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Fallback: Versuche nur Datum zu extrahieren
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if match:
            try:
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                pass

        logger.debug("Could not parse date", date_str=date_str[:50])
        return None

    def get_entries(
        self,
        url: str,
        max_entries: Optional[int] = None,
    ) -> list[FeedEntry]:
        """
        Lädt nur die Einträge eines Feeds.

        Args:
            url: Feed-URL
            max_entries: Maximale Anzahl

        Returns:
            Liste von FeedEntry
        """
        feed = self.parse_url(url)
        entries = feed.entries

        if max_entries:
            entries = entries[:max_entries]

        return entries

    def aggregate(
        self,
        urls: list[str],
        max_per_feed: int = 10,
        sort_by_date: bool = True,
    ) -> list[FeedEntry]:
        """
        Aggregiert mehrere Feeds.

        Args:
            urls: Liste von Feed-URLs
            max_per_feed: Maximale Einträge pro Feed
            sort_by_date: Nach Datum sortieren

        Returns:
            Kombinierte Liste von FeedEntry
        """
        all_entries = []

        for url in urls:
            try:
                entries = self.get_entries(url, max_per_feed)
                all_entries.extend(entries)
            except Exception as e:
                logger.warning("Feed aggregation error", url=url[:50], error=str(e))

        if sort_by_date:
            all_entries.sort(
                key=lambda e: e.published or datetime.min,
                reverse=True,
            )

        return all_entries
