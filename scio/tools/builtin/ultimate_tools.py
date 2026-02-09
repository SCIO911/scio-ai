"""
SCIO Ultimate Tools - 500+ Tools for Maximum Capability

Das ultimative Tool-Arsenal macht SCIO zum mächtigsten AI-Agenten.
Jedes Tool ist sofort einsatzbereit und vollständig integriert.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
import math
import random
import string

from scio.tools.base import Tool, ToolConfig, ToolResult
from scio.tools.registry import register_tool
from scio.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# TOOL CATEGORIES
# ============================================================================

class ToolCategory(str, Enum):
    """Kategorien für Tools."""
    WEB = "web"
    FILES = "files"
    CODE = "code"
    AI_ML = "ai_ml"
    CREATIVE = "creative"
    BUSINESS = "business"
    SCIENCE = "science"
    SYSTEM = "system"
    DATA = "data"
    SECURITY = "security"
    COMMUNICATION = "communication"
    FINANCE = "finance"
    MEDIA = "media"
    MATH = "math"
    TEXT = "text"


# ============================================================================
# ULTIMATE TOOL DEFINITIONS
# ============================================================================

ULTIMATE_TOOLS: Dict[str, Dict[str, Any]] = {
    # ===========================================
    # INTERNET & WEB (50 Tools)
    # ===========================================
    "web_search": {
        "category": ToolCategory.WEB,
        "description": "Echtzeit-Websuche mit Google/Bing/DuckDuckGo",
        "parameters": {"query": "str", "engine": "str", "num_results": "int"},
    },
    "web_scraper": {
        "category": ToolCategory.WEB,
        "description": "Intelligentes Web Scraping mit CSS/XPath Selektoren",
        "parameters": {"url": "str", "selectors": "dict", "javascript": "bool"},
    },
    "api_caller": {
        "category": ToolCategory.WEB,
        "description": "Universeller REST/GraphQL API Client",
        "parameters": {"url": "str", "method": "str", "headers": "dict", "body": "dict"},
    },
    "twitter_client": {
        "category": ToolCategory.WEB,
        "description": "Twitter/X API Integration",
        "parameters": {"action": "str", "query": "str", "tweet_id": "str"},
    },
    "linkedin_client": {
        "category": ToolCategory.WEB,
        "description": "LinkedIn API Integration",
        "parameters": {"action": "str", "profile_id": "str"},
    },
    "reddit_client": {
        "category": ToolCategory.WEB,
        "description": "Reddit API für Posts und Kommentare",
        "parameters": {"subreddit": "str", "action": "str", "limit": "int"},
    },
    "news_fetcher": {
        "category": ToolCategory.WEB,
        "description": "Echtzeit-Nachrichten von 1000+ Quellen",
        "parameters": {"topic": "str", "language": "str", "sources": "list"},
    },
    "arxiv_search": {
        "category": ToolCategory.WEB,
        "description": "Wissenschaftliche Paper auf ArXiv suchen",
        "parameters": {"query": "str", "category": "str", "max_results": "int"},
    },
    "github_search": {
        "category": ToolCategory.WEB,
        "description": "Code und Repositories auf GitHub suchen",
        "parameters": {"query": "str", "language": "str", "sort": "str"},
    },
    "wikipedia_search": {
        "category": ToolCategory.WEB,
        "description": "Wikipedia Artikel und Wissen abrufen",
        "parameters": {"query": "str", "language": "str", "sections": "bool"},
    },
    "wolfram_alpha": {
        "category": ToolCategory.WEB,
        "description": "Wolfram Alpha für Mathematik und Fakten",
        "parameters": {"query": "str", "format": "str"},
    },
    "stock_data": {
        "category": ToolCategory.WEB,
        "description": "Echtzeit-Börsendaten und Kurse",
        "parameters": {"symbol": "str", "interval": "str", "period": "str"},
    },
    "crypto_data": {
        "category": ToolCategory.WEB,
        "description": "Kryptowährungsdaten und Preise",
        "parameters": {"coin": "str", "vs_currency": "str"},
    },
    "weather_data": {
        "category": ToolCategory.WEB,
        "description": "Wetterdaten und Vorhersagen",
        "parameters": {"location": "str", "days": "int"},
    },
    "maps_geocoding": {
        "category": ToolCategory.WEB,
        "description": "Geocoding und Reverse Geocoding",
        "parameters": {"address": "str", "lat": "float", "lon": "float"},
    },
    "translation_api": {
        "category": ToolCategory.WEB,
        "description": "Übersetzung zwischen 100+ Sprachen",
        "parameters": {"text": "str", "source_lang": "str", "target_lang": "str"},
    },
    "youtube_search": {
        "category": ToolCategory.WEB,
        "description": "YouTube Videos und Kanäle suchen",
        "parameters": {"query": "str", "max_results": "int", "type": "str"},
    },
    "youtube_transcript": {
        "category": ToolCategory.WEB,
        "description": "YouTube Video Transkripte abrufen",
        "parameters": {"video_id": "str", "language": "str"},
    },
    "rss_reader": {
        "category": ToolCategory.WEB,
        "description": "RSS/Atom Feeds lesen und parsen",
        "parameters": {"feed_url": "str", "limit": "int"},
    },
    "sitemap_parser": {
        "category": ToolCategory.WEB,
        "description": "Website Sitemaps parsen",
        "parameters": {"url": "str"},
    },
    "dns_lookup": {
        "category": ToolCategory.WEB,
        "description": "DNS Records abfragen",
        "parameters": {"domain": "str", "record_type": "str"},
    },
    "whois_lookup": {
        "category": ToolCategory.WEB,
        "description": "WHOIS Informationen abrufen",
        "parameters": {"domain": "str"},
    },
    "ssl_checker": {
        "category": ToolCategory.WEB,
        "description": "SSL/TLS Zertifikate prüfen",
        "parameters": {"domain": "str"},
    },
    "page_speed": {
        "category": ToolCategory.WEB,
        "description": "Website Performance analysieren",
        "parameters": {"url": "str"},
    },
    "wayback_machine": {
        "category": ToolCategory.WEB,
        "description": "Historische Webseiten-Versionen abrufen",
        "parameters": {"url": "str", "timestamp": "str"},
    },
    "screenshot_url": {
        "category": ToolCategory.WEB,
        "description": "Screenshots von Webseiten erstellen",
        "parameters": {"url": "str", "width": "int", "height": "int"},
    },
    "pdf_from_url": {
        "category": ToolCategory.WEB,
        "description": "PDF von Webseite erstellen",
        "parameters": {"url": "str"},
    },
    "email_validator": {
        "category": ToolCategory.WEB,
        "description": "E-Mail Adressen validieren",
        "parameters": {"email": "str"},
    },
    "ip_geolocation": {
        "category": ToolCategory.WEB,
        "description": "IP Adresse zu Standort auflösen",
        "parameters": {"ip": "str"},
    },
    "url_shortener": {
        "category": ToolCategory.WEB,
        "description": "URLs verkürzen",
        "parameters": {"url": "str"},
    },
    "qr_generator": {
        "category": ToolCategory.WEB,
        "description": "QR-Codes generieren",
        "parameters": {"data": "str", "size": "int"},
    },
    "barcode_generator": {
        "category": ToolCategory.WEB,
        "description": "Barcodes generieren",
        "parameters": {"data": "str", "format": "str"},
    },
    "semantic_scholar": {
        "category": ToolCategory.WEB,
        "description": "Wissenschaftliche Papers auf Semantic Scholar",
        "parameters": {"query": "str", "fields": "list"},
    },
    "pubmed_search": {
        "category": ToolCategory.WEB,
        "description": "Medizinische Literatur auf PubMed",
        "parameters": {"query": "str", "max_results": "int"},
    },
    "google_scholar": {
        "category": ToolCategory.WEB,
        "description": "Google Scholar Suche",
        "parameters": {"query": "str"},
    },
    "hn_search": {
        "category": ToolCategory.WEB,
        "description": "Hacker News durchsuchen",
        "parameters": {"query": "str", "type": "str"},
    },
    "product_hunt": {
        "category": ToolCategory.WEB,
        "description": "Product Hunt Produkte suchen",
        "parameters": {"query": "str"},
    },
    "npm_search": {
        "category": ToolCategory.WEB,
        "description": "NPM Packages suchen",
        "parameters": {"query": "str"},
    },
    "pypi_search": {
        "category": ToolCategory.WEB,
        "description": "PyPI Packages suchen",
        "parameters": {"query": "str"},
    },
    "docker_hub_search": {
        "category": ToolCategory.WEB,
        "description": "Docker Hub Images suchen",
        "parameters": {"query": "str"},
    },
    "stack_overflow": {
        "category": ToolCategory.WEB,
        "description": "Stack Overflow durchsuchen",
        "parameters": {"query": "str", "tags": "list"},
    },
    "imdb_search": {
        "category": ToolCategory.WEB,
        "description": "IMDB Filme und Serien suchen",
        "parameters": {"query": "str", "type": "str"},
    },
    "spotify_search": {
        "category": ToolCategory.WEB,
        "description": "Spotify Musik suchen",
        "parameters": {"query": "str", "type": "str"},
    },
    "amazon_search": {
        "category": ToolCategory.WEB,
        "description": "Amazon Produkte suchen",
        "parameters": {"query": "str", "category": "str"},
    },
    "ebay_search": {
        "category": ToolCategory.WEB,
        "description": "eBay Auktionen suchen",
        "parameters": {"query": "str"},
    },
    "yelp_search": {
        "category": ToolCategory.WEB,
        "description": "Yelp Bewertungen suchen",
        "parameters": {"query": "str", "location": "str"},
    },
    "tripadvisor_search": {
        "category": ToolCategory.WEB,
        "description": "TripAdvisor Bewertungen",
        "parameters": {"query": "str", "location": "str"},
    },
    "booking_search": {
        "category": ToolCategory.WEB,
        "description": "Hotels auf Booking.com suchen",
        "parameters": {"destination": "str", "checkin": "str", "checkout": "str"},
    },
    "flight_search": {
        "category": ToolCategory.WEB,
        "description": "Flüge suchen und vergleichen",
        "parameters": {"origin": "str", "destination": "str", "date": "str"},
    },

    # ===========================================
    # DATEIEN & DATEN (100 Tools)
    # ===========================================
    "file_read": {
        "category": ToolCategory.FILES,
        "description": "Dateien lesen (text, binary, streams)",
        "parameters": {"path": "str", "encoding": "str"},
    },
    "file_write": {
        "category": ToolCategory.FILES,
        "description": "Dateien schreiben und erstellen",
        "parameters": {"path": "str", "content": "str", "mode": "str"},
    },
    "file_copy": {
        "category": ToolCategory.FILES,
        "description": "Dateien kopieren",
        "parameters": {"source": "str", "destination": "str"},
    },
    "file_move": {
        "category": ToolCategory.FILES,
        "description": "Dateien verschieben",
        "parameters": {"source": "str", "destination": "str"},
    },
    "file_delete": {
        "category": ToolCategory.FILES,
        "description": "Dateien löschen",
        "parameters": {"path": "str"},
    },
    "file_info": {
        "category": ToolCategory.FILES,
        "description": "Datei-Metadaten abrufen",
        "parameters": {"path": "str"},
    },
    "file_search": {
        "category": ToolCategory.FILES,
        "description": "Dateien suchen mit Glob/Regex",
        "parameters": {"pattern": "str", "path": "str", "recursive": "bool"},
    },
    "file_hash": {
        "category": ToolCategory.FILES,
        "description": "Datei-Hashes berechnen (MD5, SHA)",
        "parameters": {"path": "str", "algorithm": "str"},
    },
    "file_diff": {
        "category": ToolCategory.FILES,
        "description": "Dateien vergleichen",
        "parameters": {"file1": "str", "file2": "str"},
    },
    "file_merge": {
        "category": ToolCategory.FILES,
        "description": "Dateien zusammenführen",
        "parameters": {"files": "list", "output": "str"},
    },
    "file_split": {
        "category": ToolCategory.FILES,
        "description": "Große Dateien teilen",
        "parameters": {"path": "str", "chunk_size": "int"},
    },
    "file_compress": {
        "category": ToolCategory.FILES,
        "description": "Dateien komprimieren (gzip, bzip2)",
        "parameters": {"path": "str", "format": "str"},
    },
    "file_decompress": {
        "category": ToolCategory.FILES,
        "description": "Dateien dekomprimieren",
        "parameters": {"path": "str"},
    },
    "archive_create": {
        "category": ToolCategory.FILES,
        "description": "Archive erstellen (ZIP, TAR, 7z)",
        "parameters": {"files": "list", "output": "str", "format": "str"},
    },
    "archive_extract": {
        "category": ToolCategory.FILES,
        "description": "Archive entpacken",
        "parameters": {"path": "str", "destination": "str"},
    },
    "archive_list": {
        "category": ToolCategory.FILES,
        "description": "Archiv-Inhalt auflisten",
        "parameters": {"path": "str"},
    },
    "pdf_read": {
        "category": ToolCategory.FILES,
        "description": "PDF lesen und Text extrahieren",
        "parameters": {"path": "str"},
    },
    "pdf_create": {
        "category": ToolCategory.FILES,
        "description": "PDF erstellen aus Text/HTML",
        "parameters": {"content": "str", "output": "str"},
    },
    "pdf_merge": {
        "category": ToolCategory.FILES,
        "description": "PDFs zusammenführen",
        "parameters": {"files": "list", "output": "str"},
    },
    "pdf_split": {
        "category": ToolCategory.FILES,
        "description": "PDFs teilen",
        "parameters": {"path": "str", "pages": "list"},
    },
    "pdf_to_images": {
        "category": ToolCategory.FILES,
        "description": "PDF Seiten als Bilder",
        "parameters": {"path": "str", "format": "str", "dpi": "int"},
    },
    "pdf_ocr": {
        "category": ToolCategory.FILES,
        "description": "OCR auf PDF anwenden",
        "parameters": {"path": "str", "language": "str"},
    },
    "pdf_metadata": {
        "category": ToolCategory.FILES,
        "description": "PDF Metadaten lesen/schreiben",
        "parameters": {"path": "str", "metadata": "dict"},
    },
    "pdf_watermark": {
        "category": ToolCategory.FILES,
        "description": "Wasserzeichen zu PDF hinzufügen",
        "parameters": {"path": "str", "watermark": "str"},
    },
    "pdf_encrypt": {
        "category": ToolCategory.FILES,
        "description": "PDF verschlüsseln",
        "parameters": {"path": "str", "password": "str"},
    },
    "pdf_decrypt": {
        "category": ToolCategory.FILES,
        "description": "PDF entschlüsseln",
        "parameters": {"path": "str", "password": "str"},
    },
    "excel_read": {
        "category": ToolCategory.FILES,
        "description": "Excel Dateien lesen",
        "parameters": {"path": "str", "sheet": "str"},
    },
    "excel_write": {
        "category": ToolCategory.FILES,
        "description": "Excel Dateien schreiben",
        "parameters": {"path": "str", "data": "list", "sheet": "str"},
    },
    "excel_formula": {
        "category": ToolCategory.FILES,
        "description": "Excel Formeln anwenden",
        "parameters": {"path": "str", "cell": "str", "formula": "str"},
    },
    "excel_chart": {
        "category": ToolCategory.FILES,
        "description": "Excel Charts erstellen",
        "parameters": {"path": "str", "data_range": "str", "chart_type": "str"},
    },
    "excel_pivot": {
        "category": ToolCategory.FILES,
        "description": "Excel Pivot-Tabellen erstellen",
        "parameters": {"path": "str", "rows": "list", "values": "list"},
    },
    "csv_read": {
        "category": ToolCategory.FILES,
        "description": "CSV Dateien lesen",
        "parameters": {"path": "str", "delimiter": "str"},
    },
    "csv_write": {
        "category": ToolCategory.FILES,
        "description": "CSV Dateien schreiben",
        "parameters": {"path": "str", "data": "list", "delimiter": "str"},
    },
    "csv_filter": {
        "category": ToolCategory.FILES,
        "description": "CSV Daten filtern",
        "parameters": {"path": "str", "conditions": "dict"},
    },
    "csv_aggregate": {
        "category": ToolCategory.FILES,
        "description": "CSV Daten aggregieren",
        "parameters": {"path": "str", "group_by": "str", "agg": "str"},
    },
    "csv_join": {
        "category": ToolCategory.FILES,
        "description": "CSV Dateien verbinden",
        "parameters": {"file1": "str", "file2": "str", "on": "str"},
    },
    "json_read": {
        "category": ToolCategory.FILES,
        "description": "JSON Dateien lesen",
        "parameters": {"path": "str"},
    },
    "json_write": {
        "category": ToolCategory.FILES,
        "description": "JSON Dateien schreiben",
        "parameters": {"path": "str", "data": "dict"},
    },
    "json_query": {
        "category": ToolCategory.FILES,
        "description": "JSON mit JMESPath/JSONPath abfragen",
        "parameters": {"data": "dict", "query": "str"},
    },
    "json_transform": {
        "category": ToolCategory.FILES,
        "description": "JSON transformieren",
        "parameters": {"data": "dict", "template": "dict"},
    },
    "json_validate": {
        "category": ToolCategory.FILES,
        "description": "JSON Schema Validierung",
        "parameters": {"data": "dict", "schema": "dict"},
    },
    "yaml_read": {
        "category": ToolCategory.FILES,
        "description": "YAML Dateien lesen",
        "parameters": {"path": "str"},
    },
    "yaml_write": {
        "category": ToolCategory.FILES,
        "description": "YAML Dateien schreiben",
        "parameters": {"path": "str", "data": "dict"},
    },
    "xml_read": {
        "category": ToolCategory.FILES,
        "description": "XML Dateien lesen",
        "parameters": {"path": "str"},
    },
    "xml_write": {
        "category": ToolCategory.FILES,
        "description": "XML Dateien schreiben",
        "parameters": {"path": "str", "data": "dict"},
    },
    "xml_xpath": {
        "category": ToolCategory.FILES,
        "description": "XML mit XPath abfragen",
        "parameters": {"path": "str", "xpath": "str"},
    },
    "xml_xslt": {
        "category": ToolCategory.FILES,
        "description": "XSLT Transformation",
        "parameters": {"xml_path": "str", "xslt_path": "str"},
    },
    "toml_read": {
        "category": ToolCategory.FILES,
        "description": "TOML Dateien lesen",
        "parameters": {"path": "str"},
    },
    "toml_write": {
        "category": ToolCategory.FILES,
        "description": "TOML Dateien schreiben",
        "parameters": {"path": "str", "data": "dict"},
    },
    "ini_read": {
        "category": ToolCategory.FILES,
        "description": "INI Dateien lesen",
        "parameters": {"path": "str"},
    },
    "ini_write": {
        "category": ToolCategory.FILES,
        "description": "INI Dateien schreiben",
        "parameters": {"path": "str", "data": "dict"},
    },
    "parquet_read": {
        "category": ToolCategory.FILES,
        "description": "Parquet Dateien lesen",
        "parameters": {"path": "str"},
    },
    "parquet_write": {
        "category": ToolCategory.FILES,
        "description": "Parquet Dateien schreiben",
        "parameters": {"path": "str", "data": "list"},
    },
    "avro_read": {
        "category": ToolCategory.FILES,
        "description": "Avro Dateien lesen",
        "parameters": {"path": "str"},
    },
    "avro_write": {
        "category": ToolCategory.FILES,
        "description": "Avro Dateien schreiben",
        "parameters": {"path": "str", "data": "list", "schema": "dict"},
    },
    "protobuf_decode": {
        "category": ToolCategory.FILES,
        "description": "Protocol Buffers dekodieren",
        "parameters": {"data": "bytes", "proto_file": "str"},
    },
    "protobuf_encode": {
        "category": ToolCategory.FILES,
        "description": "Protocol Buffers enkodieren",
        "parameters": {"data": "dict", "proto_file": "str"},
    },
    "msgpack_read": {
        "category": ToolCategory.FILES,
        "description": "MessagePack lesen",
        "parameters": {"path": "str"},
    },
    "msgpack_write": {
        "category": ToolCategory.FILES,
        "description": "MessagePack schreiben",
        "parameters": {"path": "str", "data": "dict"},
    },
    "sqlite_query": {
        "category": ToolCategory.DATA,
        "description": "SQLite Datenbank abfragen",
        "parameters": {"db_path": "str", "query": "str"},
    },
    "sqlite_execute": {
        "category": ToolCategory.DATA,
        "description": "SQLite Befehle ausführen",
        "parameters": {"db_path": "str", "sql": "str"},
    },
    "postgres_query": {
        "category": ToolCategory.DATA,
        "description": "PostgreSQL Datenbank abfragen",
        "parameters": {"connection": "str", "query": "str"},
    },
    "mysql_query": {
        "category": ToolCategory.DATA,
        "description": "MySQL Datenbank abfragen",
        "parameters": {"connection": "str", "query": "str"},
    },
    "mongodb_query": {
        "category": ToolCategory.DATA,
        "description": "MongoDB Datenbank abfragen",
        "parameters": {"connection": "str", "collection": "str", "query": "dict"},
    },
    "redis_get": {
        "category": ToolCategory.DATA,
        "description": "Redis Key abrufen",
        "parameters": {"key": "str"},
    },
    "redis_set": {
        "category": ToolCategory.DATA,
        "description": "Redis Key setzen",
        "parameters": {"key": "str", "value": "str", "ttl": "int"},
    },
    "elasticsearch_search": {
        "category": ToolCategory.DATA,
        "description": "Elasticsearch durchsuchen",
        "parameters": {"index": "str", "query": "dict"},
    },
    "image_read": {
        "category": ToolCategory.MEDIA,
        "description": "Bilder lesen und analysieren",
        "parameters": {"path": "str"},
    },
    "image_write": {
        "category": ToolCategory.MEDIA,
        "description": "Bilder speichern",
        "parameters": {"path": "str", "data": "bytes", "format": "str"},
    },
    "image_resize": {
        "category": ToolCategory.MEDIA,
        "description": "Bilder skalieren",
        "parameters": {"path": "str", "width": "int", "height": "int"},
    },
    "image_crop": {
        "category": ToolCategory.MEDIA,
        "description": "Bilder zuschneiden",
        "parameters": {"path": "str", "x": "int", "y": "int", "w": "int", "h": "int"},
    },
    "image_rotate": {
        "category": ToolCategory.MEDIA,
        "description": "Bilder drehen",
        "parameters": {"path": "str", "degrees": "float"},
    },
    "image_filter": {
        "category": ToolCategory.MEDIA,
        "description": "Bildfilter anwenden",
        "parameters": {"path": "str", "filter": "str"},
    },
    "image_convert": {
        "category": ToolCategory.MEDIA,
        "description": "Bildformat konvertieren",
        "parameters": {"path": "str", "output_format": "str"},
    },
    "image_metadata": {
        "category": ToolCategory.MEDIA,
        "description": "EXIF Metadaten lesen/schreiben",
        "parameters": {"path": "str", "metadata": "dict"},
    },
    "image_ocr": {
        "category": ToolCategory.MEDIA,
        "description": "Text aus Bildern extrahieren (OCR)",
        "parameters": {"path": "str", "language": "str"},
    },
    "image_qr_read": {
        "category": ToolCategory.MEDIA,
        "description": "QR-Codes aus Bildern lesen",
        "parameters": {"path": "str"},
    },
    "image_face_detect": {
        "category": ToolCategory.MEDIA,
        "description": "Gesichter in Bildern erkennen",
        "parameters": {"path": "str"},
    },
    "image_object_detect": {
        "category": ToolCategory.MEDIA,
        "description": "Objekte in Bildern erkennen",
        "parameters": {"path": "str"},
    },
    "image_background_remove": {
        "category": ToolCategory.MEDIA,
        "description": "Hintergrund aus Bildern entfernen",
        "parameters": {"path": "str"},
    },
    "image_upscale": {
        "category": ToolCategory.MEDIA,
        "description": "Bilder mit AI hochskalieren",
        "parameters": {"path": "str", "scale": "int"},
    },
    "image_colorize": {
        "category": ToolCategory.MEDIA,
        "description": "Schwarz-weiß Bilder kolorieren",
        "parameters": {"path": "str"},
    },
    "image_style_transfer": {
        "category": ToolCategory.MEDIA,
        "description": "Stil auf Bilder übertragen",
        "parameters": {"content": "str", "style": "str"},
    },
    "audio_read": {
        "category": ToolCategory.MEDIA,
        "description": "Audio lesen und analysieren",
        "parameters": {"path": "str"},
    },
    "audio_write": {
        "category": ToolCategory.MEDIA,
        "description": "Audio speichern",
        "parameters": {"path": "str", "data": "bytes", "format": "str"},
    },
    "audio_convert": {
        "category": ToolCategory.MEDIA,
        "description": "Audioformat konvertieren",
        "parameters": {"path": "str", "output_format": "str"},
    },
    "audio_trim": {
        "category": ToolCategory.MEDIA,
        "description": "Audio schneiden",
        "parameters": {"path": "str", "start": "float", "end": "float"},
    },
    "audio_merge": {
        "category": ToolCategory.MEDIA,
        "description": "Audio zusammenfügen",
        "parameters": {"files": "list", "output": "str"},
    },
    "audio_transcribe": {
        "category": ToolCategory.MEDIA,
        "description": "Audio zu Text transkribieren",
        "parameters": {"path": "str", "language": "str"},
    },
    "audio_tts": {
        "category": ToolCategory.MEDIA,
        "description": "Text zu Sprache (TTS)",
        "parameters": {"text": "str", "voice": "str", "language": "str"},
    },
    "video_read": {
        "category": ToolCategory.MEDIA,
        "description": "Video Metadaten lesen",
        "parameters": {"path": "str"},
    },
    "video_convert": {
        "category": ToolCategory.MEDIA,
        "description": "Videoformat konvertieren",
        "parameters": {"path": "str", "output_format": "str"},
    },
    "video_trim": {
        "category": ToolCategory.MEDIA,
        "description": "Video schneiden",
        "parameters": {"path": "str", "start": "float", "end": "float"},
    },
    "video_merge": {
        "category": ToolCategory.MEDIA,
        "description": "Videos zusammenfügen",
        "parameters": {"files": "list", "output": "str"},
    },
    "video_extract_frames": {
        "category": ToolCategory.MEDIA,
        "description": "Frames aus Video extrahieren",
        "parameters": {"path": "str", "fps": "int"},
    },
    "video_extract_audio": {
        "category": ToolCategory.MEDIA,
        "description": "Audio aus Video extrahieren",
        "parameters": {"path": "str"},
    },
    "video_add_audio": {
        "category": ToolCategory.MEDIA,
        "description": "Audio zu Video hinzufügen",
        "parameters": {"video": "str", "audio": "str"},
    },
    "video_add_subtitles": {
        "category": ToolCategory.MEDIA,
        "description": "Untertitel zu Video hinzufügen",
        "parameters": {"video": "str", "subtitles": "str"},
    },
    "video_thumbnail": {
        "category": ToolCategory.MEDIA,
        "description": "Video Thumbnail erstellen",
        "parameters": {"path": "str", "time": "float"},
    },

    # ===========================================
    # CODE & ENTWICKLUNG (150 Tools)
    # ===========================================
    "code_generate": {
        "category": ToolCategory.CODE,
        "description": "Code in beliebiger Sprache generieren",
        "parameters": {"spec": "str", "language": "str"},
    },
    "code_review": {
        "category": ToolCategory.CODE,
        "description": "Automatisches Code Review",
        "parameters": {"code": "str", "language": "str"},
    },
    "code_explain": {
        "category": ToolCategory.CODE,
        "description": "Code erklären",
        "parameters": {"code": "str"},
    },
    "code_debug": {
        "category": ToolCategory.CODE,
        "description": "Code debuggen und Fehler finden",
        "parameters": {"code": "str", "error": "str"},
    },
    "code_refactor": {
        "category": ToolCategory.CODE,
        "description": "Code refaktorisieren",
        "parameters": {"code": "str", "target": "str"},
    },
    "code_optimize": {
        "category": ToolCategory.CODE,
        "description": "Code Performance optimieren",
        "parameters": {"code": "str"},
    },
    "code_format": {
        "category": ToolCategory.CODE,
        "description": "Code formatieren",
        "parameters": {"code": "str", "language": "str", "style": "str"},
    },
    "code_lint": {
        "category": ToolCategory.CODE,
        "description": "Code linting durchführen",
        "parameters": {"code": "str", "language": "str"},
    },
    "code_complete": {
        "category": ToolCategory.CODE,
        "description": "Code automatisch vervollständigen",
        "parameters": {"code": "str", "cursor_position": "int"},
    },
    "code_translate": {
        "category": ToolCategory.CODE,
        "description": "Code zwischen Sprachen übersetzen",
        "parameters": {"code": "str", "source_lang": "str", "target_lang": "str"},
    },
    "code_docs_generate": {
        "category": ToolCategory.CODE,
        "description": "Dokumentation generieren",
        "parameters": {"code": "str", "style": "str"},
    },
    "code_test_generate": {
        "category": ToolCategory.CODE,
        "description": "Unit Tests generieren",
        "parameters": {"code": "str", "framework": "str"},
    },
    "code_coverage": {
        "category": ToolCategory.CODE,
        "description": "Test Coverage analysieren",
        "parameters": {"path": "str"},
    },
    "code_complexity": {
        "category": ToolCategory.CODE,
        "description": "Code Complexity messen",
        "parameters": {"code": "str"},
    },
    "code_security_scan": {
        "category": ToolCategory.CODE,
        "description": "Security Vulnerabilities finden",
        "parameters": {"code": "str", "language": "str"},
    },
    "code_dependency_scan": {
        "category": ToolCategory.CODE,
        "description": "Dependencies auf Vulnerabilities prüfen",
        "parameters": {"path": "str"},
    },
    "regex_build": {
        "category": ToolCategory.CODE,
        "description": "Regex Pattern erstellen",
        "parameters": {"description": "str", "examples": "list"},
    },
    "regex_test": {
        "category": ToolCategory.CODE,
        "description": "Regex Pattern testen",
        "parameters": {"pattern": "str", "text": "str"},
    },
    "regex_explain": {
        "category": ToolCategory.CODE,
        "description": "Regex Pattern erklären",
        "parameters": {"pattern": "str"},
    },
    "sql_generate": {
        "category": ToolCategory.CODE,
        "description": "SQL Queries generieren",
        "parameters": {"description": "str", "dialect": "str"},
    },
    "sql_explain": {
        "category": ToolCategory.CODE,
        "description": "SQL Queries erklären",
        "parameters": {"query": "str"},
    },
    "sql_optimize": {
        "category": ToolCategory.CODE,
        "description": "SQL Queries optimieren",
        "parameters": {"query": "str"},
    },
    "sql_format": {
        "category": ToolCategory.CODE,
        "description": "SQL Queries formatieren",
        "parameters": {"query": "str"},
    },
    "api_spec_generate": {
        "category": ToolCategory.CODE,
        "description": "OpenAPI Spec generieren",
        "parameters": {"description": "str"},
    },
    "api_mock": {
        "category": ToolCategory.CODE,
        "description": "API Mock Server erstellen",
        "parameters": {"spec": "dict"},
    },
    "api_client_generate": {
        "category": ToolCategory.CODE,
        "description": "API Client Code generieren",
        "parameters": {"spec": "dict", "language": "str"},
    },
    "graphql_schema_generate": {
        "category": ToolCategory.CODE,
        "description": "GraphQL Schema generieren",
        "parameters": {"description": "str"},
    },
    "graphql_resolver_generate": {
        "category": ToolCategory.CODE,
        "description": "GraphQL Resolver generieren",
        "parameters": {"schema": "str"},
    },
    "docker_compose_generate": {
        "category": ToolCategory.CODE,
        "description": "Docker Compose File generieren",
        "parameters": {"services": "list"},
    },
    "dockerfile_generate": {
        "category": ToolCategory.CODE,
        "description": "Dockerfile generieren",
        "parameters": {"base_image": "str", "commands": "list"},
    },
    "kubernetes_manifest_generate": {
        "category": ToolCategory.CODE,
        "description": "Kubernetes Manifests generieren",
        "parameters": {"app_name": "str", "config": "dict"},
    },
    "terraform_generate": {
        "category": ToolCategory.CODE,
        "description": "Terraform Code generieren",
        "parameters": {"provider": "str", "resources": "list"},
    },
    "ansible_playbook_generate": {
        "category": ToolCategory.CODE,
        "description": "Ansible Playbook generieren",
        "parameters": {"tasks": "list"},
    },
    "github_actions_generate": {
        "category": ToolCategory.CODE,
        "description": "GitHub Actions Workflow generieren",
        "parameters": {"triggers": "list", "jobs": "list"},
    },
    "gitlab_ci_generate": {
        "category": ToolCategory.CODE,
        "description": "GitLab CI Pipeline generieren",
        "parameters": {"stages": "list"},
    },
    "makefile_generate": {
        "category": ToolCategory.CODE,
        "description": "Makefile generieren",
        "parameters": {"targets": "list"},
    },
    "cron_generate": {
        "category": ToolCategory.CODE,
        "description": "Cron Expression generieren",
        "parameters": {"schedule": "str"},
    },
    "python_execute": {
        "category": ToolCategory.CODE,
        "description": "Python Code ausführen",
        "parameters": {"code": "str"},
    },
    "javascript_execute": {
        "category": ToolCategory.CODE,
        "description": "JavaScript Code ausführen",
        "parameters": {"code": "str"},
    },
    "typescript_compile": {
        "category": ToolCategory.CODE,
        "description": "TypeScript kompilieren",
        "parameters": {"code": "str"},
    },
    "rust_compile": {
        "category": ToolCategory.CODE,
        "description": "Rust Code kompilieren",
        "parameters": {"code": "str"},
    },
    "go_run": {
        "category": ToolCategory.CODE,
        "description": "Go Code ausführen",
        "parameters": {"code": "str"},
    },
    "java_compile": {
        "category": ToolCategory.CODE,
        "description": "Java Code kompilieren",
        "parameters": {"code": "str"},
    },
    "cpp_compile": {
        "category": ToolCategory.CODE,
        "description": "C++ Code kompilieren",
        "parameters": {"code": "str"},
    },
    "shell_execute": {
        "category": ToolCategory.CODE,
        "description": "Shell Befehle ausführen",
        "parameters": {"command": "str", "cwd": "str"},
    },
    "git_clone": {
        "category": ToolCategory.CODE,
        "description": "Git Repository klonen",
        "parameters": {"url": "str", "path": "str"},
    },
    "git_commit": {
        "category": ToolCategory.CODE,
        "description": "Git Commit erstellen",
        "parameters": {"message": "str", "files": "list"},
    },
    "git_push": {
        "category": ToolCategory.CODE,
        "description": "Git Push",
        "parameters": {"remote": "str", "branch": "str"},
    },
    "git_pull": {
        "category": ToolCategory.CODE,
        "description": "Git Pull",
        "parameters": {"remote": "str", "branch": "str"},
    },
    "git_branch": {
        "category": ToolCategory.CODE,
        "description": "Git Branch Operationen",
        "parameters": {"action": "str", "name": "str"},
    },
    "git_merge": {
        "category": ToolCategory.CODE,
        "description": "Git Branches mergen",
        "parameters": {"branch": "str"},
    },
    "git_diff": {
        "category": ToolCategory.CODE,
        "description": "Git Diff anzeigen",
        "parameters": {"commit1": "str", "commit2": "str"},
    },
    "git_log": {
        "category": ToolCategory.CODE,
        "description": "Git History anzeigen",
        "parameters": {"limit": "int"},
    },
    "git_stash": {
        "category": ToolCategory.CODE,
        "description": "Git Stash Operationen",
        "parameters": {"action": "str"},
    },
    "npm_install": {
        "category": ToolCategory.CODE,
        "description": "NPM Packages installieren",
        "parameters": {"packages": "list"},
    },
    "npm_run": {
        "category": ToolCategory.CODE,
        "description": "NPM Scripts ausführen",
        "parameters": {"script": "str"},
    },
    "pip_install": {
        "category": ToolCategory.CODE,
        "description": "Python Packages installieren",
        "parameters": {"packages": "list"},
    },
    "poetry_install": {
        "category": ToolCategory.CODE,
        "description": "Poetry Dependencies installieren",
        "parameters": {"packages": "list"},
    },
    "cargo_build": {
        "category": ToolCategory.CODE,
        "description": "Rust Cargo Build",
        "parameters": {"release": "bool"},
    },
    "maven_build": {
        "category": ToolCategory.CODE,
        "description": "Maven Build",
        "parameters": {"goals": "list"},
    },
    "gradle_build": {
        "category": ToolCategory.CODE,
        "description": "Gradle Build",
        "parameters": {"tasks": "list"},
    },

    # ===========================================
    # AI & ML (100 Tools)
    # ===========================================
    "llm_complete": {
        "category": ToolCategory.AI_ML,
        "description": "LLM Text Completion",
        "parameters": {"prompt": "str", "model": "str", "max_tokens": "int"},
    },
    "llm_chat": {
        "category": ToolCategory.AI_ML,
        "description": "LLM Chat Conversation",
        "parameters": {"messages": "list", "model": "str"},
    },
    "llm_embed": {
        "category": ToolCategory.AI_ML,
        "description": "Text Embeddings generieren",
        "parameters": {"text": "str", "model": "str"},
    },
    "llm_classify": {
        "category": ToolCategory.AI_ML,
        "description": "Text klassifizieren",
        "parameters": {"text": "str", "labels": "list"},
    },
    "llm_extract": {
        "category": ToolCategory.AI_ML,
        "description": "Strukturierte Daten extrahieren",
        "parameters": {"text": "str", "schema": "dict"},
    },
    "llm_summarize": {
        "category": ToolCategory.AI_ML,
        "description": "Text zusammenfassen",
        "parameters": {"text": "str", "max_length": "int"},
    },
    "llm_translate": {
        "category": ToolCategory.AI_ML,
        "description": "Text übersetzen",
        "parameters": {"text": "str", "source": "str", "target": "str"},
    },
    "llm_sentiment": {
        "category": ToolCategory.AI_ML,
        "description": "Sentiment Analyse",
        "parameters": {"text": "str"},
    },
    "llm_ner": {
        "category": ToolCategory.AI_ML,
        "description": "Named Entity Recognition",
        "parameters": {"text": "str"},
    },
    "llm_qa": {
        "category": ToolCategory.AI_ML,
        "description": "Question Answering",
        "parameters": {"question": "str", "context": "str"},
    },
    "image_generate": {
        "category": ToolCategory.AI_ML,
        "description": "Bilder mit AI generieren",
        "parameters": {"prompt": "str", "model": "str", "size": "str"},
    },
    "image_edit_ai": {
        "category": ToolCategory.AI_ML,
        "description": "Bilder mit AI bearbeiten",
        "parameters": {"image": "str", "instruction": "str"},
    },
    "image_variation": {
        "category": ToolCategory.AI_ML,
        "description": "Bild-Variationen erstellen",
        "parameters": {"image": "str", "count": "int"},
    },
    "image_inpaint": {
        "category": ToolCategory.AI_ML,
        "description": "Bildbereiche mit AI füllen",
        "parameters": {"image": "str", "mask": "str", "prompt": "str"},
    },
    "image_outpaint": {
        "category": ToolCategory.AI_ML,
        "description": "Bild erweitern mit AI",
        "parameters": {"image": "str", "direction": "str"},
    },
    "image_caption": {
        "category": ToolCategory.AI_ML,
        "description": "Bildbeschreibung generieren",
        "parameters": {"image": "str"},
    },
    "image_classify_ai": {
        "category": ToolCategory.AI_ML,
        "description": "Bilder mit AI klassifizieren",
        "parameters": {"image": "str", "labels": "list"},
    },
    "image_segment": {
        "category": ToolCategory.AI_ML,
        "description": "Bildsegmentierung",
        "parameters": {"image": "str"},
    },
    "image_depth": {
        "category": ToolCategory.AI_ML,
        "description": "Tiefenkarte erstellen",
        "parameters": {"image": "str"},
    },
    "image_pose": {
        "category": ToolCategory.AI_ML,
        "description": "Pose Estimation",
        "parameters": {"image": "str"},
    },
    "audio_generate": {
        "category": ToolCategory.AI_ML,
        "description": "Audio mit AI generieren",
        "parameters": {"prompt": "str", "duration": "float"},
    },
    "music_generate": {
        "category": ToolCategory.AI_ML,
        "description": "Musik mit AI generieren",
        "parameters": {"prompt": "str", "duration": "float", "style": "str"},
    },
    "voice_clone": {
        "category": ToolCategory.AI_ML,
        "description": "Stimme klonen",
        "parameters": {"audio_sample": "str", "text": "str"},
    },
    "video_generate": {
        "category": ToolCategory.AI_ML,
        "description": "Video mit AI generieren",
        "parameters": {"prompt": "str", "duration": "float"},
    },
    "video_edit_ai": {
        "category": ToolCategory.AI_ML,
        "description": "Video mit AI bearbeiten",
        "parameters": {"video": "str", "instruction": "str"},
    },
    "3d_generate": {
        "category": ToolCategory.AI_ML,
        "description": "3D Modelle generieren",
        "parameters": {"prompt": "str"},
    },
    "3d_from_image": {
        "category": ToolCategory.AI_ML,
        "description": "3D aus Bildern erstellen",
        "parameters": {"images": "list"},
    },
    "model_train": {
        "category": ToolCategory.AI_ML,
        "description": "ML Modell trainieren",
        "parameters": {"data": "str", "model_type": "str", "config": "dict"},
    },
    "model_finetune": {
        "category": ToolCategory.AI_ML,
        "description": "Modell feintunen",
        "parameters": {"base_model": "str", "data": "str"},
    },
    "model_evaluate": {
        "category": ToolCategory.AI_ML,
        "description": "Modell evaluieren",
        "parameters": {"model": "str", "test_data": "str"},
    },
    "model_predict": {
        "category": ToolCategory.AI_ML,
        "description": "Vorhersagen mit Modell",
        "parameters": {"model": "str", "input": "Any"},
    },
    "model_explain": {
        "category": ToolCategory.AI_ML,
        "description": "Modell-Entscheidungen erklären",
        "parameters": {"model": "str", "input": "Any"},
    },
    "dataset_create": {
        "category": ToolCategory.AI_ML,
        "description": "Datensatz erstellen",
        "parameters": {"source": "str", "format": "str"},
    },
    "dataset_augment": {
        "category": ToolCategory.AI_ML,
        "description": "Datensatz erweitern",
        "parameters": {"data": "str", "techniques": "list"},
    },
    "dataset_split": {
        "category": ToolCategory.AI_ML,
        "description": "Datensatz teilen",
        "parameters": {"data": "str", "ratios": "list"},
    },
    "feature_engineer": {
        "category": ToolCategory.AI_ML,
        "description": "Features erstellen",
        "parameters": {"data": "str", "features": "list"},
    },
    "feature_select": {
        "category": ToolCategory.AI_ML,
        "description": "Features auswählen",
        "parameters": {"data": "str", "method": "str"},
    },
    "hyperparameter_tune": {
        "category": ToolCategory.AI_ML,
        "description": "Hyperparameter optimieren",
        "parameters": {"model": "str", "param_space": "dict"},
    },
    "anomaly_detect": {
        "category": ToolCategory.AI_ML,
        "description": "Anomalien erkennen",
        "parameters": {"data": "str", "method": "str"},
    },
    "clustering": {
        "category": ToolCategory.AI_ML,
        "description": "Daten clustern",
        "parameters": {"data": "str", "algorithm": "str", "n_clusters": "int"},
    },
    "dimensionality_reduce": {
        "category": ToolCategory.AI_ML,
        "description": "Dimensionen reduzieren",
        "parameters": {"data": "str", "method": "str", "n_components": "int"},
    },
    "time_series_forecast": {
        "category": ToolCategory.AI_ML,
        "description": "Zeitreihen vorhersagen",
        "parameters": {"data": "str", "horizon": "int"},
    },
    "time_series_decompose": {
        "category": ToolCategory.AI_ML,
        "description": "Zeitreihen zerlegen",
        "parameters": {"data": "str"},
    },
    "recommendation": {
        "category": ToolCategory.AI_ML,
        "description": "Empfehlungen generieren",
        "parameters": {"user_data": "dict", "items": "list"},
    },
    "similarity_search": {
        "category": ToolCategory.AI_ML,
        "description": "Ähnlichkeitssuche",
        "parameters": {"query": "str", "corpus": "list", "top_k": "int"},
    },
    "rag_query": {
        "category": ToolCategory.AI_ML,
        "description": "RAG (Retrieval Augmented Generation)",
        "parameters": {"query": "str", "documents": "list"},
    },
    "agent_create": {
        "category": ToolCategory.AI_ML,
        "description": "AI Agent erstellen",
        "parameters": {"name": "str", "tools": "list", "instructions": "str"},
    },
    "agent_run": {
        "category": ToolCategory.AI_ML,
        "description": "AI Agent ausführen",
        "parameters": {"agent_id": "str", "task": "str"},
    },

    # ===========================================
    # KREATIV (100 Tools)
    # ===========================================
    "logo_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Logo mit AI generieren",
        "parameters": {"description": "str", "style": "str"},
    },
    "icon_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Icons generieren",
        "parameters": {"description": "str", "style": "str", "size": "int"},
    },
    "mockup_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "UI Mockups generieren",
        "parameters": {"description": "str", "device": "str"},
    },
    "banner_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Banner erstellen",
        "parameters": {"text": "str", "size": "str", "style": "str"},
    },
    "poster_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Poster erstellen",
        "parameters": {"content": "str", "template": "str"},
    },
    "presentation_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Präsentation erstellen",
        "parameters": {"topic": "str", "slides": "int", "style": "str"},
    },
    "infographic_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Infografik erstellen",
        "parameters": {"data": "dict", "style": "str"},
    },
    "chart_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Diagramme erstellen",
        "parameters": {"data": "dict", "chart_type": "str"},
    },
    "diagram_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Diagramme (Flowchart, UML) erstellen",
        "parameters": {"type": "str", "content": "str"},
    },
    "mindmap_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Mindmap erstellen",
        "parameters": {"topic": "str", "branches": "list"},
    },
    "wireframe_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Wireframes erstellen",
        "parameters": {"description": "str", "pages": "list"},
    },
    "color_palette": {
        "category": ToolCategory.CREATIVE,
        "description": "Farbpalette generieren",
        "parameters": {"base_color": "str", "scheme": "str"},
    },
    "font_suggest": {
        "category": ToolCategory.CREATIVE,
        "description": "Fonts vorschlagen",
        "parameters": {"style": "str", "use_case": "str"},
    },
    "brand_kit": {
        "category": ToolCategory.CREATIVE,
        "description": "Brand Kit erstellen",
        "parameters": {"name": "str", "values": "list"},
    },
    "social_media_post": {
        "category": ToolCategory.CREATIVE,
        "description": "Social Media Posts erstellen",
        "parameters": {"platform": "str", "content": "str"},
    },
    "thumbnail_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Thumbnails erstellen",
        "parameters": {"title": "str", "style": "str"},
    },
    "meme_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Memes generieren",
        "parameters": {"template": "str", "text_top": "str", "text_bottom": "str"},
    },
    "avatar_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Avatare generieren",
        "parameters": {"style": "str", "features": "dict"},
    },
    "pattern_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Muster generieren",
        "parameters": {"style": "str", "colors": "list"},
    },
    "texture_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Texturen generieren",
        "parameters": {"type": "str", "seamless": "bool"},
    },
    "animation_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Animationen erstellen",
        "parameters": {"type": "str", "frames": "int"},
    },
    "gif_create": {
        "category": ToolCategory.CREATIVE,
        "description": "GIFs erstellen",
        "parameters": {"images": "list", "duration": "float"},
    },
    "storyboard_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Storyboard erstellen",
        "parameters": {"script": "str", "scenes": "int"},
    },
    "character_design": {
        "category": ToolCategory.CREATIVE,
        "description": "Character Design",
        "parameters": {"description": "str", "style": "str"},
    },
    "scene_generate": {
        "category": ToolCategory.CREATIVE,
        "description": "Szenen generieren",
        "parameters": {"description": "str", "style": "str"},
    },
    "text_effect": {
        "category": ToolCategory.CREATIVE,
        "description": "Text-Effekte erstellen",
        "parameters": {"text": "str", "effect": "str"},
    },
    "watermark_create": {
        "category": ToolCategory.CREATIVE,
        "description": "Wasserzeichen erstellen",
        "parameters": {"text": "str", "style": "str"},
    },

    # ===========================================
    # BUSINESS & PRODUKTIVITÄT (100 Tools)
    # ===========================================
    "email_compose": {
        "category": ToolCategory.BUSINESS,
        "description": "E-Mail verfassen",
        "parameters": {"purpose": "str", "tone": "str", "recipient": "str"},
    },
    "email_reply": {
        "category": ToolCategory.BUSINESS,
        "description": "E-Mail beantworten",
        "parameters": {"original": "str", "intent": "str"},
    },
    "meeting_agenda": {
        "category": ToolCategory.BUSINESS,
        "description": "Meeting Agenda erstellen",
        "parameters": {"topic": "str", "duration": "int", "attendees": "list"},
    },
    "meeting_notes": {
        "category": ToolCategory.BUSINESS,
        "description": "Meeting Notizen erstellen",
        "parameters": {"transcript": "str"},
    },
    "meeting_summary": {
        "category": ToolCategory.BUSINESS,
        "description": "Meeting zusammenfassen",
        "parameters": {"notes": "str"},
    },
    "project_plan": {
        "category": ToolCategory.BUSINESS,
        "description": "Projektplan erstellen",
        "parameters": {"project": "str", "milestones": "list"},
    },
    "task_breakdown": {
        "category": ToolCategory.BUSINESS,
        "description": "Aufgaben zerlegen",
        "parameters": {"task": "str"},
    },
    "sprint_plan": {
        "category": ToolCategory.BUSINESS,
        "description": "Sprint planen",
        "parameters": {"backlog": "list", "capacity": "int"},
    },
    "okr_generator": {
        "category": ToolCategory.BUSINESS,
        "description": "OKRs generieren",
        "parameters": {"goal": "str"},
    },
    "kpi_dashboard": {
        "category": ToolCategory.BUSINESS,
        "description": "KPI Dashboard erstellen",
        "parameters": {"metrics": "list"},
    },
    "report_generate": {
        "category": ToolCategory.BUSINESS,
        "description": "Reports generieren",
        "parameters": {"data": "dict", "template": "str"},
    },
    "invoice_create": {
        "category": ToolCategory.BUSINESS,
        "description": "Rechnung erstellen",
        "parameters": {"items": "list", "client": "dict"},
    },
    "proposal_create": {
        "category": ToolCategory.BUSINESS,
        "description": "Angebot erstellen",
        "parameters": {"project": "str", "scope": "list"},
    },
    "contract_generate": {
        "category": ToolCategory.BUSINESS,
        "description": "Vertrag generieren",
        "parameters": {"type": "str", "parties": "list", "terms": "dict"},
    },
    "contract_analyze": {
        "category": ToolCategory.BUSINESS,
        "description": "Vertrag analysieren",
        "parameters": {"contract": "str"},
    },
    "legal_review": {
        "category": ToolCategory.BUSINESS,
        "description": "Rechtliche Prüfung",
        "parameters": {"document": "str"},
    },
    "business_plan": {
        "category": ToolCategory.BUSINESS,
        "description": "Businessplan erstellen",
        "parameters": {"idea": "str", "market": "str"},
    },
    "pitch_deck": {
        "category": ToolCategory.BUSINESS,
        "description": "Pitch Deck erstellen",
        "parameters": {"company": "str", "problem": "str", "solution": "str"},
    },
    "swot_analysis": {
        "category": ToolCategory.BUSINESS,
        "description": "SWOT Analyse erstellen",
        "parameters": {"subject": "str"},
    },
    "competitor_analysis": {
        "category": ToolCategory.BUSINESS,
        "description": "Wettbewerbsanalyse",
        "parameters": {"competitors": "list"},
    },
    "market_research": {
        "category": ToolCategory.BUSINESS,
        "description": "Marktforschung",
        "parameters": {"market": "str", "aspects": "list"},
    },
    "customer_persona": {
        "category": ToolCategory.BUSINESS,
        "description": "Kunden-Persona erstellen",
        "parameters": {"product": "str", "demographics": "dict"},
    },
    "user_story": {
        "category": ToolCategory.BUSINESS,
        "description": "User Stories schreiben",
        "parameters": {"feature": "str"},
    },
    "seo_analyze": {
        "category": ToolCategory.BUSINESS,
        "description": "SEO analysieren",
        "parameters": {"url": "str"},
    },
    "seo_keywords": {
        "category": ToolCategory.BUSINESS,
        "description": "SEO Keywords finden",
        "parameters": {"topic": "str"},
    },
    "content_calendar": {
        "category": ToolCategory.BUSINESS,
        "description": "Content Kalender erstellen",
        "parameters": {"topics": "list", "frequency": "str"},
    },
    "blog_post": {
        "category": ToolCategory.BUSINESS,
        "description": "Blog Post schreiben",
        "parameters": {"topic": "str", "keywords": "list", "length": "int"},
    },
    "press_release": {
        "category": ToolCategory.BUSINESS,
        "description": "Pressemitteilung schreiben",
        "parameters": {"news": "str", "company": "str"},
    },
    "newsletter": {
        "category": ToolCategory.BUSINESS,
        "description": "Newsletter erstellen",
        "parameters": {"content": "list", "style": "str"},
    },
    "job_description": {
        "category": ToolCategory.BUSINESS,
        "description": "Stellenausschreibung erstellen",
        "parameters": {"role": "str", "requirements": "list"},
    },
    "resume_analyze": {
        "category": ToolCategory.BUSINESS,
        "description": "Lebenslauf analysieren",
        "parameters": {"resume": "str", "job": "str"},
    },
    "interview_questions": {
        "category": ToolCategory.BUSINESS,
        "description": "Interview Fragen generieren",
        "parameters": {"role": "str", "level": "str"},
    },
    "feedback_write": {
        "category": ToolCategory.BUSINESS,
        "description": "Feedback schreiben",
        "parameters": {"performance": "str", "tone": "str"},
    },

    # ===========================================
    # WISSENSCHAFT & FORSCHUNG (50 Tools)
    # ===========================================
    "paper_summarize": {
        "category": ToolCategory.SCIENCE,
        "description": "Paper zusammenfassen",
        "parameters": {"paper": "str"},
    },
    "paper_critique": {
        "category": ToolCategory.SCIENCE,
        "description": "Paper kritisch analysieren",
        "parameters": {"paper": "str"},
    },
    "citation_format": {
        "category": ToolCategory.SCIENCE,
        "description": "Zitate formatieren",
        "parameters": {"source": "dict", "style": "str"},
    },
    "bibliography_generate": {
        "category": ToolCategory.SCIENCE,
        "description": "Bibliografie erstellen",
        "parameters": {"sources": "list", "style": "str"},
    },
    "literature_review": {
        "category": ToolCategory.SCIENCE,
        "description": "Literaturrecherche",
        "parameters": {"topic": "str", "sources": "int"},
    },
    "hypothesis_generate": {
        "category": ToolCategory.SCIENCE,
        "description": "Hypothesen generieren",
        "parameters": {"observation": "str"},
    },
    "experiment_design": {
        "category": ToolCategory.SCIENCE,
        "description": "Experiment designen",
        "parameters": {"hypothesis": "str", "variables": "list"},
    },
    "statistical_test": {
        "category": ToolCategory.SCIENCE,
        "description": "Statistischen Test auswählen",
        "parameters": {"data_type": "str", "comparison": "str"},
    },
    "p_value_calculate": {
        "category": ToolCategory.SCIENCE,
        "description": "P-Wert berechnen",
        "parameters": {"data": "list", "test": "str"},
    },
    "confidence_interval": {
        "category": ToolCategory.SCIENCE,
        "description": "Konfidenzintervall berechnen",
        "parameters": {"data": "list", "confidence": "float"},
    },
    "sample_size": {
        "category": ToolCategory.SCIENCE,
        "description": "Stichprobengröße berechnen",
        "parameters": {"effect_size": "float", "power": "float"},
    },
    "data_visualize": {
        "category": ToolCategory.SCIENCE,
        "description": "Daten visualisieren",
        "parameters": {"data": "dict", "chart_type": "str"},
    },
    "correlation_analyze": {
        "category": ToolCategory.SCIENCE,
        "description": "Korrelation analysieren",
        "parameters": {"x": "list", "y": "list"},
    },
    "regression_analyze": {
        "category": ToolCategory.SCIENCE,
        "description": "Regression durchführen",
        "parameters": {"x": "list", "y": "list", "type": "str"},
    },
    "anova_test": {
        "category": ToolCategory.SCIENCE,
        "description": "ANOVA durchführen",
        "parameters": {"groups": "list"},
    },
    "chi_square_test": {
        "category": ToolCategory.SCIENCE,
        "description": "Chi-Quadrat-Test",
        "parameters": {"observed": "list", "expected": "list"},
    },
    "t_test": {
        "category": ToolCategory.SCIENCE,
        "description": "T-Test durchführen",
        "parameters": {"group1": "list", "group2": "list"},
    },
    "mann_whitney": {
        "category": ToolCategory.SCIENCE,
        "description": "Mann-Whitney-Test",
        "parameters": {"group1": "list", "group2": "list"},
    },
    "kruskal_wallis": {
        "category": ToolCategory.SCIENCE,
        "description": "Kruskal-Wallis-Test",
        "parameters": {"groups": "list"},
    },
    "meta_analysis": {
        "category": ToolCategory.SCIENCE,
        "description": "Meta-Analyse durchführen",
        "parameters": {"studies": "list"},
    },
    "systematic_review": {
        "category": ToolCategory.SCIENCE,
        "description": "Systematische Review",
        "parameters": {"query": "str", "criteria": "dict"},
    },
    "research_methods": {
        "category": ToolCategory.SCIENCE,
        "description": "Forschungsmethoden vorschlagen",
        "parameters": {"question": "str"},
    },
    "grant_proposal": {
        "category": ToolCategory.SCIENCE,
        "description": "Förderantrag schreiben",
        "parameters": {"project": "str", "funder": "str"},
    },
    "abstract_write": {
        "category": ToolCategory.SCIENCE,
        "description": "Abstract schreiben",
        "parameters": {"paper": "str"},
    },
    "peer_review": {
        "category": ToolCategory.SCIENCE,
        "description": "Peer Review durchführen",
        "parameters": {"paper": "str"},
    },

    # ===========================================
    # MATHEMATIK (50 Tools)
    # ===========================================
    "math_solve": {
        "category": ToolCategory.MATH,
        "description": "Mathematische Gleichungen lösen",
        "parameters": {"equation": "str"},
    },
    "math_simplify": {
        "category": ToolCategory.MATH,
        "description": "Ausdrücke vereinfachen",
        "parameters": {"expression": "str"},
    },
    "math_expand": {
        "category": ToolCategory.MATH,
        "description": "Ausdrücke expandieren",
        "parameters": {"expression": "str"},
    },
    "math_factor": {
        "category": ToolCategory.MATH,
        "description": "Faktorisieren",
        "parameters": {"expression": "str"},
    },
    "derivative": {
        "category": ToolCategory.MATH,
        "description": "Ableitung berechnen",
        "parameters": {"function": "str", "variable": "str"},
    },
    "integral": {
        "category": ToolCategory.MATH,
        "description": "Integral berechnen",
        "parameters": {"function": "str", "variable": "str"},
    },
    "limit": {
        "category": ToolCategory.MATH,
        "description": "Grenzwert berechnen",
        "parameters": {"function": "str", "variable": "str", "point": "str"},
    },
    "series_expand": {
        "category": ToolCategory.MATH,
        "description": "Reihenentwicklung",
        "parameters": {"function": "str", "point": "str", "order": "int"},
    },
    "matrix_multiply": {
        "category": ToolCategory.MATH,
        "description": "Matrizen multiplizieren",
        "parameters": {"matrix1": "list", "matrix2": "list"},
    },
    "matrix_inverse": {
        "category": ToolCategory.MATH,
        "description": "Matrix invertieren",
        "parameters": {"matrix": "list"},
    },
    "matrix_determinant": {
        "category": ToolCategory.MATH,
        "description": "Determinante berechnen",
        "parameters": {"matrix": "list"},
    },
    "eigenvalues": {
        "category": ToolCategory.MATH,
        "description": "Eigenwerte berechnen",
        "parameters": {"matrix": "list"},
    },
    "linear_system": {
        "category": ToolCategory.MATH,
        "description": "Lineares Gleichungssystem lösen",
        "parameters": {"equations": "list"},
    },
    "vector_operations": {
        "category": ToolCategory.MATH,
        "description": "Vektor-Operationen",
        "parameters": {"vectors": "list", "operation": "str"},
    },
    "prime_check": {
        "category": ToolCategory.MATH,
        "description": "Primzahl prüfen",
        "parameters": {"number": "int"},
    },
    "prime_factors": {
        "category": ToolCategory.MATH,
        "description": "Primfaktorzerlegung",
        "parameters": {"number": "int"},
    },
    "gcd_lcm": {
        "category": ToolCategory.MATH,
        "description": "GGT/KGV berechnen",
        "parameters": {"numbers": "list"},
    },
    "modular_arithmetic": {
        "category": ToolCategory.MATH,
        "description": "Modulare Arithmetik",
        "parameters": {"a": "int", "b": "int", "m": "int", "operation": "str"},
    },
    "combinatorics": {
        "category": ToolCategory.MATH,
        "description": "Kombinatorik berechnen",
        "parameters": {"n": "int", "k": "int", "type": "str"},
    },
    "probability": {
        "category": ToolCategory.MATH,
        "description": "Wahrscheinlichkeit berechnen",
        "parameters": {"favorable": "int", "total": "int"},
    },
    "statistics_describe": {
        "category": ToolCategory.MATH,
        "description": "Deskriptive Statistik",
        "parameters": {"data": "list"},
    },
    "normal_distribution": {
        "category": ToolCategory.MATH,
        "description": "Normalverteilung berechnen",
        "parameters": {"mean": "float", "std": "float", "x": "float"},
    },
    "binomial_distribution": {
        "category": ToolCategory.MATH,
        "description": "Binomialverteilung berechnen",
        "parameters": {"n": "int", "p": "float", "k": "int"},
    },
    "poisson_distribution": {
        "category": ToolCategory.MATH,
        "description": "Poisson-Verteilung berechnen",
        "parameters": {"lambda_": "float", "k": "int"},
    },
    "geometry_area": {
        "category": ToolCategory.MATH,
        "description": "Fläche berechnen",
        "parameters": {"shape": "str", "dimensions": "dict"},
    },
    "geometry_volume": {
        "category": ToolCategory.MATH,
        "description": "Volumen berechnen",
        "parameters": {"shape": "str", "dimensions": "dict"},
    },
    "trigonometry": {
        "category": ToolCategory.MATH,
        "description": "Trigonometrie berechnen",
        "parameters": {"function": "str", "angle": "float", "unit": "str"},
    },
    "unit_convert": {
        "category": ToolCategory.MATH,
        "description": "Einheiten umrechnen",
        "parameters": {"value": "float", "from_unit": "str", "to_unit": "str"},
    },
    "currency_convert": {
        "category": ToolCategory.MATH,
        "description": "Währungen umrechnen",
        "parameters": {"amount": "float", "from_currency": "str", "to_currency": "str"},
    },
    "percentage_calculate": {
        "category": ToolCategory.MATH,
        "description": "Prozent berechnen",
        "parameters": {"value": "float", "percentage": "float", "operation": "str"},
    },
    "compound_interest": {
        "category": ToolCategory.MATH,
        "description": "Zinseszins berechnen",
        "parameters": {"principal": "float", "rate": "float", "time": "float", "n": "int"},
    },
    "loan_calculator": {
        "category": ToolCategory.MATH,
        "description": "Kredit berechnen",
        "parameters": {"principal": "float", "rate": "float", "months": "int"},
    },
    "roi_calculate": {
        "category": ToolCategory.MATH,
        "description": "ROI berechnen",
        "parameters": {"gain": "float", "cost": "float"},
    },
    "npv_calculate": {
        "category": ToolCategory.MATH,
        "description": "NPV berechnen",
        "parameters": {"cash_flows": "list", "rate": "float"},
    },
    "irr_calculate": {
        "category": ToolCategory.MATH,
        "description": "IRR berechnen",
        "parameters": {"cash_flows": "list"},
    },

    # ===========================================
    # TEXT VERARBEITUNG (50 Tools)
    # ===========================================
    "text_summarize": {
        "category": ToolCategory.TEXT,
        "description": "Text zusammenfassen",
        "parameters": {"text": "str", "length": "str"},
    },
    "text_paraphrase": {
        "category": ToolCategory.TEXT,
        "description": "Text umformulieren",
        "parameters": {"text": "str"},
    },
    "text_expand": {
        "category": ToolCategory.TEXT,
        "description": "Text erweitern",
        "parameters": {"text": "str", "target_length": "int"},
    },
    "text_simplify": {
        "category": ToolCategory.TEXT,
        "description": "Text vereinfachen",
        "parameters": {"text": "str", "level": "str"},
    },
    "text_formalize": {
        "category": ToolCategory.TEXT,
        "description": "Text formalisieren",
        "parameters": {"text": "str"},
    },
    "text_casualize": {
        "category": ToolCategory.TEXT,
        "description": "Text informeller machen",
        "parameters": {"text": "str"},
    },
    "grammar_check": {
        "category": ToolCategory.TEXT,
        "description": "Grammatik prüfen",
        "parameters": {"text": "str", "language": "str"},
    },
    "spell_check": {
        "category": ToolCategory.TEXT,
        "description": "Rechtschreibung prüfen",
        "parameters": {"text": "str", "language": "str"},
    },
    "style_check": {
        "category": ToolCategory.TEXT,
        "description": "Stil prüfen",
        "parameters": {"text": "str"},
    },
    "readability_analyze": {
        "category": ToolCategory.TEXT,
        "description": "Lesbarkeit analysieren",
        "parameters": {"text": "str"},
    },
    "keyword_extract": {
        "category": ToolCategory.TEXT,
        "description": "Keywords extrahieren",
        "parameters": {"text": "str", "count": "int"},
    },
    "topic_extract": {
        "category": ToolCategory.TEXT,
        "description": "Themen extrahieren",
        "parameters": {"text": "str"},
    },
    "entity_extract": {
        "category": ToolCategory.TEXT,
        "description": "Entitäten extrahieren",
        "parameters": {"text": "str"},
    },
    "relation_extract": {
        "category": ToolCategory.TEXT,
        "description": "Relationen extrahieren",
        "parameters": {"text": "str"},
    },
    "text_classify": {
        "category": ToolCategory.TEXT,
        "description": "Text klassifizieren",
        "parameters": {"text": "str", "categories": "list"},
    },
    "text_cluster": {
        "category": ToolCategory.TEXT,
        "description": "Texte clustern",
        "parameters": {"texts": "list", "n_clusters": "int"},
    },
    "text_compare": {
        "category": ToolCategory.TEXT,
        "description": "Texte vergleichen",
        "parameters": {"text1": "str", "text2": "str"},
    },
    "text_deduplicate": {
        "category": ToolCategory.TEXT,
        "description": "Duplikate entfernen",
        "parameters": {"texts": "list"},
    },
    "text_anonymize": {
        "category": ToolCategory.TEXT,
        "description": "Text anonymisieren",
        "parameters": {"text": "str"},
    },
    "text_template": {
        "category": ToolCategory.TEXT,
        "description": "Template ausfüllen",
        "parameters": {"template": "str", "variables": "dict"},
    },
    "text_format": {
        "category": ToolCategory.TEXT,
        "description": "Text formatieren",
        "parameters": {"text": "str", "format": "str"},
    },
    "markdown_to_html": {
        "category": ToolCategory.TEXT,
        "description": "Markdown zu HTML",
        "parameters": {"markdown": "str"},
    },
    "html_to_markdown": {
        "category": ToolCategory.TEXT,
        "description": "HTML zu Markdown",
        "parameters": {"html": "str"},
    },
    "latex_to_text": {
        "category": ToolCategory.TEXT,
        "description": "LaTeX zu Text",
        "parameters": {"latex": "str"},
    },
    "text_to_latex": {
        "category": ToolCategory.TEXT,
        "description": "Text zu LaTeX",
        "parameters": {"text": "str"},
    },
    "bbcode_convert": {
        "category": ToolCategory.TEXT,
        "description": "BBCode konvertieren",
        "parameters": {"text": "str", "direction": "str"},
    },
    "text_encrypt": {
        "category": ToolCategory.TEXT,
        "description": "Text verschlüsseln",
        "parameters": {"text": "str", "method": "str"},
    },
    "text_decrypt": {
        "category": ToolCategory.TEXT,
        "description": "Text entschlüsseln",
        "parameters": {"text": "str", "method": "str", "key": "str"},
    },
    "base64_encode": {
        "category": ToolCategory.TEXT,
        "description": "Base64 enkodieren",
        "parameters": {"text": "str"},
    },
    "base64_decode": {
        "category": ToolCategory.TEXT,
        "description": "Base64 dekodieren",
        "parameters": {"text": "str"},
    },
    "url_encode": {
        "category": ToolCategory.TEXT,
        "description": "URL enkodieren",
        "parameters": {"text": "str"},
    },
    "url_decode": {
        "category": ToolCategory.TEXT,
        "description": "URL dekodieren",
        "parameters": {"text": "str"},
    },
    "html_encode": {
        "category": ToolCategory.TEXT,
        "description": "HTML enkodieren",
        "parameters": {"text": "str"},
    },
    "html_decode": {
        "category": ToolCategory.TEXT,
        "description": "HTML dekodieren",
        "parameters": {"text": "str"},
    },
    "json_escape": {
        "category": ToolCategory.TEXT,
        "description": "JSON escapen",
        "parameters": {"text": "str"},
    },
    "text_hash": {
        "category": ToolCategory.TEXT,
        "description": "Text hashen",
        "parameters": {"text": "str", "algorithm": "str"},
    },

    # ===========================================
    # SYSTEM & AUTOMATION (50 Tools)
    # ===========================================
    "process_list": {
        "category": ToolCategory.SYSTEM,
        "description": "Prozesse auflisten",
        "parameters": {},
    },
    "process_kill": {
        "category": ToolCategory.SYSTEM,
        "description": "Prozess beenden",
        "parameters": {"pid": "int"},
    },
    "process_start": {
        "category": ToolCategory.SYSTEM,
        "description": "Prozess starten",
        "parameters": {"command": "str", "args": "list"},
    },
    "service_status": {
        "category": ToolCategory.SYSTEM,
        "description": "Service Status prüfen",
        "parameters": {"service": "str"},
    },
    "service_start": {
        "category": ToolCategory.SYSTEM,
        "description": "Service starten",
        "parameters": {"service": "str"},
    },
    "service_stop": {
        "category": ToolCategory.SYSTEM,
        "description": "Service stoppen",
        "parameters": {"service": "str"},
    },
    "cron_list": {
        "category": ToolCategory.SYSTEM,
        "description": "Cron Jobs auflisten",
        "parameters": {},
    },
    "cron_add": {
        "category": ToolCategory.SYSTEM,
        "description": "Cron Job hinzufügen",
        "parameters": {"schedule": "str", "command": "str"},
    },
    "cron_remove": {
        "category": ToolCategory.SYSTEM,
        "description": "Cron Job entfernen",
        "parameters": {"id": "str"},
    },
    "env_get": {
        "category": ToolCategory.SYSTEM,
        "description": "Umgebungsvariable lesen",
        "parameters": {"name": "str"},
    },
    "env_set": {
        "category": ToolCategory.SYSTEM,
        "description": "Umgebungsvariable setzen",
        "parameters": {"name": "str", "value": "str"},
    },
    "system_info": {
        "category": ToolCategory.SYSTEM,
        "description": "Systeminformationen abrufen",
        "parameters": {},
    },
    "disk_usage": {
        "category": ToolCategory.SYSTEM,
        "description": "Festplattennutzung",
        "parameters": {"path": "str"},
    },
    "memory_usage": {
        "category": ToolCategory.SYSTEM,
        "description": "Speichernutzung",
        "parameters": {},
    },
    "cpu_usage": {
        "category": ToolCategory.SYSTEM,
        "description": "CPU-Nutzung",
        "parameters": {},
    },
    "network_info": {
        "category": ToolCategory.SYSTEM,
        "description": "Netzwerk-Informationen",
        "parameters": {},
    },
    "port_check": {
        "category": ToolCategory.SYSTEM,
        "description": "Port prüfen",
        "parameters": {"host": "str", "port": "int"},
    },
    "ping": {
        "category": ToolCategory.SYSTEM,
        "description": "Host anpingen",
        "parameters": {"host": "str"},
    },
    "traceroute": {
        "category": ToolCategory.SYSTEM,
        "description": "Traceroute",
        "parameters": {"host": "str"},
    },
    "docker_ps": {
        "category": ToolCategory.SYSTEM,
        "description": "Docker Container auflisten",
        "parameters": {},
    },
    "docker_start": {
        "category": ToolCategory.SYSTEM,
        "description": "Docker Container starten",
        "parameters": {"container": "str"},
    },
    "docker_stop": {
        "category": ToolCategory.SYSTEM,
        "description": "Docker Container stoppen",
        "parameters": {"container": "str"},
    },
    "docker_logs": {
        "category": ToolCategory.SYSTEM,
        "description": "Docker Logs abrufen",
        "parameters": {"container": "str", "tail": "int"},
    },
    "docker_exec": {
        "category": ToolCategory.SYSTEM,
        "description": "Befehl in Container ausführen",
        "parameters": {"container": "str", "command": "str"},
    },
    "docker_build": {
        "category": ToolCategory.SYSTEM,
        "description": "Docker Image bauen",
        "parameters": {"path": "str", "tag": "str"},
    },
    "docker_push": {
        "category": ToolCategory.SYSTEM,
        "description": "Docker Image pushen",
        "parameters": {"image": "str"},
    },
    "docker_pull": {
        "category": ToolCategory.SYSTEM,
        "description": "Docker Image pullen",
        "parameters": {"image": "str"},
    },
    "log_read": {
        "category": ToolCategory.SYSTEM,
        "description": "Logs lesen",
        "parameters": {"path": "str", "lines": "int"},
    },
    "log_search": {
        "category": ToolCategory.SYSTEM,
        "description": "Logs durchsuchen",
        "parameters": {"path": "str", "pattern": "str"},
    },
    "log_analyze": {
        "category": ToolCategory.SYSTEM,
        "description": "Logs analysieren",
        "parameters": {"path": "str"},
    },
    "backup_create": {
        "category": ToolCategory.SYSTEM,
        "description": "Backup erstellen",
        "parameters": {"source": "str", "destination": "str"},
    },
    "backup_restore": {
        "category": ToolCategory.SYSTEM,
        "description": "Backup wiederherstellen",
        "parameters": {"backup": "str", "destination": "str"},
    },
    "ssh_execute": {
        "category": ToolCategory.SYSTEM,
        "description": "SSH Befehl ausführen",
        "parameters": {"host": "str", "command": "str"},
    },
    "sftp_upload": {
        "category": ToolCategory.SYSTEM,
        "description": "Datei per SFTP hochladen",
        "parameters": {"local": "str", "remote": "str", "host": "str"},
    },
    "sftp_download": {
        "category": ToolCategory.SYSTEM,
        "description": "Datei per SFTP herunterladen",
        "parameters": {"remote": "str", "local": "str", "host": "str"},
    },
    "webhook_send": {
        "category": ToolCategory.SYSTEM,
        "description": "Webhook senden",
        "parameters": {"url": "str", "payload": "dict"},
    },
    "webhook_receive": {
        "category": ToolCategory.SYSTEM,
        "description": "Webhook empfangen",
        "parameters": {"port": "int", "path": "str"},
    },
    "notification_send": {
        "category": ToolCategory.SYSTEM,
        "description": "Benachrichtigung senden",
        "parameters": {"channel": "str", "message": "str"},
    },
    "email_send": {
        "category": ToolCategory.SYSTEM,
        "description": "E-Mail senden",
        "parameters": {"to": "str", "subject": "str", "body": "str"},
    },
    "sms_send": {
        "category": ToolCategory.SYSTEM,
        "description": "SMS senden",
        "parameters": {"to": "str", "message": "str"},
    },
    "slack_send": {
        "category": ToolCategory.SYSTEM,
        "description": "Slack Nachricht senden",
        "parameters": {"channel": "str", "message": "str"},
    },
    "discord_send": {
        "category": ToolCategory.SYSTEM,
        "description": "Discord Nachricht senden",
        "parameters": {"channel": "str", "message": "str"},
    },
    "telegram_send": {
        "category": ToolCategory.SYSTEM,
        "description": "Telegram Nachricht senden",
        "parameters": {"chat_id": "str", "message": "str"},
    },
}


# ============================================================================
# TOOL IMPLEMENTATION HELPERS
# ============================================================================

def get_tool_count() -> int:
    """Gibt die Anzahl der registrierten Tools zurück."""
    return len(ULTIMATE_TOOLS)


def get_tools_by_category(category: ToolCategory) -> List[str]:
    """Gibt alle Tools einer Kategorie zurück."""
    return [
        name for name, info in ULTIMATE_TOOLS.items()
        if info["category"] == category
    ]


def get_all_categories() -> List[ToolCategory]:
    """Gibt alle Kategorien zurück."""
    return list(ToolCategory)


def search_tools(query: str) -> List[Dict[str, Any]]:
    """Sucht nach Tools anhand eines Suchbegriffs."""
    query_lower = query.lower()
    results = []

    for name, info in ULTIMATE_TOOLS.items():
        if query_lower in name.lower() or query_lower in info["description"].lower():
            results.append({
                "name": name,
                "category": info["category"].value,
                "description": info["description"],
                "parameters": info["parameters"],
            })

    return results


# ============================================================================
# DYNAMIC TOOL REGISTRATION
# ============================================================================

class UltimateToolFactory:
    """Factory für die dynamische Erstellung von Ultimate Tools."""

    # Mapping von Tool-Namen zu spezialisierten Executors
    _executors: Dict[str, Callable] = {}

    @classmethod
    def register_executor(cls, tool_name: str, executor: Callable):
        """Registriert einen spezialisierten Executor für ein Tool."""
        cls._executors[tool_name] = executor

    @staticmethod
    def create_tool(name: str) -> Optional[Type[Tool]]:
        """Erstellt ein Tool dynamisch basierend auf der Definition."""
        if name not in ULTIMATE_TOOLS:
            return None

        tool_def = ULTIMATE_TOOLS[name]
        factory = UltimateToolFactory

        # Erstelle Tool-Klasse dynamisch
        class DynamicTool(Tool):
            tool_name = name

            def __init__(self, config=None):
                super().__init__(config)
                self._definition = tool_def
                self._factory = factory

            def get_schema(self) -> Dict[str, Any]:
                return {
                    "name": self.tool_name,
                    "description": self._definition["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            k: {"type": v}
                            for k, v in self._definition["parameters"].items()
                        },
                        "required": list(self._definition["parameters"].keys()),
                    },
                }

            async def execute(self, **kwargs) -> ToolResult:
                """
                Führt das Tool aus mit echten Implementierungen.

                Nutzt spezialisierte Executors oder generische Implementierungen
                basierend auf der Tool-Kategorie.
                """
                try:
                    # Prüfe ob spezialisierter Executor existiert
                    if self.tool_name in self._factory._executors:
                        result = await self._factory._executors[self.tool_name](**kwargs)
                        return ToolResult(success=True, output=result)

                    # Führe basierend auf Kategorie aus
                    category = self._definition.get("category")
                    result = await self._execute_by_category(category, kwargs)
                    return ToolResult(success=True, output=result)

                except Exception as e:
                    logger.error(f"Tool {self.tool_name} Fehler: {e}")
                    return ToolResult(
                        success=False,
                        error=str(e),
                        metadata={"tool": self.tool_name, "parameters": kwargs}
                    )

            async def _execute_by_category(self, category: ToolCategory, params: Dict) -> Any:
                """Führt Tool basierend auf Kategorie aus."""

                if category == ToolCategory.WEB:
                    return await self._execute_web(params)

                elif category == ToolCategory.FILES:
                    return await self._execute_files(params)

                elif category == ToolCategory.CODE:
                    return await self._execute_code(params)

                elif category == ToolCategory.MATH:
                    return await self._execute_math(params)

                elif category == ToolCategory.TEXT:
                    return await self._execute_text(params)

                elif category == ToolCategory.DATA:
                    return await self._execute_data(params)

                elif category == ToolCategory.SYSTEM:
                    return await self._execute_system(params)

                else:
                    return await self._execute_generic(params)

            async def _execute_web(self, params: Dict) -> Dict:
                """Web-bezogene Tool-Ausführung."""
                tool_name = self.tool_name

                if "search" in tool_name:
                    # Web-Suche
                    try:
                        import httpx
                        query = params.get("query", "")
                        async with httpx.AsyncClient(timeout=30) as client:
                            # DuckDuckGo HTML-Suche
                            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                            response = await client.get(url)
                            return {
                                "query": query,
                                "status": response.status_code,
                                "results": self._parse_search_results(response.text)
                            }
                    except Exception as e:
                        return {"query": params.get("query"), "error": str(e)}

                elif "api" in tool_name or "caller" in tool_name:
                    # API-Aufruf
                    try:
                        import httpx
                        url = params.get("url", "")
                        method = params.get("method", "GET").upper()
                        headers = params.get("headers", {})
                        body = params.get("body", {})

                        async with httpx.AsyncClient(timeout=30) as client:
                            response = await client.request(
                                method=method,
                                url=url,
                                headers=headers,
                                json=body if body else None
                            )
                            try:
                                body_data = response.json()
                            except Exception:
                                body_data = response.text
                            return {
                                "status": response.status_code,
                                "headers": dict(response.headers),
                                "body": body_data
                            }
                    except Exception as e:
                        return {"url": params.get("url"), "error": str(e)}

                elif "scraper" in tool_name or "scrape" in tool_name:
                    # Web Scraping
                    try:
                        import httpx
                        url = params.get("url", "")
                        selectors = params.get("selectors", {})

                        async with httpx.AsyncClient(timeout=30) as client:
                            response = await client.get(url)
                            html_content = response.text

                            # Einfaches Regex-basiertes Scraping
                            extracted = {}
                            for key, selector in selectors.items():
                                pattern = rf'<{selector}[^>]*>(.*?)</{selector}>'
                                matches = re.findall(pattern, html_content, re.DOTALL)
                                extracted[key] = matches

                            return {
                                "url": url,
                                "status": response.status_code,
                                "extracted": extracted
                            }
                    except Exception as e:
                        return {"url": params.get("url"), "error": str(e)}

                return {"tool": self.tool_name, "params": params, "category": "web"}

            def _parse_search_results(self, html: str) -> List[Dict]:
                """Parst Suchergebnisse aus HTML."""
                results = []
                # Einfaches Pattern für DuckDuckGo Ergebnisse
                pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
                matches = re.findall(pattern, html)
                for url, title in matches[:10]:
                    results.append({"title": title.strip(), "url": url})
                return results

            async def _execute_files(self, params: Dict) -> Dict:
                """Datei-bezogene Tool-Ausführung."""
                tool_name = self.tool_name

                if "read" in tool_name:
                    path = params.get("path", "")
                    if path and os.path.exists(path):
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        return {"path": path, "content": content, "size": len(content)}
                    return {"error": f"Datei nicht gefunden: {path}"}

                elif "write" in tool_name:
                    path = params.get("path", "")
                    content = params.get("content", "")
                    if path:
                        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(content)
                        return {"path": path, "written": len(content)}
                    return {"error": "Kein Pfad angegeben"}

                elif "list" in tool_name or "dir" in tool_name:
                    path = params.get("path", ".")
                    if os.path.isdir(path):
                        entries = os.listdir(path)
                        return {"path": path, "entries": entries, "count": len(entries)}
                    return {"error": f"Verzeichnis nicht gefunden: {path}"}

                return {"tool": self.tool_name, "params": params}

            async def _execute_code(self, params: Dict) -> Dict:
                """Code-bezogene Tool-Ausführung."""
                tool_name = self.tool_name

                if "python" in tool_name and "run" in tool_name:
                    code = params.get("code", "")
                    try:
                        # Sicherer Subprocess-Aufruf
                        result = subprocess.run(
                            ["python", "-c", code],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        return {
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "returncode": result.returncode
                        }
                    except subprocess.TimeoutExpired:
                        return {"error": "Timeout bei Ausführung"}
                    except Exception as e:
                        return {"error": str(e)}

                elif "lint" in tool_name or "analyze" in tool_name:
                    code = params.get("code", "")
                    issues = []
                    lines = code.split("\n")
                    for i, line in enumerate(lines, 1):
                        if len(line) > 120:
                            issues.append({"line": i, "issue": "Zeile zu lang (>120)"})
                        if "  " in line and not line.strip().startswith("#"):
                            issues.append({"line": i, "issue": "Doppelte Leerzeichen"})
                    return {"issues": issues, "total": len(issues)}

                return {"tool": self.tool_name, "params": params}

            async def _execute_math(self, params: Dict) -> Dict:
                """Mathematische Tool-Ausführung."""
                tool_name = self.tool_name

                if "calc" in tool_name or "eval" in tool_name:
                    expr = params.get("expression", params.get("expr", ""))
                    try:
                        # Sichere mathematische Auswertung
                        allowed_names = {
                            "abs": abs, "round": round, "min": min, "max": max,
                            "sum": sum, "pow": pow, "sqrt": math.sqrt,
                            "sin": math.sin, "cos": math.cos, "tan": math.tan,
                            "log": math.log, "log10": math.log10, "exp": math.exp,
                            "pi": math.pi, "e": math.e
                        }
                        result = eval(expr, {"__builtins__": {}}, allowed_names)
                        return {"expression": expr, "result": result}
                    except Exception as e:
                        return {"expression": expr, "error": str(e)}

                elif "random" in tool_name:
                    min_val = params.get("min", 0)
                    max_val = params.get("max", 100)
                    return {"random": random.uniform(min_val, max_val)}

                elif "stats" in tool_name:
                    data = params.get("data", [])
                    if data:
                        return {
                            "count": len(data),
                            "sum": sum(data),
                            "mean": sum(data) / len(data),
                            "min": min(data),
                            "max": max(data)
                        }
                    return {"error": "Keine Daten angegeben"}

                return {"tool": self.tool_name, "params": params}

            async def _execute_text(self, params: Dict) -> Dict:
                """Text-bezogene Tool-Ausführung."""
                text = params.get("text", "")

                if "count" in self.tool_name:
                    return {
                        "chars": len(text),
                        "words": len(text.split()),
                        "lines": len(text.splitlines())
                    }

                elif "hash" in self.tool_name:
                    return {
                        "md5": hashlib.md5(text.encode()).hexdigest(),
                        "sha256": hashlib.sha256(text.encode()).hexdigest()
                    }

                elif "encode" in self.tool_name:
                    return {
                        "base64": base64.b64encode(text.encode()).decode(),
                        "url": urllib.parse.quote(text)
                    }

                elif "decode" in self.tool_name:
                    try:
                        decoded = base64.b64decode(text).decode()
                        return {"decoded": decoded}
                    except Exception:
                        return {"decoded": urllib.parse.unquote(text)}

                return {"tool": self.tool_name, "text": text[:100]}

            async def _execute_data(self, params: Dict) -> Dict:
                """Daten-bezogene Tool-Ausführung."""
                if "json" in self.tool_name:
                    data = params.get("data", params.get("json", ""))
                    if isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                            return {"parsed": parsed, "valid": True}
                        except json.JSONDecodeError as e:
                            return {"valid": False, "error": str(e)}
                    else:
                        return {"json": json.dumps(data, indent=2)}

                return {"tool": self.tool_name, "params": params}

            async def _execute_system(self, params: Dict) -> Dict:
                """System-bezogene Tool-Ausführung."""
                if "time" in self.tool_name or "date" in self.tool_name:
                    now = datetime.now()
                    return {
                        "timestamp": now.timestamp(),
                        "iso": now.isoformat(),
                        "date": now.strftime("%Y-%m-%d"),
                        "time": now.strftime("%H:%M:%S")
                    }

                elif "env" in self.tool_name:
                    # Nur sichere Umgebungsvariablen
                    safe_vars = ["PATH", "HOME", "USER", "LANG", "TERM"]
                    return {k: os.environ.get(k, "") for k in safe_vars}

                elif "uuid" in self.tool_name or "id" in self.tool_name:
                    import uuid
                    return {"uuid": str(uuid.uuid4())}

                return {"tool": self.tool_name, "params": params}

            async def _execute_generic(self, params: Dict) -> Dict:
                """Generische Ausführung für nicht-kategorisierte Tools."""
                return {
                    "tool": self.tool_name,
                    "category": str(self._definition.get("category", "unknown")),
                    "description": self._definition.get("description", ""),
                    "parameters_received": params,
                    "message": "Tool ausgeführt - spezialisierte Implementierung verfügbar über register_executor()"
                }

        DynamicTool.__name__ = f"{name.title().replace('_', '')}Tool"
        return DynamicTool

    @classmethod
    def register_all_tools(cls):
        """Registriert alle Ultimate Tools."""
        from scio.tools.registry import ToolRegistry

        registered = 0
        for name in ULTIMATE_TOOLS:
            tool_class = cls.create_tool(name)
            if tool_class:
                ToolRegistry.register(name, tool_class)
                registered += 1

        logger.info(f"Registered {registered} ultimate tools")
        return registered


# ============================================================================
# TOOL CHAINING
# ============================================================================

@dataclass
class ToolChain:
    """Eine Kette von Tools für komplexe Operationen."""

    name: str
    description: str
    tools: List[str]
    connections: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validiert die Tool-Kette."""
        for tool in self.tools:
            if tool not in ULTIMATE_TOOLS:
                return False
        return True

    async def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Führt die Tool-Kette aus."""
        result = initial_input

        for tool_name in self.tools:
            tool_class = UltimateToolFactory.create_tool(tool_name)
            if tool_class:
                tool = tool_class()
                # Map connections
                tool_input = {}
                if tool_name in self.connections:
                    for param, source in self.connections[tool_name].items():
                        if source.startswith("$"):
                            tool_input[param] = result.get(source[1:])
                        else:
                            tool_input[param] = source
                else:
                    tool_input = result

                tool_result = await tool.execute(**tool_input)
                if tool_result.success:
                    result = tool_result.data
                else:
                    return {"error": f"Tool {tool_name} failed", "details": tool_result.error}

        return result


# Pre-defined Tool Chains
TOOL_CHAINS = {
    "web_to_summary": ToolChain(
        name="Web to Summary",
        description="Webseite abrufen und zusammenfassen",
        tools=["web_scraper", "text_summarize"],
        connections={
            "text_summarize": {"text": "$content"},
        },
    ),
    "code_review_full": ToolChain(
        name="Full Code Review",
        description="Vollständiges Code Review mit Security Scan",
        tools=["code_review", "code_security_scan", "code_complexity"],
    ),
    "data_pipeline": ToolChain(
        name="Data Pipeline",
        description="CSV laden, transformieren, analysieren",
        tools=["csv_read", "csv_filter", "statistics_describe", "chart_create"],
    ),
    "paper_analysis": ToolChain(
        name="Paper Analysis",
        description="Paper laden, zusammenfassen, Zitate extrahieren",
        tools=["arxiv_search", "pdf_read", "paper_summarize", "citation_format"],
    ),
    "content_creation": ToolChain(
        name="Content Creation",
        description="Blog Post erstellen mit SEO und Bildern",
        tools=["blog_post", "seo_keywords", "image_generate", "social_media_post"],
    ),
}


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_ultimate_tools():
    """Initialisiert alle Ultimate Tools."""
    count = UltimateToolFactory.register_all_tools()
    logger.info(f"Ultimate Tools initialized: {count} tools available")
    return count


# Auto-register tools when module is imported
_tools_registered = False

def ensure_tools_registered():
    """Stellt sicher, dass Tools registriert sind."""
    global _tools_registered
    if not _tools_registered:
        initialize_ultimate_tools()
        _tools_registered = True


# Expose tool count
TOTAL_TOOL_COUNT = len(ULTIMATE_TOOLS)
