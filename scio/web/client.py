"""
SCIO Web Client

HTTP-Client mit synchronem und asynchronem Support.
"""

import asyncio
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from http.client import HTTPResponse as BaseHTTPResponse
from typing import Any, Optional, Union
import gzip
import zlib

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class HttpMethod(str, Enum):
    """HTTP-Methoden."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ContentType(str, Enum):
    """Häufige Content-Types."""
    JSON = "application/json"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"
    HTML = "text/html"
    XML = "application/xml"
    BINARY = "application/octet-stream"


@dataclass
class HttpResponse:
    """HTTP-Antwort."""

    status_code: int
    headers: dict[str, str]
    body: bytes
    url: str
    method: str
    elapsed_ms: float
    request_id: str = field(default_factory=lambda: generate_id("req"))
    timestamp: datetime = field(default_factory=now_utc)

    @property
    def ok(self) -> bool:
        """Prüft ob Request erfolgreich war (2xx)."""
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        """Body als String."""
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """Body als JSON."""
        return json.loads(self.body)

    @property
    def content_type(self) -> Optional[str]:
        """Content-Type Header."""
        return self.headers.get("Content-Type", self.headers.get("content-type"))

    def raise_for_status(self) -> None:
        """Wirft Exception bei Fehler-Status."""
        if not self.ok:
            raise HttpError(
                f"HTTP {self.status_code}",
                status_code=self.status_code,
                response=self,
            )


class HttpError(Exception):
    """HTTP-Fehler."""

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        response: Optional[HttpResponse] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


@dataclass
class RequestConfig:
    """Konfiguration für HTTP-Requests."""

    timeout: float = 30.0
    max_redirects: int = 10
    verify_ssl: bool = True
    user_agent: str = "SCIO/1.0 (Scientific Intelligent Operations)"
    max_retries: int = 3
    retry_delay: float = 1.0
    follow_redirects: bool = True


class WebClient:
    """
    Synchroner HTTP-Client.

    Features:
    - GET, POST, PUT, DELETE, etc.
    - JSON-Support
    - File-Downloads
    - Cookie-Handling
    - Retry-Logik
    - Komprimierung
    """

    def __init__(self, config: Optional[RequestConfig] = None):
        self.config = config or RequestConfig()
        self._cookies: dict[str, str] = {}
        self._default_headers: dict[str, str] = {
            "User-Agent": self.config.user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        logger.info("WebClient initialized")

    def request(
        self,
        method: Union[str, HttpMethod],
        url: str,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, str]] = None,
        data: Optional[Union[str, bytes, dict]] = None,
        json_data: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        """
        Führt einen HTTP-Request aus.

        Args:
            method: HTTP-Methode
            url: Ziel-URL
            headers: Zusätzliche Header
            params: Query-Parameter
            data: Request-Body
            json_data: JSON-Body (wird automatisch serialisiert)
            timeout: Timeout in Sekunden

        Returns:
            HttpResponse mit Antwortdaten
        """
        method_str = method.value if isinstance(method, HttpMethod) else method.upper()
        start_time = datetime.now()

        # URL mit Query-Parametern
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}" if "?" not in url else f"{url}&{query}"

        # Headers zusammenstellen
        request_headers = self._default_headers.copy()
        if self._cookies:
            cookie_str = "; ".join(f"{k}={v}" for k, v in self._cookies.items())
            request_headers["Cookie"] = cookie_str
        if headers:
            request_headers.update(headers)

        # Body vorbereiten
        body_bytes: Optional[bytes] = None
        if json_data is not None:
            body_bytes = json.dumps(json_data).encode("utf-8")
            request_headers["Content-Type"] = ContentType.JSON.value
        elif data is not None:
            if isinstance(data, dict):
                body_bytes = urllib.parse.urlencode(data).encode("utf-8")
                request_headers["Content-Type"] = ContentType.FORM.value
            elif isinstance(data, str):
                body_bytes = data.encode("utf-8")
            else:
                body_bytes = data

        # Request erstellen
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers=request_headers,
            method=method_str,
        )

        # SSL-Kontext
        ssl_context = None
        if url.startswith("https://"):
            ssl_context = ssl.create_default_context()
            if not self.config.verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

        # Request mit Retry-Logik
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(
                    req,
                    timeout=timeout or self.config.timeout,
                    context=ssl_context,
                ) as response:
                    response_body = self._read_response(response)
                    response_headers = dict(response.headers)

                    # Cookies speichern
                    self._extract_cookies(response_headers)

                    elapsed = (datetime.now() - start_time).total_seconds() * 1000

                    http_response = HttpResponse(
                        status_code=response.status,
                        headers=response_headers,
                        body=response_body,
                        url=response.url,
                        method=method_str,
                        elapsed_ms=elapsed,
                    )

                    logger.debug(
                        "HTTP request completed",
                        method=method_str,
                        url=url[:100],
                        status=response.status,
                        elapsed_ms=elapsed,
                    )

                    return http_response

            except urllib.error.HTTPError as e:
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                response_body = e.read()
                return HttpResponse(
                    status_code=e.code,
                    headers=dict(e.headers),
                    body=response_body,
                    url=url,
                    method=method_str,
                    elapsed_ms=elapsed,
                )

            except urllib.error.URLError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    import time
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    logger.warning(
                        "Request failed, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                    )

        raise HttpError(f"Request failed after {self.config.max_retries} attempts: {last_error}")

    def _read_response(self, response: BaseHTTPResponse) -> bytes:
        """Liest und dekomprimiert die Response."""
        body = response.read()
        encoding = response.headers.get("Content-Encoding", "").lower()

        if encoding == "gzip":
            body = gzip.decompress(body)
        elif encoding == "deflate":
            body = zlib.decompress(body)

        return body

    def _extract_cookies(self, headers: dict[str, str]) -> None:
        """Extrahiert Cookies aus Response-Headers."""
        set_cookie = headers.get("Set-Cookie", headers.get("set-cookie", ""))
        if set_cookie:
            for cookie in set_cookie.split(","):
                parts = cookie.strip().split(";")[0]
                if "=" in parts:
                    name, value = parts.split("=", 1)
                    self._cookies[name.strip()] = value.strip()

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        """GET-Request."""
        return self.request(HttpMethod.GET, url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> HttpResponse:
        """POST-Request."""
        return self.request(HttpMethod.POST, url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> HttpResponse:
        """PUT-Request."""
        return self.request(HttpMethod.PUT, url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> HttpResponse:
        """DELETE-Request."""
        return self.request(HttpMethod.DELETE, url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> HttpResponse:
        """PATCH-Request."""
        return self.request(HttpMethod.PATCH, url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> HttpResponse:
        """HEAD-Request."""
        return self.request(HttpMethod.HEAD, url, **kwargs)

    def download(
        self,
        url: str,
        path: str,
        chunk_size: int = 8192,
        progress_callback: Optional[callable] = None,
    ) -> int:
        """
        Lädt eine Datei herunter.

        Args:
            url: Download-URL
            path: Zielpfad
            chunk_size: Größe der Chunks
            progress_callback: Callback für Fortschritt (bytes_downloaded, total_bytes)

        Returns:
            Anzahl heruntergeladener Bytes
        """
        response = self.get(url)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with open(path, "wb") as f:
            f.write(response.body)
            downloaded = len(response.body)

        if progress_callback:
            progress_callback(downloaded, total_size)

        logger.info("File downloaded", url=url[:50], path=path, size=downloaded)
        return downloaded

    def set_cookie(self, name: str, value: str) -> None:
        """Setzt einen Cookie."""
        self._cookies[name] = value

    def clear_cookies(self) -> None:
        """Löscht alle Cookies."""
        self._cookies.clear()


class AsyncWebClient:
    """
    Asynchroner HTTP-Client.

    Nutzt asyncio für nicht-blockierende Requests.
    """

    def __init__(self, config: Optional[RequestConfig] = None):
        self.config = config or RequestConfig()
        self._sync_client = WebClient(config)
        logger.info("AsyncWebClient initialized")

    async def request(
        self,
        method: Union[str, HttpMethod],
        url: str,
        **kwargs: Any,
    ) -> HttpResponse:
        """Asynchroner HTTP-Request."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_client.request(method, url, **kwargs),
        )

    async def get(self, url: str, **kwargs: Any) -> HttpResponse:
        """Asynchroner GET-Request."""
        return await self.request(HttpMethod.GET, url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> HttpResponse:
        """Asynchroner POST-Request."""
        return await self.request(HttpMethod.POST, url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> HttpResponse:
        """Asynchroner PUT-Request."""
        return await self.request(HttpMethod.PUT, url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> HttpResponse:
        """Asynchroner DELETE-Request."""
        return await self.request(HttpMethod.DELETE, url, **kwargs)

    async def gather(
        self,
        requests: list[tuple[str, str, dict[str, Any]]],
    ) -> list[HttpResponse]:
        """
        Führt mehrere Requests parallel aus.

        Args:
            requests: Liste von (method, url, kwargs) Tupeln

        Returns:
            Liste von Responses
        """
        tasks = [
            self.request(method, url, **kwargs)
            for method, url, kwargs in requests
        ]
        return await asyncio.gather(*tasks)

    async def download(
        self,
        url: str,
        path: str,
        **kwargs: Any,
    ) -> int:
        """Asynchroner Download."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_client.download(url, path, **kwargs),
        )
