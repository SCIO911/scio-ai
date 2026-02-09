"""
SCIO API Client

REST, GraphQL und Webhook-Funktionalitäten.
"""

import base64
import hashlib
import hmac
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Optional, Union
from urllib.parse import urljoin

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc
from scio.web.client import WebClient, HttpResponse, RequestConfig

logger = get_logger(__name__)


@dataclass
class APIConfig:
    """API-Client Konfiguration."""

    base_url: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    auth_type: str = "bearer"  # bearer, basic, api_key, hmac
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit: Optional[int] = None  # Requests pro Sekunde
    headers: dict[str, str] = field(default_factory=dict)


class APIClient:
    """
    Generischer API-Client.

    Features:
    - Authentifizierung (Bearer, Basic, API Key, HMAC)
    - Rate Limiting
    - Automatische Retry-Logik
    - Response-Caching
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self._client = WebClient(RequestConfig(
            timeout=config.timeout,
            max_retries=config.max_retries,
        ))
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._last_request: Optional[datetime] = None
        logger.info("APIClient initialized", base_url=config.base_url[:50])

    def _get_auth_headers(self) -> dict[str, str]:
        """Erstellt Authentifizierungs-Header."""
        headers = self.config.headers.copy()

        if not self.config.api_key:
            return headers

        if self.config.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        elif self.config.auth_type == "basic":
            secret = self.config.api_secret or ""
            credentials = base64.b64encode(
                f"{self.config.api_key}:{secret}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        elif self.config.auth_type == "api_key":
            headers["X-API-Key"] = self.config.api_key

        return headers

    def _rate_limit_wait(self) -> None:
        """Wartet falls Rate Limit erreicht."""
        if not self.config.rate_limit or not self._last_request:
            return

        import time
        min_interval = 1.0 / self.config.rate_limit
        elapsed = (datetime.now() - self._last_request).total_seconds()

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        cache_ttl: Optional[int] = None,
    ) -> HttpResponse:
        """
        Führt einen API-Request aus.

        Args:
            method: HTTP-Methode
            endpoint: API-Endpoint (wird an base_url angehängt)
            params: Query-Parameter
            data: Request-Body (wird als JSON gesendet)
            headers: Zusätzliche Header
            cache_ttl: Cache-Lebensdauer in Sekunden

        Returns:
            HttpResponse
        """
        self._rate_limit_wait()

        url = urljoin(self.config.base_url, endpoint)

        # Cache prüfen (nur für GET)
        if method.upper() == "GET" and cache_ttl:
            cache_key = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
            if cache_key in self._cache:
                cached_time, cached_response = self._cache[cache_key]
                age = (datetime.now() - cached_time).total_seconds()
                if age < cache_ttl:
                    logger.debug("Cache hit", endpoint=endpoint)
                    return cached_response

        # Headers zusammenstellen
        request_headers = self._get_auth_headers()
        request_headers["Content-Type"] = "application/json"
        request_headers["Accept"] = "application/json"
        if headers:
            request_headers.update(headers)

        # HMAC-Signatur falls konfiguriert
        if self.config.auth_type == "hmac" and self.config.api_secret:
            signature = self._create_hmac_signature(method, endpoint, data)
            request_headers["X-Signature"] = signature

        self._last_request = datetime.now()

        response = self._client.request(
            method=method,
            url=url,
            params=params,
            json_data=data,
            headers=request_headers,
        )

        # Cache speichern
        if method.upper() == "GET" and cache_ttl and response.ok:
            cache_key = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
            self._cache[cache_key] = (datetime.now(), response)

        return response

    def _create_hmac_signature(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict],
    ) -> str:
        """Erstellt HMAC-Signatur."""
        message = f"{method.upper()}{endpoint}"
        if data:
            message += json.dumps(data, sort_keys=True)

        signature = hmac.new(
            self.config.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return signature

    def get(self, endpoint: str, **kwargs: Any) -> HttpResponse:
        """GET-Request."""
        return self.request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs: Any) -> HttpResponse:
        """POST-Request."""
        return self.request("POST", endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs: Any) -> HttpResponse:
        """PUT-Request."""
        return self.request("PUT", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs: Any) -> HttpResponse:
        """DELETE-Request."""
        return self.request("DELETE", endpoint, **kwargs)

    def patch(self, endpoint: str, **kwargs: Any) -> HttpResponse:
        """PATCH-Request."""
        return self.request("PATCH", endpoint, **kwargs)

    def clear_cache(self) -> None:
        """Löscht den Cache."""
        self._cache.clear()


class RESTClient(APIClient):
    """
    REST API Client mit CRUD-Operationen.

    Vereinfacht die Arbeit mit RESTful APIs.
    """

    def list(
        self,
        resource: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Listet Ressourcen auf (GET /resource)."""
        response = self.get(f"/{resource}", params=params)
        response.raise_for_status()
        return response.json()

    def create(
        self,
        resource: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Erstellt eine Ressource (POST /resource)."""
        response = self.post(f"/{resource}", data=data)
        response.raise_for_status()
        return response.json()

    def read(
        self,
        resource: str,
        resource_id: Union[str, int],
    ) -> dict[str, Any]:
        """Liest eine Ressource (GET /resource/:id)."""
        response = self.get(f"/{resource}/{resource_id}")
        response.raise_for_status()
        return response.json()

    def update(
        self,
        resource: str,
        resource_id: Union[str, int],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Aktualisiert eine Ressource (PUT /resource/:id)."""
        response = self.put(f"/{resource}/{resource_id}", data=data)
        response.raise_for_status()
        return response.json()

    def partial_update(
        self,
        resource: str,
        resource_id: Union[str, int],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Teilaktualisierung (PATCH /resource/:id)."""
        response = self.patch(f"/{resource}/{resource_id}", data=data)
        response.raise_for_status()
        return response.json()

    def destroy(
        self,
        resource: str,
        resource_id: Union[str, int],
    ) -> bool:
        """Löscht eine Ressource (DELETE /resource/:id)."""
        response = self.delete(f"/{resource}/{resource_id}")
        return response.ok


class GraphQLClient(APIClient):
    """
    GraphQL API Client.

    Features:
    - Query/Mutation Ausführung
    - Variable-Support
    - Introspection
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        auth_type: str = "bearer",
    ):
        config = APIConfig(
            base_url=endpoint,
            api_key=api_key,
            auth_type=auth_type,
        )
        super().__init__(config)
        self._endpoint = endpoint

    def execute(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
        operation_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Führt eine GraphQL-Query oder Mutation aus.

        Args:
            query: GraphQL-Query oder Mutation
            variables: Query-Variablen
            operation_name: Name der Operation (optional)

        Returns:
            Query-Ergebnis
        """
        payload: dict[str, Any] = {"query": query}

        if variables:
            payload["variables"] = variables
        if operation_name:
            payload["operationName"] = operation_name

        response = self.post("", data=payload)
        response.raise_for_status()

        result = response.json()

        if "errors" in result:
            errors = result["errors"]
            error_messages = [e.get("message", str(e)) for e in errors]
            raise GraphQLError(f"GraphQL errors: {error_messages}", errors=errors)

        return result.get("data", {})

    def query(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Führt eine Query aus."""
        return self.execute(query, variables)

    def mutate(
        self,
        mutation: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Führt eine Mutation aus."""
        return self.execute(mutation, variables)

    def introspect(self) -> dict[str, Any]:
        """Führt eine Introspection-Query aus."""
        query = """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    kind
                    fields {
                        name
                        type {
                            name
                            kind
                        }
                    }
                }
                queryType { name }
                mutationType { name }
            }
        }
        """
        return self.execute(query)


class GraphQLError(Exception):
    """GraphQL-Fehler."""

    def __init__(self, message: str, errors: Optional[list] = None):
        super().__init__(message)
        self.errors = errors or []


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP-Handler für Webhooks."""

    callbacks: dict[str, Callable] = {}
    secret: Optional[str] = None

    def log_message(self, format: str, *args: Any) -> None:
        """Unterdrücke Standard-Logging."""
        pass

    def do_POST(self) -> None:
        """Behandelt POST-Requests."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Signatur prüfen falls secret konfiguriert
            if self.secret:
                signature = self.headers.get("X-Signature", "")
                expected = hmac.new(
                    self.secret.encode(),
                    body,
                    hashlib.sha256,
                ).hexdigest()

                if not hmac.compare_digest(signature, expected):
                    self.send_error(401, "Invalid signature")
                    return

            # Payload parsen
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {"raw": body.decode("utf-8", errors="replace")}

            # Callback aufrufen
            path = self.path.strip("/")
            if path in self.callbacks:
                result = self.callbacks[path](payload, dict(self.headers))
                response = json.dumps(result if result else {"ok": True})
            elif "" in self.callbacks:
                result = self.callbacks[""](payload, dict(self.headers))
                response = json.dumps(result if result else {"ok": True})
            else:
                self.send_error(404, "No handler for this path")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.encode())

        except Exception as e:
            logger.error("Webhook handler error", error=str(e))
            self.send_error(500, str(e))

    def do_GET(self) -> None:
        """Health check."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')


class WebhookServer:
    """
    Webhook-Server für eingehende Requests.

    Features:
    - Registrierung von Callbacks
    - Signatur-Validierung
    - Async-Verarbeitung
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        secret: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.secret = secret
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._callbacks: dict[str, Callable] = {}
        logger.info("WebhookServer initialized", host=host, port=port)

    def register(
        self,
        path: str,
        callback: Callable[[dict, dict], Optional[dict]],
    ) -> None:
        """
        Registriert einen Webhook-Handler.

        Args:
            path: URL-Pfad (z.B. "github" für /github)
            callback: Funktion die (payload, headers) empfängt
        """
        self._callbacks[path.strip("/")] = callback
        logger.debug("Webhook registered", path=path)

    def start(self, blocking: bool = False) -> None:
        """
        Startet den Webhook-Server.

        Args:
            blocking: Wenn True, blockiert bis Server beendet
        """
        # Handler-Klasse konfigurieren
        WebhookHandler.callbacks = self._callbacks
        WebhookHandler.secret = self.secret

        self._server = HTTPServer((self.host, self.port), WebhookHandler)

        if blocking:
            logger.info("Webhook server starting (blocking)", port=self.port)
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
            logger.info("Webhook server started", port=self.port)

    def stop(self) -> None:
        """Stoppt den Server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("Webhook server stopped")

    def __enter__(self) -> "WebhookServer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
