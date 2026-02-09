"""
SCIO API Agent

Agent fuer REST und GraphQL API-Interaktionen.
"""

import json
from typing import Any, Optional
from urllib.parse import urljoin

from pydantic import Field

from scio.agents.base import Agent, AgentConfig, AgentContext
from scio.agents.registry import register_agent
from scio.core.exceptions import AgentError
from scio.core.logging import get_logger

logger = get_logger(__name__)


class APIConfig(AgentConfig):
    """Konfiguration fuer den API Agent."""

    name: str = Field(default="API Agent")
    description: str = Field(default="Agent fuer REST und GraphQL API-Aufrufe")

    # Base settings
    base_url: Optional[str] = Field(
        default=None,
        description="Basis-URL fuer alle API-Aufrufe"
    )

    # Authentication
    auth_type: Optional[str] = Field(
        default=None,
        description="Authentifizierungstyp: bearer, basic, api_key, oauth2"
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Auth-Token oder API-Key"
    )
    auth_header: str = Field(
        default="Authorization",
        description="Header fuer API-Key-Auth"
    )

    # Request settings
    timeout_seconds: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0)

    # Headers
    default_headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )

    # Rate limiting
    rate_limit: Optional[int] = Field(
        default=None,
        description="Max Requests pro Minute"
    )

    # Response settings
    max_response_size_mb: int = Field(default=10, ge=1)


@register_agent("api")
class APIAgent(Agent[dict[str, Any], dict[str, Any]]):
    """
    Agent fuer API-Interaktionen.

    Unterstuetzt:
    - REST APIs (GET, POST, PUT, PATCH, DELETE)
    - GraphQL APIs
    - Verschiedene Authentifizierungsmethoden
    - Automatische Retries
    - Rate Limiting
    """

    agent_type = "api"
    version = "1.0"

    def __init__(self, config: APIConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = APIConfig(**config)
        super().__init__(config)
        self.config: APIConfig = config
        self._request_count = 0
        self._last_request_time = 0

    def _get_headers(self, extra_headers: dict = None) -> dict[str, str]:
        """Erstellt die Request-Headers."""
        headers = dict(self.config.default_headers)

        # Authentication
        if self.config.auth_type and self.config.auth_token:
            if self.config.auth_type == "bearer":
                headers["Authorization"] = f"Bearer {self.config.auth_token}"
            elif self.config.auth_type == "basic":
                import base64
                encoded = base64.b64encode(self.config.auth_token.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"
            elif self.config.auth_type == "api_key":
                headers[self.config.auth_header] = self.config.auth_token
            # oauth2 would require more complex handling

        if extra_headers:
            headers.update(extra_headers)

        return headers

    def _build_url(self, endpoint: str) -> str:
        """Baut die vollstaendige URL."""
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        if self.config.base_url:
            return urljoin(self.config.base_url.rstrip("/") + "/", endpoint.lstrip("/"))
        return endpoint

    async def execute(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Fuehrt API-Aufrufe aus."""

        api_type = input_data.get("type", "rest")

        if api_type == "rest":
            return await self._execute_rest(input_data, context)
        elif api_type == "graphql":
            return await self._execute_graphql(input_data, context)
        else:
            raise AgentError(
                f"Unbekannter API-Typ: {api_type}",
                agent_id=self.agent_id
            )

    async def _execute_rest(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Fuehrt einen REST-API-Aufruf aus."""
        import aiohttp

        method = input_data.get("method", "GET").upper()
        endpoint = input_data.get("endpoint", "")
        params = input_data.get("params", {})
        body = input_data.get("body")
        headers = self._get_headers(input_data.get("headers", {}))

        url = self._build_url(endpoint)

        self.logger.info(
            "Making REST request",
            method=method,
            url=url
        )

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(self.config.max_retries + 1):
                try:
                    async with session.request(
                        method=method,
                        url=url,
                        params=params,
                        json=body if body else None,
                        headers=headers,
                    ) as response:
                        content_type = response.headers.get("Content-Type", "")

                        # Check response size
                        content_length = response.headers.get("Content-Length")
                        if content_length:
                            size_mb = int(content_length) / (1024 * 1024)
                            if size_mb > self.config.max_response_size_mb:
                                raise AgentError(
                                    f"Response zu gross: {size_mb:.1f}MB",
                                    agent_id=self.agent_id
                                )

                        # Parse response
                        if "application/json" in content_type:
                            data = await response.json()
                        else:
                            data = await response.text()

                        return {
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "data": data,
                            "success": 200 <= response.status < 300,
                            "url": str(response.url),
                        }

                except aiohttp.ClientError as e:
                    if attempt < self.config.max_retries:
                        self.logger.warning(
                            "Request failed, retrying",
                            attempt=attempt + 1,
                            error=str(e)
                        )
                        import asyncio
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    raise AgentError(
                        f"API-Fehler nach {self.config.max_retries + 1} Versuchen: {e}",
                        agent_id=self.agent_id
                    )

    async def _execute_graphql(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Fuehrt einen GraphQL-API-Aufruf aus."""
        import aiohttp

        endpoint = input_data.get("endpoint", "/graphql")
        query = input_data.get("query", "")
        variables = input_data.get("variables", {})
        operation_name = input_data.get("operation_name")
        headers = self._get_headers(input_data.get("headers", {}))

        if not query:
            raise AgentError("GraphQL query erforderlich", agent_id=self.agent_id)

        url = self._build_url(endpoint)

        payload = {
            "query": query,
            "variables": variables,
        }
        if operation_name:
            payload["operationName"] = operation_name

        self.logger.info("Making GraphQL request", url=url)

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(self.config.max_retries + 1):
                try:
                    async with session.post(
                        url=url,
                        json=payload,
                        headers=headers,
                    ) as response:
                        data = await response.json()

                        # Check for GraphQL errors
                        errors = data.get("errors", [])
                        has_errors = len(errors) > 0

                        return {
                            "status_code": response.status,
                            "data": data.get("data"),
                            "errors": errors,
                            "success": response.status == 200 and not has_errors,
                            "url": str(response.url),
                        }

                except aiohttp.ClientError as e:
                    if attempt < self.config.max_retries:
                        self.logger.warning(
                            "GraphQL request failed, retrying",
                            attempt=attempt + 1,
                            error=str(e)
                        )
                        import asyncio
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    raise AgentError(
                        f"GraphQL-Fehler nach {self.config.max_retries + 1} Versuchen: {e}",
                        agent_id=self.agent_id
                    )


@register_agent("webhook")
class WebhookAgent(Agent[dict[str, Any], dict[str, Any]]):
    """
    Agent zum Senden von Webhooks.

    Spezialisierte Version des API-Agents fuer Webhook-Integrationen.
    """

    agent_type = "webhook"
    version = "1.0"

    def __init__(self, config: APIConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = APIConfig(**config)
        super().__init__(config)
        self.config: APIConfig = config

    async def execute(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Sendet einen Webhook."""
        import aiohttp

        url = input_data.get("url") or self.config.base_url
        if not url:
            raise AgentError("Webhook URL erforderlich", agent_id=self.agent_id)

        payload = input_data.get("payload", {})
        headers = {
            "Content-Type": "application/json",
            **(input_data.get("headers", {}))
        }

        # Add signature if secret provided
        secret = input_data.get("secret")
        if secret:
            import hashlib
            import hmac
            signature = hmac.new(
                secret.encode(),
                json.dumps(payload, sort_keys=True).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        self.logger.info("Sending webhook", url=url)

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    url=url,
                    json=payload,
                    headers=headers,
                ) as response:
                    try:
                        data = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError):
                        data = await response.text()

                    return {
                        "status_code": response.status,
                        "data": data,
                        "success": 200 <= response.status < 300,
                        "url": str(response.url),
                    }

            except aiohttp.ClientError as e:
                raise AgentError(
                    f"Webhook-Fehler: {e}",
                    agent_id=self.agent_id
                )
