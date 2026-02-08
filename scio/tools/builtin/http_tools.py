"""
SCIO HTTP Tools

Tools für HTTP-Anfragen.
"""

from typing import Any, Optional

from scio.core.logging import get_logger
from scio.tools.base import Tool, ToolConfig
from scio.tools.registry import register_tool

logger = get_logger(__name__)


class HttpClientConfig(ToolConfig):
    """Konfiguration für HttpClient."""

    timeout: int = 30
    max_redirects: int = 5
    verify_ssl: bool = True
    default_headers: dict[str, str] = {}


@register_tool("http_client")
class HttpClientTool(Tool[dict[str, Any], dict[str, Any]]):
    """Tool für HTTP-Anfragen."""

    tool_name = "http_client"
    version = "1.0"

    def __init__(self, config: Optional[HttpClientConfig | dict] = None):
        if config is None:
            config = HttpClientConfig(name="http_client")
        elif isinstance(config, dict):
            config = HttpClientConfig(**config)
        super().__init__(config)
        self.config: HttpClientConfig = config

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Führt HTTP-Anfrage aus."""
        import aiohttp

        url = input_data.get("url")
        method = input_data.get("method", "GET").upper()
        headers = {**self.config.default_headers, **input_data.get("headers", {})}
        params = input_data.get("params")
        data = input_data.get("data")
        json_data = input_data.get("json")

        if not url:
            raise ValueError("Keine URL angegeben")

        self.logger.info("HTTP request", method=method, url=url)

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(ssl=self.config.verify_ssl)

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        ) as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                allow_redirects=True,
                max_redirects=self.config.max_redirects,
            ) as response:
                # Versuche JSON zu parsen
                content_type = response.headers.get("content-type", "")

                if "application/json" in content_type:
                    body = await response.json()
                else:
                    body = await response.text()

                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": body,
                    "url": str(response.url),
                }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": "Führt HTTP-Anfragen aus (GET, POST, PUT, DELETE, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Die URL für die Anfrage",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                        "description": "HTTP-Methode",
                        "default": "GET",
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP-Header",
                    },
                    "params": {
                        "type": "object",
                        "description": "URL-Parameter",
                    },
                    "data": {
                        "type": "string",
                        "description": "Request-Body als String",
                    },
                    "json": {
                        "type": "object",
                        "description": "Request-Body als JSON",
                    },
                },
                "required": ["url"],
            },
        }
