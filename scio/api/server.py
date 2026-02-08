"""
SCIO API Server

Startet den SCIO API Server.
"""

import uvicorn

from scio.api.app import app
from scio.core.config import get_config


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Startet den API-Server."""
    config = get_config()

    uvicorn.run(
        "scio.api.app:app",
        host=host,
        port=port,
        reload=reload or config.debug,
        log_level="debug" if config.debug else "info",
    )


if __name__ == "__main__":
    run_server()
