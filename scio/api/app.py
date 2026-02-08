"""
SCIO API Application

FastAPI-Anwendung für die SCIO REST API.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scio import __version__
from scio.api.routes import router
from scio.core.config import get_config
from scio.core.logging import setup_logging, get_logger

# Importiere Builtin-Module um sie zu registrieren
import scio.agents.builtin  # noqa
import scio.tools.builtin  # noqa

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application Lifespan Handler."""
    # Startup
    setup_logging()
    logger.info("SCIO API starting", version=__version__)

    yield

    # Shutdown
    logger.info("SCIO API shutting down")


def create_app() -> FastAPI:
    """Erstellt die FastAPI-Anwendung."""
    config = get_config()

    app = FastAPI(
        title="SCIO API",
        description="REST API für das SCIO Scientific Intelligent Operations Framework",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Router einbinden
    app.include_router(router, prefix="/api/v1")

    return app


# Globale App-Instanz
app = create_app()
