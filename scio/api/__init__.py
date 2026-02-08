"""
SCIO REST API

FastAPI-basierte REST API für SCIO.
"""

from scio.api.app import create_app, app
from scio.api.routes import router

__all__ = ["create_app", "app", "router"]
