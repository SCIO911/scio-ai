#!/usr/bin/env python3
"""
SCIO - Routes
Alle API-Endpunkte für die AI-Workstation
"""

from flask import Blueprint

# Create blueprints
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')
webhooks_bp = Blueprint('webhooks', __name__, url_prefix='/webhooks')

# Import route modules
from . import api, admin, webhooks

# Import tools blueprint
from .tools import tools_bp

__all__ = ['api_bp', 'admin_bp', 'webhooks_bp', 'tools_bp']
