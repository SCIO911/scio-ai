"""
SCIO - Scientific Intelligent Operations Framework

Ein wissenschaftliches Agentenframework für reproduzierbare, sichere und
nachvollziehbare Forschung.
"""

__version__ = "0.1.0"
__author__ = "SCIO Team"

from scio.core.config import Config
from scio.core.logging import get_logger

__all__ = ["Config", "get_logger", "__version__"]
