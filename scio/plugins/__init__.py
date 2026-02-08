"""
SCIO Plugins - Erweiterungssystem

Ermöglicht das Laden und Verwalten von SCIO-Erweiterungen.
"""

from scio.plugins.base import Plugin, SimplePlugin, PluginMetadata
from scio.plugins.loader import PluginLoader, get_plugin_loader, load_plugin

__all__ = [
    "Plugin",
    "SimplePlugin",
    "PluginMetadata",
    "PluginLoader",
    "get_plugin_loader",
    "load_plugin",
]
