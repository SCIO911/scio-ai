"""
SCIO Plugin Loader

System zum Laden und Verwalten von Plugins.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional, Type

from scio.core.exceptions import PluginError
from scio.core.logging import get_logger
from scio.plugins.base import Plugin, PluginMetadata

logger = get_logger(__name__)


class PluginLoader:
    """
    Lädt und verwaltet SCIO-Plugins.

    Plugins können aus:
    - Python-Paketen (pip install)
    - Lokalen Verzeichnissen
    - Einzelnen Python-Dateien
    geladen werden.
    """

    def __init__(self, plugin_dirs: Optional[list[Path]] = None):
        self.plugin_dirs = plugin_dirs or []
        self.loaded_plugins: dict[str, Plugin] = {}
        self.logger = get_logger(__name__, component="plugin_loader")

    def discover(self) -> list[PluginMetadata]:
        """
        Entdeckt verfügbare Plugins.

        Returns:
            Liste von Plugin-Metadaten
        """
        discovered = []

        # Suche in Plugin-Verzeichnissen
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            for item in plugin_dir.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    # Python-Paket
                    meta = self._extract_metadata_from_package(item)
                    if meta:
                        discovered.append(meta)

                elif item.suffix == ".py" and not item.name.startswith("_"):
                    # Einzelne Python-Datei
                    meta = self._extract_metadata_from_file(item)
                    if meta:
                        discovered.append(meta)

        return discovered

    def load(self, plugin_name: str, config: Optional[dict] = None) -> Plugin:
        """
        Lädt ein Plugin.

        Args:
            plugin_name: Name des Plugins oder Pfad
            config: Optionale Konfiguration

        Returns:
            Geladene Plugin-Instanz
        """
        if plugin_name in self.loaded_plugins:
            self.logger.info("Plugin already loaded", plugin=plugin_name)
            return self.loaded_plugins[plugin_name]

        self.logger.info("Loading plugin", plugin=plugin_name)

        # Versuche als installiertes Paket zu laden
        try:
            module = importlib.import_module(f"scio_plugin_{plugin_name}")
        except ImportError:
            # Versuche als lokales Plugin
            module = self._load_local_plugin(plugin_name)

        # Finde Plugin-Klasse
        plugin_class = self._find_plugin_class(module)
        if plugin_class is None:
            raise PluginError(
                f"Keine Plugin-Klasse in {plugin_name} gefunden",
                plugin_name=plugin_name,
            )

        # Erstelle Instanz
        plugin = plugin_class(config or {})

        # Initialisiere Plugin
        plugin.on_load()

        self.loaded_plugins[plugin_name] = plugin
        self.logger.info("Plugin loaded", plugin=plugin_name, version=plugin.metadata.version)

        return plugin

    def unload(self, plugin_name: str) -> bool:
        """Entlädt ein Plugin."""
        if plugin_name not in self.loaded_plugins:
            return False

        plugin = self.loaded_plugins[plugin_name]
        plugin.on_unload()

        del self.loaded_plugins[plugin_name]
        self.logger.info("Plugin unloaded", plugin=plugin_name)

        return True

    def get(self, plugin_name: str) -> Optional[Plugin]:
        """Gibt ein geladenes Plugin zurück."""
        return self.loaded_plugins.get(plugin_name)

    def list_loaded(self) -> list[str]:
        """Gibt Namen aller geladenen Plugins zurück."""
        return list(self.loaded_plugins.keys())

    def _load_local_plugin(self, plugin_name: str) -> Any:
        """Lädt ein lokales Plugin."""
        for plugin_dir in self.plugin_dirs:
            # Als Verzeichnis
            plugin_path = plugin_dir / plugin_name
            if plugin_path.is_dir() and (plugin_path / "__init__.py").exists():
                return self._load_from_path(plugin_path / "__init__.py", plugin_name)

            # Als Datei
            plugin_file = plugin_dir / f"{plugin_name}.py"
            if plugin_file.exists():
                return self._load_from_path(plugin_file, plugin_name)

        raise PluginError(f"Plugin nicht gefunden: {plugin_name}", plugin_name=plugin_name)

    def _load_from_path(self, path: Path, name: str) -> Any:
        """Lädt ein Modul aus einem Pfad."""
        spec = importlib.util.spec_from_file_location(f"scio_plugin_{name}", path)
        if spec is None or spec.loader is None:
            raise PluginError(f"Konnte Plugin nicht laden: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"scio_plugin_{name}"] = module
        spec.loader.exec_module(module)

        return module

    def _find_plugin_class(self, module: Any) -> Optional[Type[Plugin]]:
        """Findet die Plugin-Klasse in einem Modul."""
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Plugin)
                and obj is not Plugin
            ):
                return obj
        return None

    def _extract_metadata_from_package(self, path: Path) -> Optional[PluginMetadata]:
        """Extrahiert Metadaten aus einem Plugin-Paket."""
        try:
            module = self._load_from_path(path / "__init__.py", path.name)
            plugin_class = self._find_plugin_class(module)
            if plugin_class:
                return plugin_class.metadata
        except Exception as e:
            self.logger.warning(f"Konnte Metadaten nicht lesen: {path}", error=str(e))
        return None

    def _extract_metadata_from_file(self, path: Path) -> Optional[PluginMetadata]:
        """Extrahiert Metadaten aus einer Plugin-Datei."""
        try:
            module = self._load_from_path(path, path.stem)
            plugin_class = self._find_plugin_class(module)
            if plugin_class:
                return plugin_class.metadata
        except Exception as e:
            self.logger.warning(f"Konnte Metadaten nicht lesen: {path}", error=str(e))
        return None


# Globale Instanz
_loader: Optional[PluginLoader] = None


def get_plugin_loader() -> PluginLoader:
    """Gibt den globalen Plugin-Loader zurück."""
    global _loader
    if _loader is None:
        _loader = PluginLoader()
    return _loader


def load_plugin(name: str, config: Optional[dict] = None) -> Plugin:
    """Lädt ein Plugin."""
    return get_plugin_loader().load(name, config)
