"""
SCIO Sandbox

Sichere Ausführungsumgebung für Code und Agenten.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from scio.core.config import get_config
from scio.core.exceptions import SecurityError
from scio.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SandboxConfig:
    """Konfiguration für die Sandbox."""

    enabled: bool = True
    allowed_paths: list[Path] = field(default_factory=list)
    blocked_modules: list[str] = field(
        default_factory=lambda: ["os", "subprocess", "shutil", "socket", "ctypes", "multiprocessing"]
    )
    safe_modules: list[str] = field(
        default_factory=lambda: ["json", "math", "re", "datetime", "collections", "itertools", "functools", "typing", "dataclasses", "enum", "copy", "io", "base64", "hashlib", "random", "statistics", "decimal", "fractions"]
    )
    network_enabled: bool = False
    max_memory_mb: int = 1024
    max_cpu_time_seconds: int = 300
    allow_file_write: bool = False
    allowed_write_paths: list[Path] = field(default_factory=list)


class Sandbox:
    """
    Sandbox für sichere Code-Ausführung.

    Stellt eine isolierte Umgebung bereit mit:
    - Eingeschränktem Dateisystem-Zugriff
    - Blockierten gefährlichen Modulen
    - Ressourcen-Limits
    - Netzwerk-Kontrolle
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.logger = get_logger(__name__, component="sandbox")

        if not self.config.enabled:
            self.logger.warning("Sandbox is DISABLED - running in unsafe mode")

    def check_path_access(self, path: Path | str, write: bool = False) -> bool:
        """
        Prüft ob Zugriff auf einen Pfad erlaubt ist.

        Args:
            path: Zu prüfender Pfad
            write: True für Schreibzugriff

        Returns:
            True wenn Zugriff erlaubt

        Raises:
            SecurityError: Bei nicht erlaubtem Zugriff
        """
        if not self.config.enabled:
            return True

        path = Path(path).resolve()

        # Schreibzugriff
        if write:
            if not self.config.allow_file_write:
                raise SecurityError(
                    f"Schreibzugriff nicht erlaubt: {path}",
                    details={"path": str(path)},
                )

            allowed = any(
                self._is_subpath(path, allowed_path)
                for allowed_path in self.config.allowed_write_paths
            )
            if not allowed:
                raise SecurityError(
                    f"Schreibzugriff außerhalb erlaubter Pfade: {path}",
                    details={
                        "path": str(path),
                        "allowed": [str(p) for p in self.config.allowed_write_paths],
                    },
                )

        # Lesezugriff
        if self.config.allowed_paths:
            allowed = any(
                self._is_subpath(path, allowed_path)
                for allowed_path in self.config.allowed_paths
            )
            if not allowed:
                raise SecurityError(
                    f"Lesezugriff außerhalb erlaubter Pfade: {path}",
                    details={
                        "path": str(path),
                        "allowed": [str(p) for p in self.config.allowed_paths],
                    },
                )

        return True

    def check_module_import(self, module_name: str) -> bool:
        """
        Prüft ob ein Modul-Import erlaubt ist.

        Args:
            module_name: Name des zu importierenden Moduls

        Returns:
            True wenn Import erlaubt

        Raises:
            SecurityError: Bei blockiertem Modul
        """
        if not self.config.enabled:
            return True

        for blocked in self.config.blocked_modules:
            if module_name == blocked or module_name.startswith(f"{blocked}."):
                raise SecurityError(
                    f"Import von '{module_name}' ist blockiert",
                    details={"module": module_name, "blocked_pattern": blocked},
                )

        return True

    def check_module_allowed(self, module_name: str) -> bool:
        """
        Prüft ob ein Modul in der Sandbox erlaubt ist.

        Args:
            module_name: Name des Moduls

        Returns:
            True wenn Modul erlaubt, False wenn blockiert
        """
        if not self.config.enabled:
            return True

        # Check if in safe list
        base_module = module_name.split(".")[0]
        if base_module in self.config.safe_modules:
            return True

        # Check if blocked
        for blocked in self.config.blocked_modules:
            if module_name == blocked or module_name.startswith(f"{blocked}."):
                return False

        # Default: allow if not explicitly blocked
        return True

    def check_network_access(self, host: str, port: int) -> bool:
        """
        Prüft ob Netzwerkzugriff erlaubt ist.

        Args:
            host: Ziel-Host
            port: Ziel-Port

        Returns:
            True wenn Zugriff erlaubt

        Raises:
            SecurityError: Bei nicht erlaubtem Netzwerkzugriff
        """
        if not self.config.enabled:
            return True

        if not self.config.network_enabled:
            raise SecurityError(
                f"Netzwerkzugriff nicht erlaubt: {host}:{port}",
                details={"host": host, "port": port},
            )

        return True

    def create_restricted_globals(self) -> dict[str, Any]:
        """
        Erstellt ein eingeschränktes globals-Dictionary für exec/eval.

        Returns:
            Dictionary mit erlaubten Builtins
        """
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }

        return {"__builtins__": safe_builtins}

    @staticmethod
    def _is_subpath(path: Path, parent: Path) -> bool:
        """Prüft ob path ein Unterpfad von parent ist."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def __enter__(self) -> "Sandbox":
        """Context Manager Entry."""
        self.logger.debug("Entering sandbox")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context Manager Exit."""
        self.logger.debug("Exiting sandbox")
