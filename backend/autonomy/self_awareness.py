#!/usr/bin/env python3
"""
SCIO - Self Awareness
Ermöglicht SCIO seine eigene Codebase zu verstehen

Features:
- Codebase-Scanner
- AST-Analyse (Abstract Syntax Tree)
- Struktur-Mapping
- Abhängigkeits-Analyse
- Worker/Route/Model Discovery
"""

import os
import ast
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CodeFile:
    """Repräsentiert eine Code-Datei"""
    path: str
    relative_path: str
    size: int
    last_modified: float
    hash: str
    language: str
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    docstring: str = ""


@dataclass
class WorkerInfo:
    """Informationen über einen Worker"""
    name: str
    file_path: str
    class_name: str
    methods: List[str]
    capabilities: List[str]
    models: List[str]


@dataclass
class RouteInfo:
    """Informationen über eine Route"""
    blueprint: str
    file_path: str
    endpoints: List[Dict[str, str]]


class SelfAwareness:
    """
    SCIO Self-Awareness System

    Scannt und analysiert die eigene Codebase um:
    - Struktur zu verstehen
    - Vorhandene Fähigkeiten zu erkennen
    - Abhängigkeiten zu identifizieren
    """

    def __init__(self):
        from backend.config import Config
        self.base_path = Path(getattr(Config, 'BASE_DIR', 'C:/SCIO'))
        self.backend_path = self.base_path / "backend"
        self.frontend_path = self.base_path / "frontend"
        self.data_path = self.base_path / "data"

        self._code_files: Dict[str, CodeFile] = {}
        self._workers: Dict[str, WorkerInfo] = {}
        self._routes: Dict[str, RouteInfo] = {}
        self._models: Dict[str, dict] = {}
        self._last_scan: Optional[datetime] = None

    def initialize(self) -> bool:
        """Initialisiert Self-Awareness mit initialem Scan"""
        try:
            self.scan_codebase()
            return True
        except Exception as e:
            print(f"[ERROR] Self-Awareness Init fehlgeschlagen: {e}")
            return False

    def scan_codebase(self) -> dict:
        """Scannt die gesamte Codebase"""
        self._code_files.clear()
        self._workers.clear()
        self._routes.clear()

        # Scan Python files
        python_files = list(self.backend_path.rglob("*.py"))

        for file_path in python_files:
            try:
                code_file = self._analyze_python_file(file_path)
                if code_file:
                    self._code_files[str(file_path)] = code_file
            except Exception as e:
                print(f"[WARN] Konnte {file_path} nicht analysieren: {e}")

        # Discover workers
        self._discover_workers()

        # Discover routes
        self._discover_routes()

        self._last_scan = datetime.now()

        return self.get_summary()

    def _analyze_python_file(self, file_path: Path) -> Optional[CodeFile]:
        """Analysiert eine Python-Datei"""
        try:
            content = file_path.read_text(encoding='utf-8')
            stat = file_path.stat()

            # Parse AST
            tree = ast.parse(content)

            # Extract information
            classes = []
            functions = []
            imports = []
            docstring = ast.get_docstring(tree) or ""

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return CodeFile(
                path=str(file_path),
                relative_path=str(file_path.relative_to(self.base_path)),
                size=stat.st_size,
                last_modified=stat.st_mtime,
                hash=hashlib.md5(content.encode()).hexdigest(),
                language="python",
                classes=classes,
                functions=list(set(functions)),
                imports=list(set(imports)),
                docstring=docstring[:500] if docstring else "",
            )

        except Exception as e:
            return None

    def _discover_workers(self):
        """Entdeckt alle Worker in der Codebase"""
        workers_path = self.backend_path / "workers"

        if not workers_path.exists():
            return

        for file_path in workers_path.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name == "base_worker.py":
                continue

            try:
                content = file_path.read_text(encoding='utf-8')
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if inherits from BaseWorker
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id == "BaseWorker":
                                worker_info = self._extract_worker_info(node, file_path, content)
                                if worker_info:
                                    self._workers[worker_info.name] = worker_info
                                break

            except Exception as e:
                print(f"[WARN] Worker Discovery für {file_path}: {e}")

    def _extract_worker_info(self, class_node: ast.ClassDef, file_path: Path, content: str) -> Optional[WorkerInfo]:
        """Extrahiert Worker-Informationen aus AST"""
        methods = []
        capabilities = []

        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)

                # Detect capabilities from method names
                if item.name.startswith("generate"):
                    capabilities.append("generation")
                elif item.name.startswith("process"):
                    capabilities.append("processing")
                elif item.name.startswith("train"):
                    capabilities.append("training")
                elif item.name.startswith("analyze"):
                    capabilities.append("analysis")
                elif item.name.startswith("convert"):
                    capabilities.append("conversion")

        # Find models from constants
        models = []
        if "_MODELS" in content or "MODELS = " in content:
            # Simple regex-like extraction
            for line in content.split("\n"):
                if "'" in line and ("model" in line.lower() or "hf_id" in line.lower()):
                    # Extract quoted strings
                    parts = line.split("'")
                    for i, part in enumerate(parts):
                        if "/" in part and i % 2 == 1:  # Hugging Face model ID
                            models.append(part)

        return WorkerInfo(
            name=class_node.name,
            file_path=str(file_path),
            class_name=class_node.name,
            methods=methods,
            capabilities=list(set(capabilities)),
            models=models[:10],  # Limit to first 10
        )

    def _discover_routes(self):
        """Entdeckt alle API Routes"""
        routes_path = self.backend_path / "routes"

        if not routes_path.exists():
            return

        for file_path in routes_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                content = file_path.read_text(encoding='utf-8')

                # Find Blueprint name
                blueprint_name = file_path.stem

                # Find endpoints
                endpoints = []
                lines = content.split("\n")

                for i, line in enumerate(lines):
                    if "@" in line and ".route(" in line:
                        # Extract route path
                        start = line.find("'") or line.find('"')
                        if start > 0:
                            end = line.find("'", start + 1) or line.find('"', start + 1)
                            if end > start:
                                path = line[start + 1:end]

                                # Find method
                                method = "GET"
                                if "methods=" in line:
                                    if "POST" in line:
                                        method = "POST"
                                    elif "PUT" in line:
                                        method = "PUT"
                                    elif "DELETE" in line:
                                        method = "DELETE"

                                # Find function name (next line)
                                if i + 1 < len(lines):
                                    next_line = lines[i + 1]
                                    if "def " in next_line:
                                        func_name = next_line.split("def ")[1].split("(")[0]
                                        endpoints.append({
                                            "path": path,
                                            "method": method,
                                            "function": func_name,
                                        })

                if endpoints:
                    self._routes[blueprint_name] = RouteInfo(
                        blueprint=blueprint_name,
                        file_path=str(file_path),
                        endpoints=endpoints,
                    )

            except Exception as e:
                print(f"[WARN] Route Discovery für {file_path}: {e}")

    def get_summary(self) -> dict:
        """Gibt Zusammenfassung der Codebase zurück"""
        return {
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "statistics": {
                "total_files": len(self._code_files),
                "total_classes": sum(len(f.classes) for f in self._code_files.values()),
                "total_functions": sum(len(f.functions) for f in self._code_files.values()),
                "total_workers": len(self._workers),
                "total_routes": len(self._routes),
                "total_endpoints": sum(len(r.endpoints) for r in self._routes.values()),
            },
            "workers": list(self._workers.keys()),
            "routes": list(self._routes.keys()),
        }

    def get_workers(self) -> Dict[str, WorkerInfo]:
        """Gibt alle entdeckten Worker zurück"""
        return self._workers

    def get_worker(self, name: str) -> Optional[WorkerInfo]:
        """Gibt spezifischen Worker zurück"""
        return self._workers.get(name)

    def get_routes(self) -> Dict[str, RouteInfo]:
        """Gibt alle entdeckten Routes zurück"""
        return self._routes

    def get_file(self, path: str) -> Optional[CodeFile]:
        """Gibt Informationen über eine Datei zurück"""
        return self._code_files.get(path)

    def get_all_classes(self) -> List[str]:
        """Gibt alle Klassen in der Codebase zurück"""
        classes = []
        for file in self._code_files.values():
            classes.extend(file.classes)
        return classes

    def get_all_imports(self) -> List[str]:
        """Gibt alle verwendeten Imports zurück"""
        imports = set()
        for file in self._code_files.values():
            imports.update(file.imports)
        return list(imports)

    def read_file(self, path: str) -> Optional[str]:
        """Liest Inhalt einer Datei"""
        try:
            full_path = Path(path)
            if not full_path.is_absolute():
                full_path = self.base_path / path

            return full_path.read_text(encoding='utf-8')
        except Exception:
            return None

    def file_exists(self, path: str) -> bool:
        """Prüft ob Datei existiert"""
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = self.base_path / path
        return full_path.exists()

    def get_structure(self) -> dict:
        """Gibt vollständige Codebase-Struktur zurück"""
        structure = {
            "backend": {},
            "frontend": {},
            "data": {},
        }

        for file_path, code_file in self._code_files.items():
            parts = code_file.relative_path.split(os.sep)
            if len(parts) > 1:
                section = parts[0]
                if section in structure:
                    structure[section][code_file.relative_path] = {
                        "classes": code_file.classes,
                        "functions": code_file.functions[:10],
                        "docstring": code_file.docstring[:200],
                    }

        return structure


# Singleton
_self_awareness: Optional[SelfAwareness] = None

def get_self_awareness() -> SelfAwareness:
    """Gibt Singleton-Instanz zurück"""
    global _self_awareness
    if _self_awareness is None:
        _self_awareness = SelfAwareness()
    return _self_awareness
