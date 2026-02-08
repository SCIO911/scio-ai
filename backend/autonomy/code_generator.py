#!/usr/bin/env python3
"""
SCIO - Code Generator
Generiert neuen Code für SCIO selbst

WICHTIG: Verwendet IMMER echte KI (LLM) für Code-Generierung.
KEINE Templates, KEINE Platzhalter - nur echter KI-generierter Code.
"""

import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class GeneratedCode:
    """Generiertes Code-Paket"""
    file_path: str
    content: str
    description: str
    imports_needed: List[str]
    dependencies: List[str]
    tests: Optional[str] = None


class CodeGenerator:
    """
    SCIO Code Generator

    Verwendet den AI Programmer für ECHTE KI-basierte Code-Generierung.
    KEINE Templates - alle Code-Generierung erfolgt durch LLM.
    """

    def __init__(self):
        from backend.config import Config
        self.self_awareness = None
        self.memory = None
        self.base_path = Path(getattr(Config, 'BASE_DIR', 'C:/SCIO'))
        self._generated_files: List[GeneratedCode] = []
        self._ai_programmer = None

    def _get_ai_programmer(self):
        """Lazy-Load des AI Programmers"""
        if self._ai_programmer is None:
            from backend.autonomy.ai_programmer import get_ai_programmer
            self._ai_programmer = get_ai_programmer()
            if not self._ai_programmer._initialized:
                self._ai_programmer.initialize()
        return self._ai_programmer

    def initialize(self, self_awareness, memory) -> bool:
        """Initialisiert den Code Generator"""
        self.self_awareness = self_awareness
        self.memory = memory
        return True

    def generate_worker(
        self,
        name: str,
        description: str,
        capabilities: List[str],
        models: List[Dict[str, str]] = None,
    ) -> GeneratedCode:
        """
        Generiert einen neuen Worker mit ECHTER KI.

        Args:
            name: Worker-Name (z.B. "Translation")
            description: Beschreibung
            capabilities: Liste von Fähigkeiten
            models: Liste von Modellen
        """
        worker_name_lower = name.lower().replace(" ", "_")

        # Baue detaillierte Anforderungen für den AI Programmer
        requirements = [
            f"Worker-Name: {name}",
            f"Beschreibung: {description}",
            "Muss von BaseWorker erben",
            "Muss initialize(), process(), cleanup() implementieren",
            "Singleton-Pattern mit get_*_worker() Funktion",
            "GPU-Support mit torch.cuda.is_available()",
            "Thread-Safety mit Lock",
        ]

        for cap in capabilities:
            requirements.append(f"Fähigkeit: {cap} - vollständig implementiert")

        if models:
            models_desc = ", ".join([f"{m['name']}: {m.get('hf_id', 'custom')}" for m in models])
            requirements.append(f"Unterstützte Modelle: {models_desc}")

        # AI Programmer für echte Code-Generierung verwenden
        from backend.autonomy.ai_programmer import ProgrammingTask

        task = ProgrammingTask(
            description=f"Erstelle einen vollständigen {name} Worker für SCIO. {description}",
            requirements=requirements,
            constraints=[
                "KEIN Platzhalter-Code",
                "KEINE pass-Statements ohne Implementierung",
                "KEINE # TODO Kommentare",
                "Vollständig funktionsfähiger Code",
            ],
            target_file=f"backend/workers/{worker_name_lower}_worker.py",
            context_files=[
                "backend/workers/base_worker.py",
                "backend/workers/llm_inference.py",
            ],
        )

        ai = self._get_ai_programmer()
        program = ai.write_program(task, auto_deploy=False, require_tests=True)

        generated = GeneratedCode(
            file_path=task.target_file,
            content=program.code,
            description=f"Worker für {description}",
            imports_needed=["torch", "transformers"],
            dependencies=[],
        )

        self._generated_files.append(generated)

        if self.memory:
            self.memory.log_event(
                "code_generation",
                f"Generated worker: {name}",
                {"file": task.target_file, "capabilities": capabilities, "quality": program.quality.value}
            )

        return generated

    def generate_route(
        self,
        name: str,
        description: str,
        endpoints: List[Dict[str, str]],
    ) -> GeneratedCode:
        """
        Generiert neue API Routes mit ECHTER KI.

        Args:
            name: Route-Name (z.B. "Translation")
            description: Beschreibung
            endpoints: Liste von Endpoints [{path, method, function, description}]
        """
        route_name_lower = name.lower().replace(" ", "_")

        # Baue Endpoint-Beschreibungen
        endpoint_reqs = []
        for ep in endpoints:
            ep_desc = (
                f"Endpoint: {ep.get('method', 'POST')} {ep.get('path', '/')} - "
                f"{ep.get('description', '')} (Funktion: {ep.get('function', 'handler')})"
            )
            endpoint_reqs.append(ep_desc)

        requirements = [
            f"Route-Name: {name}",
            f"Beschreibung: {description}",
            "Flask Blueprint verwenden",
            "@require_api_key Decorator für Authentifizierung",
            "Vollständige Error-Handling mit try/except",
            "JSON Response mit status-Feld",
            "Job-Queue Integration für asynchrone Verarbeitung",
        ] + endpoint_reqs

        from backend.autonomy.ai_programmer import ProgrammingTask

        task = ProgrammingTask(
            description=f"Erstelle Flask Routes für {name}. {description}",
            requirements=requirements,
            constraints=[
                "Vollständige Implementierung aller Endpoints",
                "Korrekte HTTP Status Codes",
                "Input-Validierung für alle Endpoints",
            ],
            target_file=f"backend/routes/{route_name_lower}.py",
            context_files=[
                "backend/routes/inference.py",
                "backend/services/api_keys.py",
            ],
        )

        ai = self._get_ai_programmer()
        program = ai.write_program(task, auto_deploy=False, require_tests=True)

        generated = GeneratedCode(
            file_path=task.target_file,
            content=program.code,
            description=f"API Routes für {description}",
            imports_needed=["flask"],
            dependencies=[],
        )

        self._generated_files.append(generated)
        return generated

    def generate_service(
        self,
        name: str,
        description: str,
        methods: List[Dict[str, str]],
        init_vars: List[str] = None,
    ) -> GeneratedCode:
        """
        Generiert einen neuen Service mit ECHTER KI.

        Args:
            name: Service-Name
            description: Beschreibung
            methods: Liste von Methoden [{name, description, params, returns}]
            init_vars: Initialisierungs-Variablen
        """
        service_name_lower = name.lower().replace(" ", "_")

        # Baue Methoden-Beschreibungen
        method_reqs = []
        for m in methods:
            m_desc = (
                f"Methode: {m.get('name', 'method')}({m.get('params', '')}) -> {m.get('returns', 'Any')} - "
                f"{m.get('description', '')}"
            )
            method_reqs.append(m_desc)

        requirements = [
            f"Service-Name: {name}",
            f"Beschreibung: {description}",
            "Thread-Safety mit threading.Lock",
            "Singleton-Pattern mit get_*_service() Funktion",
            "initialize() und shutdown() Methoden",
            "Vollständige Implementierung aller Methoden",
        ] + method_reqs

        if init_vars:
            requirements.append(f"Initialisierungs-Variablen: {', '.join(init_vars)}")

        from backend.autonomy.ai_programmer import ProgrammingTask

        task = ProgrammingTask(
            description=f"Erstelle einen vollständigen {name} Service. {description}",
            requirements=requirements,
            constraints=[
                "Keine pass-Statements",
                "Echte Implementierung für alle Methoden",
                "Fehlerbehandlung in allen Methoden",
            ],
            target_file=f"backend/services/{service_name_lower}.py",
            context_files=[
                "backend/services/hardware_monitor.py",
                "backend/services/job_queue.py",
            ],
        )

        ai = self._get_ai_programmer()
        program = ai.write_program(task, auto_deploy=False, require_tests=True)

        generated = GeneratedCode(
            file_path=task.target_file,
            content=program.code,
            description=f"Service für {description}",
            imports_needed=[],
            dependencies=[],
        )

        self._generated_files.append(generated)
        return generated

    def generate_capability(self, capability_name: str, capability_info: dict) -> Optional[GeneratedCode]:
        """
        Generiert Code für eine fehlende Fähigkeit.

        Args:
            capability_name: Name der Fähigkeit
            capability_info: Info über die Fähigkeit

        Returns:
            GeneratedCode oder None
        """
        category = capability_info.get("category", "")
        description = capability_info.get("description", "")

        if category in ["llm", "image", "audio", "video", "code", "vision", "embeddings", "document", "3d"]:
            return self._generate_worker_extension(capability_name, description, category)
        elif category == "automation":
            return self.generate_service(
                name=capability_name.replace("_", " ").title(),
                description=description,
                methods=[{"name": capability_name, "description": description}],
            )
        elif category == "integration":
            return self._generate_integration(capability_name, description)

        return None

    def _generate_worker_extension(self, capability: str, description: str, category: str) -> GeneratedCode:
        """Generiert eine Worker-Erweiterung mit ECHTER KI."""
        worker_mapping = {
            "llm": "llm_inference",
            "image": "image_gen",
            "audio": "audio_worker",
            "video": "video_worker",
            "code": "code_worker",
            "vision": "vision_worker",
            "embeddings": "embedding_worker",
            "document": "document_worker",
            "3d": "threed_worker",
        }

        worker_file = worker_mapping.get(category, "base_worker")

        from backend.autonomy.ai_programmer import ProgrammingTask

        task = ProgrammingTask(
            description=f"Erweitere den {worker_file} Worker um die Fähigkeit: {capability}. {description}",
            requirements=[
                f"Neue Methode: {capability}",
                "Vollständige Implementierung",
                "Kompatibel mit existierendem Worker",
                "Dokumentation (Docstring)",
            ],
            target_file=f"backend/workers/{worker_file}.py",
            modify_existing=True,
        )

        ai = self._get_ai_programmer()
        program = ai.write_program(task, auto_deploy=False)

        return GeneratedCode(
            file_path=task.target_file,
            content=program.code,
            description=f"Erweiterung für {description}",
            imports_needed=[],
            dependencies=[],
        )

    def _generate_integration(self, name: str, description: str) -> GeneratedCode:
        """Generiert eine Integration mit ECHTER KI."""
        integration_name = name.replace("integration_", "")

        from backend.autonomy.ai_programmer import ProgrammingTask

        task = ProgrammingTask(
            description=f"Erstelle eine {integration_name} Integration für SCIO. {description}",
            requirements=[
                f"Integration-Name: {integration_name}",
                "API-Key aus Umgebungsvariablen",
                "Enable/Disable via Umgebungsvariable",
                "httpx für HTTP Requests",
                "get() und post() Methoden",
                "get_status() Methode",
                "Singleton-Pattern",
                "Vollständige Fehlerbehandlung",
            ],
            target_file=f"backend/integrations/{integration_name}.py",
        )

        ai = self._get_ai_programmer()
        program = ai.write_program(task, auto_deploy=False)

        return GeneratedCode(
            file_path=task.target_file,
            content=program.code,
            description=f"Integration für {description}",
            imports_needed=["httpx"],
            dependencies=[],
        )

    def save_generated(self, generated: GeneratedCode, backup: bool = True) -> bool:
        """
        Speichert generierten Code.

        Args:
            generated: GeneratedCode Objekt
            backup: Backup erstellen wenn Datei existiert

        Returns:
            True wenn erfolgreich
        """
        try:
            file_path = self.base_path / generated.file_path

            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f".py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(generated.content, encoding='utf-8')

            if self.memory:
                self.memory.log_event(
                    "code_saved",
                    f"Saved generated code: {generated.file_path}",
                    {"backup": backup}
                )

            return True

        except Exception as e:
            print(f"[ERROR] Code speichern fehlgeschlagen: {e}")
            return False

    def get_generated_files(self) -> List[GeneratedCode]:
        """Gibt alle generierten Dateien zurück"""
        return self._generated_files

    def clear_generated(self):
        """Löscht generierte Dateien aus dem Cache"""
        self._generated_files.clear()

    def fix_syntax(self, code: str, error_info: dict) -> dict:
        """
        Repariert Syntax-Fehler im Code mit ECHTER KI.

        Args:
            code: Der Code mit Syntax-Fehler
            error_info: Dict mit line, offset, message

        Returns:
            dict mit success und code
        """
        from backend.autonomy.ai_programmer import ProgrammingTask

        error_line = error_info.get('line', 1)
        error_msg = error_info.get('message', 'Syntax error')

        task = ProgrammingTask(
            description=f"Repariere den Syntax-Fehler in Zeile {error_line}: {error_msg}",
            existing_code=code,
            modify_existing=True,
            constraints=[
                "Nur den Fehler beheben",
                "Keine anderen Änderungen",
                "Gleiche Funktionalität beibehalten",
            ],
        )

        try:
            ai = self._get_ai_programmer()
            program = ai.write_program(task, auto_deploy=False)

            if program.test_results.get('syntax_ok', False):
                return {'success': True, 'code': program.code}
            else:
                return {'success': False, 'code': code, 'error': 'Could not fix syntax'}

        except Exception as e:
            return {'success': False, 'code': code, 'error': str(e)}

    def generate_method(self, method_name: str, class_name: str = None,
                        parameters: List[str] = None, return_type: str = 'None',
                        docstring: str = None) -> dict:
        """
        Generiert eine Python-Methode mit ECHTER KI.

        Args:
            method_name: Name der Methode
            class_name: Optionaler Klassenname
            parameters: Liste der Parameter
            return_type: Rückgabetyp
            docstring: Dokumentation

        Returns:
            dict mit success und code
        """
        from backend.autonomy.ai_programmer import ProgrammingTask

        params_str = ", ".join(parameters) if parameters else ""
        desc = docstring or f"{method_name} method"

        if class_name:
            task_desc = f"Erstelle eine Instanz-Methode {method_name} für die Klasse {class_name}"
        else:
            task_desc = f"Erstelle eine Funktion {method_name}"

        task = ProgrammingTask(
            description=f"{task_desc}. {desc}",
            requirements=[
                f"Methodenname: {method_name}",
                f"Parameter: {params_str}" if params_str else "Keine Parameter",
                f"Rückgabetyp: {return_type}",
                "Vollständige Implementierung",
                "Keine pass-Statements",
            ],
        )

        try:
            ai = self._get_ai_programmer()
            program = ai.write_program(task, auto_deploy=False)

            return {'success': True, 'code': program.code, 'method_name': method_name}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def insert_method(self, file_path: str, class_name: str, method_code: str) -> dict:
        """
        Fügt eine Methode in eine bestehende Klasse ein.

        Args:
            file_path: Pfad zur Python-Datei
            class_name: Name der Klasse
            method_code: Der Methoden-Code

        Returns:
            dict mit success
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {'success': False, 'error': f'File not found: {file_path}'}

            content = path.read_text(encoding='utf-8')

            # Find the class definition
            class_pattern = rf'class\s+{class_name}\s*[:\(]'
            match = re.search(class_pattern, content)

            if not match:
                return {'success': False, 'error': f'Class not found: {class_name}'}

            # Find the end of the class
            class_start = match.start()
            lines = content.split('\n')

            # Find line number of class definition
            char_count = 0
            class_line = 0
            for i, line in enumerate(lines):
                if char_count >= class_start:
                    class_line = i
                    break
                char_count += len(line) + 1

            # Find class indentation
            class_indent = len(lines[class_line]) - len(lines[class_line].lstrip())

            # Find last method in class
            insert_line = class_line + 1
            for i in range(class_line + 1, len(lines)):
                line = lines[i]
                if line.strip():
                    line_indent = len(line) - len(line.lstrip())
                    if line_indent <= class_indent and not line.strip().startswith('#'):
                        insert_line = i
                        break
                    if line.lstrip().startswith('def '):
                        insert_line = i + 1
                        for j in range(i + 1, len(lines)):
                            next_line = lines[j]
                            if next_line.strip():
                                next_indent = len(next_line) - len(next_line.lstrip())
                                if next_indent <= class_indent + 4:
                                    insert_line = j
                                    break
                                insert_line = j + 1

            # Insert the method code
            method_lines = method_code.split('\n')
            indented_method = []
            for line in method_lines:
                if line.strip():
                    indented_method.append(' ' * (class_indent + 4) + line.lstrip())
                else:
                    indented_method.append('')

            lines.insert(insert_line, '')
            for i, method_line in enumerate(indented_method):
                lines.insert(insert_line + 1 + i, method_line)

            new_content = '\n'.join(lines)
            path.write_text(new_content, encoding='utf-8')

            if self.memory:
                self.memory.log_event(
                    "method_inserted",
                    f"Inserted method into {class_name} in {file_path}"
                )

            return {'success': True}

        except Exception as e:
            return {'success': False, 'error': str(e)}


# Singleton
_code_generator: Optional[CodeGenerator] = None

def get_code_generator() -> CodeGenerator:
    """Gibt Singleton-Instanz zurück"""
    global _code_generator
    if _code_generator is None:
        _code_generator = CodeGenerator()
    return _code_generator
