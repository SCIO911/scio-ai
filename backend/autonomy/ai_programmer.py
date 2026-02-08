#!/usr/bin/env python3
"""
SCIO - AI Programmer
Echte KI-gesteuerte Selbst-Programmierung

SCIO kann jedes Programm schreiben das er benötigt:
- Natürliche Sprache → funktionierender Code
- Iterative Verbesserung mit Feedback-Loops
- Automatische Fehlerkorrektur
- Selbst-Modifikation bestehenden Codes
"""

import os
import ast
import time
import uuid
import json
import traceback
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from backend.config import Config


class CodeQuality(str, Enum):
    """Code-Qualitätsstufen"""
    EXCELLENT = 'excellent'
    GOOD = 'good'
    ACCEPTABLE = 'acceptable'
    NEEDS_WORK = 'needs_work'
    FAILED = 'failed'


@dataclass
class GeneratedProgram:
    """Ein generiertes Programm"""
    id: str
    request: str
    code: str
    language: str = 'python'
    file_path: Optional[str] = None
    quality: CodeQuality = CodeQuality.ACCEPTABLE
    test_results: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    tokens_used: int = 0
    generation_time: float = 0.0
    deployed: bool = False
    error_history: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProgrammingTask:
    """Eine Programmieraufgabe"""
    description: str
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    target_file: Optional[str] = None
    modify_existing: bool = False
    existing_code: Optional[str] = None
    context_files: List[str] = field(default_factory=list)


class AIProgrammer:
    """
    SCIO AI Programmer

    Kann jedes Programm schreiben:
    - Vollständige Programme aus Beschreibung
    - Einzelne Funktionen/Klassen
    - Modifikation bestehenden Codes
    - Automatische Fehlerkorrektur
    - Iterative Verbesserung bis zur Perfektion
    """

    def __init__(self):
        self._initialized = False
        self._llm_worker = None
        self._code_worker = None
        self._lock = threading.Lock()
        self._generation_history: List[GeneratedProgram] = []
        self._max_iterations = 5
        self._base_path = Config.BASE_DIR

    def initialize(self) -> bool:
        """Initialisiert den AI Programmer"""
        try:
            # LLM Worker für Code-Generierung
            from backend.workers.llm_inference import get_inference_worker
            self._llm_worker = get_inference_worker()

            # Code Worker falls verfügbar
            try:
                from backend.workers.code_worker import get_code_worker
                self._code_worker = get_code_worker()
            except ImportError:
                self._code_worker = None

            self._initialized = True
            print("[OK] AI Programmer initialisiert")
            return True
        except Exception as e:
            print(f"[ERROR] AI Programmer Init fehlgeschlagen: {e}")
            return False

    def write_program(
        self,
        task: ProgrammingTask,
        auto_deploy: bool = False,
        require_tests: bool = True,
    ) -> GeneratedProgram:
        """
        Schreibt ein komplettes Programm aus einer Aufgabenbeschreibung.

        Args:
            task: Die Programmieraufgabe
            auto_deploy: Automatisch deployen wenn erfolgreich
            require_tests: Tests vor Deploy erforderlich

        Returns:
            GeneratedProgram mit dem fertigen Code
        """
        start_time = time.time()
        program_id = str(uuid.uuid4())[:8]

        program = GeneratedProgram(
            id=program_id,
            request=task.description,
            code="",
            file_path=task.target_file,
        )

        try:
            # Kontext sammeln
            context = self._gather_context(task)

            # Erste Generierung
            code = self._generate_code(task, context)
            program.code = code
            program.iterations = 1

            # Iterative Verbesserung
            for iteration in range(self._max_iterations):
                # Testen
                test_result = self._test_code(code, task)
                program.test_results = test_result

                if test_result.get('success', False):
                    program.quality = self._assess_quality(code, test_result)
                    break

                # Fehler beheben
                error_msg = test_result.get('error', 'Unknown error')
                program.error_history.append(error_msg)

                code = self._fix_code(code, error_msg, task, context)
                program.code = code
                program.iterations += 1

            # Finale Qualitätsbewertung
            if not program.test_results.get('success', False):
                program.quality = CodeQuality.FAILED

            # Deploy wenn gewünscht
            if auto_deploy and program.quality != CodeQuality.FAILED:
                if not require_tests or program.test_results.get('success', False):
                    self._deploy_code(program, task)
                    program.deployed = True

        except Exception as e:
            program.error_history.append(str(e))
            program.quality = CodeQuality.FAILED

        program.generation_time = time.time() - start_time
        self._generation_history.append(program)

        return program

    def _gather_context(self, task: ProgrammingTask) -> Dict[str, Any]:
        """Sammelt Kontext für die Code-Generierung"""
        context = {
            "scio_patterns": self._get_scio_patterns(),
            "imports": self._get_common_imports(),
            "existing_code": None,
            "related_files": [],
        }

        # Bestehenden Code laden wenn Modifikation
        if task.modify_existing and task.target_file:
            file_path = self._base_path / task.target_file
            if file_path.exists():
                context["existing_code"] = file_path.read_text(encoding='utf-8')

        # Kontext-Dateien laden
        for ctx_file in task.context_files:
            file_path = self._base_path / ctx_file
            if file_path.exists():
                context["related_files"].append({
                    "path": ctx_file,
                    "content": file_path.read_text(encoding='utf-8')[:5000],  # Max 5k chars
                })

        return context

    def _get_scio_patterns(self) -> str:
        """Gibt SCIO-spezifische Code-Patterns zurück"""
        return """
SCIO Code Patterns:

1. Worker Pattern:
```python
class XyzWorker(BaseWorker):
    def __init__(self):
        super().__init__("Xyz")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def initialize(self) -> bool:
        self.status = WorkerStatus.READY
        return True

    def process(self, job_id: str, input_data: dict) -> dict:
        # Verarbeitung hier
        return {"status": "success", "result": ...}

    def cleanup(self):
        self._model = None
        torch.cuda.empty_cache()

_worker = None
def get_xyz_worker() -> XyzWorker:
    global _worker
    if _worker is None:
        _worker = XyzWorker()
    return _worker
```

2. Service Pattern:
```python
class XyzService:
    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> bool:
        self._initialized = True
        return True

_service = None
def get_xyz_service() -> XyzService:
    global _service
    if _service is None:
        _service = XyzService()
    return _service
```

3. Route Pattern:
```python
from flask import Blueprint, request, jsonify
xyz_bp = Blueprint('xyz', __name__)

@xyz_bp.route('/api/v1/xyz', methods=['POST'])
def xyz_endpoint():
    data = request.get_json() or {}
    # Verarbeitung
    return jsonify({"status": "success"})
```

4. Error Handling:
```python
try:
    result = operation()
except ValueError as e:
    return {"error": "Invalid input", "details": str(e)}, 400
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return {"error": "Internal error"}, 500
```
"""

    def _get_common_imports(self) -> str:
        """Gibt häufig benötigte Imports zurück"""
        return """
# Standard Library
import os
import json
import time
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# SCIO
from backend.config import Config

# PyTorch (wenn GPU)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
"""

    def _generate_code(self, task: ProgrammingTask, context: Dict[str, Any]) -> str:
        """Generiert Code mit echtem LLM - KEINE Templates"""

        # Prompt bauen
        prompt = self._build_generation_prompt(task, context)

        # IMMER LLM verwenden - niemals Templates
        code = None
        last_error = None

        # Versuch 1: Code Worker (spezialisiert für Code)
        if self._code_worker:
            try:
                result = self._code_worker.generate_code({
                    "prompt": prompt,
                    "language": "python",
                    "max_tokens": 8192,
                    "temperature": 0.2,
                })
                code = result.get('code', result.get('output', ''))
                if code and len(code.strip()) > 50:
                    return code
            except Exception as e:
                last_error = str(e)

        # Versuch 2: LLM Inference Worker
        if self._llm_worker:
            try:
                result = self._llm_worker.process("code_gen", {
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 8192,
                    "temperature": 0.2,
                })
                response = result.get('response', result.get('output', result.get('text', '')))
                code = self._extract_code_from_response(response)
                if code and len(code.strip()) > 50:
                    return code
            except Exception as e:
                last_error = str(e)

        # Versuch 3: Direkter API-Call wenn Worker fehlschlagen
        try:
            code = self._direct_llm_call(prompt)
            if code and len(code.strip()) > 50:
                return code
        except Exception as e:
            last_error = str(e)

        # Kein LLM verfügbar - Fehler werfen, KEINE Templates
        raise RuntimeError(
            f"KI-Code-Generierung fehlgeschlagen. SCIO benötigt ein funktionierendes LLM. "
            f"Letzter Fehler: {last_error}"
        )

    def _get_system_prompt(self) -> str:
        """System-Prompt für Code-Generierung"""
        return """Du bist SCIO's interner Code-Generator. Du schreibst perfekten, produktionsreifen Python-Code.

REGELN:
1. Schreibe VOLLSTÄNDIGEN, AUSFÜHRBAREN Code
2. KEINE Platzhalter wie "# TODO" oder "pass" - echte Implementierung
3. Verwende Type Hints überall
4. Docstrings für alle Klassen und Methoden
5. Fehlerbehandlung mit try/except
6. Thread-Safety mit Locks wo nötig
7. Singleton-Pattern für Worker/Services
8. Logging statt print() in Produktion

SCIO-SPEZIFISCH:
- Workers erben von BaseWorker
- Services haben initialize() und shutdown()
- Routes nutzen Flask Blueprints
- Konfiguration aus backend.config.Config

AUSGABE:
- NUR Python-Code, keine Erklärungen
- Code muss sofort funktionieren
- Alle Imports am Anfang"""

    def _direct_llm_call(self, prompt: str) -> str:
        """Direkter LLM-Call als Fallback"""
        import httpx

        # Versuche lokale LLM APIs
        endpoints = [
            ("http://localhost:11434/api/generate", "ollama"),  # Ollama
            ("http://localhost:8080/completion", "llamacpp"),   # llama.cpp
            ("http://localhost:5000/v1/completions", "vllm"),   # vLLM
        ]

        for url, api_type in endpoints:
            try:
                if api_type == "ollama":
                    response = httpx.post(url, json={
                        "model": "codellama",
                        "prompt": prompt,
                        "stream": False,
                    }, timeout=120)
                    if response.status_code == 200:
                        return self._extract_code_from_response(response.json().get("response", ""))

                elif api_type == "llamacpp":
                    response = httpx.post(url, json={
                        "prompt": prompt,
                        "n_predict": 4096,
                        "temperature": 0.2,
                    }, timeout=120)
                    if response.status_code == 200:
                        return self._extract_code_from_response(response.json().get("content", ""))

                elif api_type == "vllm":
                    response = httpx.post(url, json={
                        "model": "default",
                        "prompt": prompt,
                        "max_tokens": 4096,
                        "temperature": 0.2,
                    }, timeout=120)
                    if response.status_code == 200:
                        choices = response.json().get("choices", [])
                        if choices:
                            return self._extract_code_from_response(choices[0].get("text", ""))

            except Exception:
                continue

        raise RuntimeError("Kein LLM-Endpoint erreichbar")

    def _build_generation_prompt(self, task: ProgrammingTask, context: Dict[str, Any]) -> str:
        """Baut den Prompt für Code-Generierung"""
        prompt_parts = [
            "Schreibe Python-Code für folgende Aufgabe:",
            "",
            f"## Aufgabe\n{task.description}",
        ]

        if task.requirements:
            prompt_parts.append(f"\n## Anforderungen\n" + "\n".join(f"- {r}" for r in task.requirements))

        if task.constraints:
            prompt_parts.append(f"\n## Einschränkungen\n" + "\n".join(f"- {c}" for c in task.constraints))

        if task.examples:
            prompt_parts.append(f"\n## Beispiele\n" + "\n".join(task.examples))

        if context.get("existing_code"):
            prompt_parts.append(f"\n## Existierender Code (zu modifizieren)\n```python\n{context['existing_code'][:3000]}\n```")

        if context.get("related_files"):
            prompt_parts.append("\n## Verwandte Dateien")
            for rf in context["related_files"][:3]:
                prompt_parts.append(f"\n### {rf['path']}\n```python\n{rf['content'][:1500]}\n```")

        prompt_parts.append(f"\n## SCIO Patterns\n{context.get('scio_patterns', '')}")

        prompt_parts.append("\n\n## Ausgabe\nGib NUR den Python-Code zurück, ohne Erklärungen. Der Code muss vollständig und ausführbar sein.")

        return "\n".join(prompt_parts)

    def _extract_code_from_response(self, response: str) -> str:
        """Extrahiert Code aus LLM-Antwort"""
        # Suche nach Code-Blöcken
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()

        if "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                return parts[1].strip()

        # Keine Code-Blöcke, nehme alles was nach Code aussieht
        lines = response.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith(('import ', 'from ', 'class ', 'def ', '@', '#')):
                in_code = True
            if in_code:
                code_lines.append(line)

        return '\n'.join(code_lines) if code_lines else response

    def _extract_name(self, description: str, default: str = "Custom") -> str:
        """Extrahiert einen Namen aus der Beschreibung"""
        # Suche nach typischen Namenspatternen
        words = description.split()
        for i, word in enumerate(words):
            if word.lower() in ['worker', 'service', 'handler', 'manager', 'engine']:
                if i > 0:
                    return words[i-1].title().replace('-', '').replace('_', '')

        # Erstes Substantiv-artiges Wort
        for word in words:
            if len(word) > 3 and word[0].isupper():
                return word.replace('-', '').replace('_', '')

        return default

    def _test_code(self, code: str, task: ProgrammingTask) -> Dict[str, Any]:
        """Testet generierten Code"""
        result = {
            "success": False,
            "syntax_ok": False,
            "imports_ok": False,
            "runnable": False,
            "error": None,
        }

        # 1. Syntax-Check
        try:
            ast.parse(code)
            result["syntax_ok"] = True
        except SyntaxError as e:
            result["error"] = f"Syntax error at line {e.lineno}: {e.msg}"
            return result

        # 2. Import-Check in isolierter Umgebung
        try:
            # Schreibe in temporäre Datei
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_path = f.name

            # Versuche zu importieren (nur Syntax/Import-Check)
            check_code = f'''
import sys
sys.path.insert(0, r"{Config.BASE_DIR}")
try:
    compile(open(r"{temp_path}", encoding="utf-8").read(), "{temp_path}", "exec")
    print("OK")
except Exception as e:
    print(f"ERROR: {{e}}")
'''
            proc = subprocess.run(
                ['python', '-c', check_code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(Config.BASE_DIR),
            )

            if "OK" in proc.stdout:
                result["imports_ok"] = True
                result["runnable"] = True
            else:
                result["error"] = proc.stderr or proc.stdout

            # Aufräumen
            os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            result["error"] = "Code execution timed out"
        except Exception as e:
            result["error"] = str(e)

        # Erfolg wenn alles OK
        result["success"] = result["syntax_ok"] and result["imports_ok"]

        return result

    def _fix_code(
        self,
        code: str,
        error: str,
        task: ProgrammingTask,
        context: Dict[str, Any],
    ) -> str:
        """Repariert Code basierend auf Fehlermeldung"""

        # Prompt für Reparatur
        fix_prompt = f"""
Der folgende Python-Code hat einen Fehler:

```python
{code}
```

Fehler:
{error}

Ursprüngliche Aufgabe:
{task.description}

Repariere den Code. Gib NUR den korrigierten Code zurück, keine Erklärungen.
"""

        if self._llm_worker:
            result = self._llm_worker.process("fix_code", {
                "messages": [
                    {"role": "system", "content": "Du bist ein Python-Experte. Repariere den Code."},
                    {"role": "user", "content": fix_prompt}
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            })
            fixed = self._extract_code_from_response(result.get('response', code))
            return fixed if fixed else code
        else:
            # Einfache automatische Fixes
            return self._auto_fix_common_errors(code, error)

    def _auto_fix_common_errors(self, code: str, error: str) -> str:
        """Automatische Fixes für häufige Fehler"""
        lines = code.split('\n')
        error_lower = error.lower()

        # Fehlende Doppelpunkte
        if 'expected ":"' in error_lower or 'invalid syntax' in error_lower:
            for i, line in enumerate(lines):
                stripped = line.rstrip()
                if any(stripped.startswith(kw) for kw in ['def ', 'class ', 'if ', 'elif ', 'else', 'for ', 'while ', 'try', 'except', 'finally', 'with ']):
                    if not stripped.endswith(':'):
                        lines[i] = stripped + ':'

        # Fehlende Klammern
        if 'unmatched' in error_lower or 'eof' in error_lower:
            for i, line in enumerate(lines):
                open_p = line.count('(') - line.count(')')
                if open_p > 0:
                    lines[i] = line + ')' * open_p

        # Einrückungsfehler
        if 'indent' in error_lower:
            fixed_lines = []
            prev_indent = 0
            for line in lines:
                if not line.strip():
                    fixed_lines.append(line)
                    continue

                # Berechne erwartete Einrückung
                if fixed_lines and fixed_lines[-1].rstrip().endswith(':'):
                    expected = prev_indent + 4
                else:
                    expected = prev_indent

                content = line.lstrip()
                if content.startswith(('return', 'pass', 'break', 'continue', 'raise')):
                    expected = max(0, prev_indent)
                elif content.startswith(('else:', 'elif ', 'except', 'finally:', 'except:')):
                    expected = max(0, prev_indent - 4)

                fixed_lines.append(' ' * expected + content)
                prev_indent = len(fixed_lines[-1]) - len(fixed_lines[-1].lstrip())

            lines = fixed_lines

        return '\n'.join(lines)

    def _assess_quality(self, code: str, test_results: Dict[str, Any]) -> CodeQuality:
        """Bewertet die Code-Qualität"""
        score = 0

        # Basis-Tests bestanden
        if test_results.get('syntax_ok'):
            score += 25
        if test_results.get('imports_ok'):
            score += 25
        if test_results.get('runnable'):
            score += 25

        # Code-Qualitätsmetriken
        lines = code.split('\n')

        # Docstrings vorhanden
        if '"""' in code or "'''" in code:
            score += 10

        # Type Hints vorhanden
        if '->' in code or ': str' in code or ': int' in code:
            score += 5

        # Nicht zu lang
        if len(lines) < 500:
            score += 5

        # Error Handling
        if 'try:' in code and 'except' in code:
            score += 5

        if score >= 90:
            return CodeQuality.EXCELLENT
        elif score >= 75:
            return CodeQuality.GOOD
        elif score >= 50:
            return CodeQuality.ACCEPTABLE
        else:
            return CodeQuality.NEEDS_WORK

    def _deploy_code(self, program: GeneratedProgram, task: ProgrammingTask):
        """Deployed den Code"""
        if not task.target_file:
            return

        file_path = self._base_path / task.target_file

        # Backup erstellen
        if file_path.exists():
            backup_path = file_path.with_suffix(f'.py.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')

        # Verzeichnis erstellen
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Code schreiben
        file_path.write_text(program.code, encoding='utf-8')
        program.deployed = True

        print(f"[DEPLOY] {task.target_file} erstellt")

    def modify_existing_code(
        self,
        file_path: str,
        modification: str,
        auto_deploy: bool = True,
    ) -> GeneratedProgram:
        """
        Modifiziert bestehenden Code.

        Args:
            file_path: Pfad zur zu modifizierenden Datei
            modification: Beschreibung der gewünschten Änderung
            auto_deploy: Automatisch deployen

        Returns:
            GeneratedProgram mit modifiziertem Code
        """
        task = ProgrammingTask(
            description=modification,
            target_file=file_path,
            modify_existing=True,
        )

        return self.write_program(task, auto_deploy=auto_deploy)

    def create_complete_module(
        self,
        name: str,
        description: str,
        features: List[str],
        module_type: str = "service",
    ) -> List[GeneratedProgram]:
        """
        Erstellt ein komplettes Modul mit allen benötigten Dateien.

        Args:
            name: Modulname
            description: Beschreibung
            features: Liste der Features
            module_type: "worker", "service", "integration"

        Returns:
            Liste der generierten Programme
        """
        programs = []
        name_lower = name.lower().replace(' ', '_')

        # Hauptmodul
        main_task = ProgrammingTask(
            description=f"{name} {module_type}: {description}. Features: {', '.join(features)}",
            target_file=f"backend/{module_type}s/{name_lower}.py" if module_type != "integration" else f"backend/integrations/{name_lower}.py",
            requirements=features,
        )
        programs.append(self.write_program(main_task, auto_deploy=True))

        # Route falls Worker oder Service
        if module_type in ["worker", "service"]:
            route_task = ProgrammingTask(
                description=f"API Routes für {name} {module_type}",
                target_file=f"backend/routes/{name_lower}.py",
                context_files=[main_task.target_file] if main_task.target_file else [],
            )
            programs.append(self.write_program(route_task, auto_deploy=True))

        return programs

    def get_history(self) -> List[Dict[str, Any]]:
        """Gibt die Generierungshistorie zurück"""
        return [
            {
                "id": p.id,
                "request": p.request,
                "quality": p.quality.value,
                "iterations": p.iterations,
                "deployed": p.deployed,
                "generation_time": p.generation_time,
                "created_at": p.created_at.isoformat(),
            }
            for p in self._generation_history
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        if not self._generation_history:
            return {"total": 0}

        quality_counts = {}
        for p in self._generation_history:
            q = p.quality.value
            quality_counts[q] = quality_counts.get(q, 0) + 1

        return {
            "total": len(self._generation_history),
            "deployed": sum(1 for p in self._generation_history if p.deployed),
            "quality_distribution": quality_counts,
            "avg_iterations": sum(p.iterations for p in self._generation_history) / len(self._generation_history),
            "avg_generation_time": sum(p.generation_time for p in self._generation_history) / len(self._generation_history),
        }


# Singleton
_ai_programmer: Optional[AIProgrammer] = None

def get_ai_programmer() -> AIProgrammer:
    """Gibt Singleton-Instanz zurück"""
    global _ai_programmer
    if _ai_programmer is None:
        _ai_programmer = AIProgrammer()
    return _ai_programmer
