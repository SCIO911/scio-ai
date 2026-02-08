#!/usr/bin/env python3
"""
SCIO - Code Generator
Generiert neuen Code für SCIO selbst

Features:
- Worker-Generation
- Route-Generation
- Service-Generation
- Integration-Generation
- Basierend auf existierenden Patterns
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


# Templates für verschiedene Code-Typen
WORKER_TEMPLATE = '''#!/usr/bin/env python3
"""
SCIO - {worker_name} Worker
{description}
Automatisch generiert von SCIO Autonomy System
Generiert am: {timestamp}
"""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base_worker import BaseWorker, WorkerStatus, model_manager
from backend.config import Config

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    TORCH_AVAILABLE = False

# Transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Available Models
{worker_name_upper}_MODELS = {{
{models_config}
}}


class {class_name}(BaseWorker):
    """
    {worker_name} Worker - {description}

    Features:
{features_list}
    """

    def __init__(self):
        super().__init__("{worker_name}")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None
        self._current_model_id = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        if not TRANSFORMERS_AVAILABLE:
            self._error_message = "Transformers library not available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] {worker_name} Worker bereit (Device: {{self._device}})")
        return True

    def _load_model(self, model_id: str):
        """Load a model"""
        if model_id not in {worker_name_upper}_MODELS:
            model_id = list({worker_name_upper}_MODELS.keys())[0]  # Default

        model_info = {worker_name_upper}_MODELS[model_id]
        hf_id = model_info['hf_id']

        def loader():
            print(f"[LOAD] Lade {{model_info['name']}}...")

            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            return {{"model": model, "tokenizer": tokenizer}}

        result = model_manager.get_model(hf_id, loader)
        self._model = result["model"]
        self._tokenizer = result["tokenizer"]
        self._current_model_id = model_id
        print(f"[OK] {{model_info['name']}} geladen")

{methods}

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process a job"""
        task_type = input_data.get("task", "{default_task}")
        model = input_data.get("model", list({worker_name_upper}_MODELS.keys())[0])

        self.notify_progress(job_id, 0.1, f"Starting {{task_type}}")

{process_handlers}

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._model = None
        self._tokenizer = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] {worker_name} Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        return {worker_name_upper}_MODELS


# Singleton Instance
_{worker_name_lower}_worker: Optional[{class_name}] = None


def get_{worker_name_lower}_worker() -> {class_name}:
    """Get singleton instance"""
    global _{worker_name_lower}_worker
    if _{worker_name_lower}_worker is None:
        _{worker_name_lower}_worker = {class_name}()
    return _{worker_name_lower}_worker
'''


ROUTE_TEMPLATE = '''#!/usr/bin/env python3
"""
SCIO - {route_name} Routes
{description}
Automatisch generiert von SCIO Autonomy System
Generiert am: {timestamp}
"""

from flask import Blueprint, request, jsonify
from backend.services.api_keys import require_api_key

{route_name_lower}_bp = Blueprint('{route_name_lower}', __name__)


{endpoints}


def register_{route_name_lower}_routes(app):
    """Registriert {route_name} Routes"""
    app.register_blueprint({route_name_lower}_bp, url_prefix='/api/v1/{route_name_lower}')
'''


SERVICE_TEMPLATE = '''#!/usr/bin/env python3
"""
SCIO - {service_name} Service
{description}
Automatisch generiert von SCIO Autonomy System
Generiert am: {timestamp}
"""

import threading
from typing import Optional, List, Dict, Any
from datetime import datetime


class {class_name}:
    """
    {service_name} Service

    {description}
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
{init_vars}

    def initialize(self) -> bool:
        """Initialisiert den Service"""
        try:
{init_code}
            self._initialized = True
            print(f"[OK] {service_name} Service initialisiert")
            return True
        except Exception as e:
            print(f"[ERROR] {service_name} Init fehlgeschlagen: {{e}}")
            return False

{methods}

    def shutdown(self):
        """Beendet den Service"""
        with self._lock:
            self._initialized = False
        print(f"[OK] {service_name} Service beendet")


# Singleton
_{service_name_lower}_service: Optional[{class_name}] = None


def get_{service_name_lower}_service() -> {class_name}:
    """Gibt Singleton-Instanz zurück"""
    global _{service_name_lower}_service
    if _{service_name_lower}_service is None:
        _{service_name_lower}_service = {class_name}()
    return _{service_name_lower}_service
'''


class CodeGenerator:
    """
    SCIO Code Generator

    Generiert neuen Code basierend auf:
    - Erkannten Lücken
    - Existierenden Patterns
    - Best Practices
    """

    def __init__(self):
        from backend.config import Config
        self.self_awareness = None
        self.memory = None
        self.base_path = Path(getattr(Config, 'BASE_DIR', 'C:/SCIO'))
        self._generated_files: List[GeneratedCode] = []

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
        Generiert einen neuen Worker

        Args:
            name: Worker-Name (z.B. "Translation")
            description: Beschreibung
            capabilities: Liste von Fähigkeiten (z.B. ["translate_text", "detect_language"])
            models: Liste von Modellen [{name, hf_id, vram_gb}]
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worker_name = name
        worker_name_lower = name.lower().replace(" ", "_")
        worker_name_upper = worker_name_lower.upper()
        class_name = f"{name.replace(' ', '')}Worker"

        # Generate models config
        if models:
            models_config = ""
            for m in models:
                models_config += f"    '{m['name']}': {{\n"
                models_config += f"        'name': '{m.get('display_name', m['name'])}',\n"
                models_config += f"        'hf_id': '{m['hf_id']}',\n"
                models_config += f"        'vram_gb': {m.get('vram_gb', 8)},\n"
                models_config += f"        'context_length': {m.get('context_length', 8192)},\n"
                models_config += "    },\n"
        else:
            models_config = "    # Add models here\n"

        # Generate features list
        features_list = ""
        for cap in capabilities:
            features_list += f"    - {cap.replace('_', ' ').title()}\n"

        # Generate methods
        methods = ""
        for cap in capabilities:
            method_name = cap
            methods += self._generate_method(method_name, worker_name)
            methods += "\n"

        # Generate process handlers
        process_handlers = ""
        for i, cap in enumerate(capabilities):
            if i == 0:
                process_handlers += f'        if task_type == "{cap}":\n'
            else:
                process_handlers += f'        elif task_type == "{cap}":\n'
            process_handlers += f'            result = self.{cap}(input_data)\n\n'

        process_handlers += '        else:\n'
        process_handlers += '            raise ValueError(f"Unknown task type: {task_type}")\n'

        # Fill template
        content = WORKER_TEMPLATE.format(
            worker_name=worker_name,
            worker_name_lower=worker_name_lower,
            worker_name_upper=worker_name_upper,
            class_name=class_name,
            description=description,
            timestamp=timestamp,
            models_config=models_config,
            features_list=features_list,
            methods=methods,
            process_handlers=process_handlers,
            default_task=capabilities[0] if capabilities else "process",
        )

        file_path = f"backend/workers/{worker_name_lower}_worker.py"

        generated = GeneratedCode(
            file_path=file_path,
            content=content,
            description=f"Worker für {description}",
            imports_needed=["torch", "transformers"],
            dependencies=[],
        )

        self._generated_files.append(generated)

        if self.memory:
            self.memory.log_event(
                "code_generation",
                f"Generated worker: {worker_name}",
                {"file": file_path, "capabilities": capabilities}
            )

        return generated

    def _generate_method(self, method_name: str, worker_name: str) -> str:
        """Generiert eine Methode für einen Worker"""
        return f'''    def {method_name}(self, input_data: dict) -> dict:
        """
        {method_name.replace('_', ' ').title()}

        Args:
            input_data: Eingabedaten

        Returns:
            dict mit Ergebnis
        """
        start_time = time.time()

        # Load model if needed
        model_id = input_data.get("model")
        if model_id and (self._current_model_id != model_id or self._model is None):
            self._load_model(model_id)

        # TODO: Implement {method_name}
        # This is a placeholder - implement actual logic

        result = {{
            "status": "success",
            "message": "{method_name} completed",
            "gpu_seconds": time.time() - start_time,
        }}

        return result
'''

    def generate_route(
        self,
        name: str,
        description: str,
        endpoints: List[Dict[str, str]],
    ) -> GeneratedCode:
        """
        Generiert neue API Routes

        Args:
            name: Route-Name (z.B. "Translation")
            description: Beschreibung
            endpoints: Liste von Endpoints [{path, method, function, description}]
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        route_name = name
        route_name_lower = name.lower().replace(" ", "_")

        # Generate endpoints
        endpoints_code = ""
        for ep in endpoints:
            path = ep.get("path", "/")
            method = ep.get("method", "POST")
            func_name = ep.get("function", "handler")
            ep_description = ep.get("description", "")

            endpoints_code += f'''
@{route_name_lower}_bp.route('{path}', methods=['{method}'])
@require_api_key
def {func_name}():
    """
    {ep_description}
    """
    try:
        data = request.get_json() or {{}}

        # TODO: Implement {func_name}
        result = {{"status": "success", "message": "{func_name} not implemented yet"}}

        return jsonify(result)

    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

'''

        content = ROUTE_TEMPLATE.format(
            route_name=route_name,
            route_name_lower=route_name_lower,
            description=description,
            timestamp=timestamp,
            endpoints=endpoints_code,
        )

        file_path = f"backend/routes/{route_name_lower}.py"

        generated = GeneratedCode(
            file_path=file_path,
            content=content,
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
        Generiert einen neuen Service

        Args:
            name: Service-Name
            description: Beschreibung
            methods: Liste von Methoden [{name, description, params, returns}]
            init_vars: Initialisierungs-Variablen
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        service_name = name
        service_name_lower = name.lower().replace(" ", "_")
        class_name = f"{name.replace(' ', '')}Service"

        # Generate init vars
        init_vars_code = ""
        if init_vars:
            for var in init_vars:
                init_vars_code += f"        self.{var} = None\n"

        # Generate methods
        methods_code = ""
        for m in methods:
            method_name = m.get("name", "method")
            method_desc = m.get("description", "")
            params = m.get("params", "")
            returns = m.get("returns", "None")

            methods_code += f'''
    def {method_name}(self{", " + params if params else ""}) -> {returns}:
        """
        {method_desc}
        """
        with self._lock:
            # TODO: Implement {method_name}
            pass
'''

        content = SERVICE_TEMPLATE.format(
            service_name=service_name,
            service_name_lower=service_name_lower,
            class_name=class_name,
            description=description,
            timestamp=timestamp,
            init_vars=init_vars_code,
            init_code="            pass  # TODO: Add initialization",
            methods=methods_code,
        )

        file_path = f"backend/services/{service_name_lower}.py"

        generated = GeneratedCode(
            file_path=file_path,
            content=content,
            description=f"Service für {description}",
            imports_needed=[],
            dependencies=[],
        )

        self._generated_files.append(generated)

        return generated

    def generate_capability(self, capability_name: str, capability_info: dict) -> Optional[GeneratedCode]:
        """
        Generiert Code für eine fehlende Fähigkeit

        Args:
            capability_name: Name der Fähigkeit
            capability_info: Info über die Fähigkeit

        Returns:
            GeneratedCode oder None
        """
        category = capability_info.get("category", "")
        description = capability_info.get("description", "")

        # Decide what type of code to generate
        if category in ["llm", "image", "audio", "video", "code", "vision", "embeddings", "document", "3d"]:
            # Generate worker method or extension
            return self._generate_worker_extension(capability_name, description, category)

        elif category == "automation":
            # Generate automation service
            return self.generate_service(
                name=capability_name.replace("_", " ").title(),
                description=description,
                methods=[{"name": capability_name, "description": description}],
            )

        elif category == "integration":
            # Generate integration
            return self._generate_integration(capability_name, description)

        return None

    def _generate_worker_extension(self, capability: str, description: str, category: str) -> GeneratedCode:
        """Generiert eine Worker-Erweiterung"""
        # Find existing worker for this category
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
        method_code = self._generate_method(capability, category)

        return GeneratedCode(
            file_path=f"backend/workers/{worker_file}.py",
            content=method_code,
            description=f"Erweiterung für {description}",
            imports_needed=[],
            dependencies=[],
        )

    def _generate_integration(self, name: str, description: str) -> GeneratedCode:
        """Generiert eine Integration"""
        integration_name = name.replace("integration_", "")

        content = f'''#!/usr/bin/env python3
"""
SCIO - {integration_name.title()} Integration
{description}
Automatisch generiert von SCIO Autonomy System
"""

import os
import httpx
from typing import Optional, Dict, Any


class {integration_name.title()}Integration:
    """
    {integration_name.title()} Integration

    {description}
    """

    def __init__(self):
        self.api_key = os.getenv("{integration_name.upper()}_API_KEY")
        self.enabled = os.getenv("{integration_name.upper()}_ENABLED", "false").lower() == "true"
        self._client = None

    def initialize(self) -> bool:
        """Initialisiert die Integration"""
        if not self.enabled:
            print(f"[INFO] {integration_name.title()} Integration deaktiviert")
            return False

        if not self.api_key:
            print(f"[WARN] {integration_name.upper()}_API_KEY nicht gesetzt")
            return False

        self._client = httpx.Client(timeout=30)
        print(f"[OK] {integration_name.title()} Integration initialisiert")
        return True

    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """GET Request an die API"""
        if not self._client:
            return {{"error": "Integration nicht initialisiert"}}
        try:
            response = self._client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {{"error": str(e)}}

    def post(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """POST Request an die API"""
        if not self._client:
            return {{"error": "Integration nicht initialisiert"}}
        try:
            response = self._client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {{"error": str(e)}}

    def get_status(self) -> Dict[str, Any]:
        """Gibt Status der Integration zurück"""
        return {{
            "enabled": self.enabled,
            "initialized": self._client is not None,
            "api_key_set": bool(self.api_key),
        }}

    def shutdown(self):
        """Beendet die Integration"""
        if self._client:
            self._client.close()


# Singleton
_{integration_name}_integration = None

def get_{integration_name}_integration():
    global _{integration_name}_integration
    if _{integration_name}_integration is None:
        _{integration_name}_integration = {integration_name.title()}Integration()
    return _{integration_name}_integration
'''

        return GeneratedCode(
            file_path=f"backend/integrations/{integration_name}.py",
            content=content,
            description=f"Integration für {description}",
            imports_needed=["httpx"],
            dependencies=[],
        )

    def save_generated(self, generated: GeneratedCode, backup: bool = True) -> bool:
        """
        Speichert generierten Code

        Args:
            generated: GeneratedCode Objekt
            backup: Backup erstellen wenn Datei existiert

        Returns:
            True wenn erfolgreich
        """
        try:
            file_path = self.base_path / generated.file_path

            # Create backup if file exists
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f".py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')

            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
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
        Repariert Syntax-Fehler im Code

        Args:
            code: Der Code mit Syntax-Fehler
            error_info: Dict mit line, offset, message

        Returns:
            dict mit success und code
        """
        try:
            lines = code.split('\n')
            error_line = error_info.get('line', 1) - 1
            error_msg = error_info.get('message', '')

            # Common syntax fixes
            if 0 <= error_line < len(lines):
                line = lines[error_line]

                # Fix missing colons
                if 'expected \":\"' in error_msg.lower() or 'invalid syntax' in error_msg.lower():
                    # Check for def, class, if, for, while, try, except, with without colon
                    keywords = ['def ', 'class ', 'if ', 'elif ', 'else', 'for ', 'while ', 'try', 'except', 'finally', 'with ']
                    for kw in keywords:
                        if kw in line and not line.rstrip().endswith(':'):
                            lines[error_line] = line.rstrip() + ':'
                            break

                # Fix unmatched parentheses
                if 'unmatched' in error_msg.lower() or 'unexpected EOF' in error_msg.lower():
                    open_parens = line.count('(') - line.count(')')
                    open_brackets = line.count('[') - line.count(']')
                    open_braces = line.count('{') - line.count('}')

                    if open_parens > 0:
                        lines[error_line] = line.rstrip() + ')' * open_parens
                    if open_brackets > 0:
                        lines[error_line] = line.rstrip() + ']' * open_brackets
                    if open_braces > 0:
                        lines[error_line] = line.rstrip() + '}' * open_braces

                # Fix indentation errors
                if 'indent' in error_msg.lower():
                    # Try to match previous line's indentation
                    if error_line > 0:
                        prev_line = lines[error_line - 1]
                        prev_indent = len(prev_line) - len(prev_line.lstrip())
                        current_content = line.lstrip()
                        # Check if prev line ends with colon (needs more indent)
                        if prev_line.rstrip().endswith(':'):
                            lines[error_line] = ' ' * (prev_indent + 4) + current_content
                        else:
                            lines[error_line] = ' ' * prev_indent + current_content

            fixed_code = '\n'.join(lines)

            # Verify the fix worked
            import ast
            try:
                ast.parse(fixed_code)
                return {'success': True, 'code': fixed_code}
            except SyntaxError:
                # If still broken, return original with note
                return {'success': False, 'code': code, 'error': 'Could not auto-fix syntax'}

        except Exception as e:
            return {'success': False, 'code': code, 'error': str(e)}

    def generate_method(self, method_name: str, class_name: str = None,
                        parameters: List[str] = None, return_type: str = 'None',
                        docstring: str = None) -> dict:
        """
        Generiert eine Python-Methode

        Args:
            method_name: Name der Methode
            class_name: Optionaler Klassenname (für Instanz-Methoden)
            parameters: Liste der Parameter
            return_type: Rückgabetyp
            docstring: Dokumentation

        Returns:
            dict mit success und code
        """
        try:
            params = parameters or []

            # Build parameter string
            if class_name:
                # Instance method - add self
                param_str = 'self'
                if params:
                    param_str += ', ' + ', '.join(params)
            else:
                param_str = ', '.join(params) if params else ''

            # Build docstring
            doc = docstring or f'{method_name} method'
            doc_lines = f'"""{doc}"""'

            # Generate method code
            indent = '    ' if class_name else ''
            code = f'''{indent}def {method_name}({param_str}) -> {return_type}:
{indent}    {doc_lines}
{indent}    # Auto-generated implementation
{indent}    pass
'''

            return {'success': True, 'code': code, 'method_name': method_name}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def insert_method(self, file_path: str, class_name: str, method_code: str) -> dict:
        """
        Fügt eine Methode in eine bestehende Klasse ein

        Args:
            file_path: Pfad zur Python-Datei
            class_name: Name der Klasse
            method_code: Der Methoden-Code

        Returns:
            dict mit success
        """
        try:
            from pathlib import Path
            import re

            path = Path(file_path)
            if not path.exists():
                return {'success': False, 'error': f'File not found: {file_path}'}

            content = path.read_text(encoding='utf-8')

            # Find the class definition
            class_pattern = rf'class\s+{class_name}\s*[:\(]'
            match = re.search(class_pattern, content)

            if not match:
                return {'success': False, 'error': f'Class not found: {class_name}'}

            # Find the end of the class (next class definition or end of file)
            class_start = match.start()
            lines = content.split('\n')

            # Find line number of class definition
            char_count = 0
            class_line = 0
            for i, line in enumerate(lines):
                if char_count >= class_start:
                    class_line = i
                    break
                char_count += len(line) + 1  # +1 for newline

            # Find class indentation
            class_indent = len(lines[class_line]) - len(lines[class_line].lstrip())

            # Find last method in class (look for def at class_indent + 4)
            insert_line = class_line + 1
            for i in range(class_line + 1, len(lines)):
                line = lines[i]
                if line.strip():
                    line_indent = len(line) - len(line.lstrip())
                    if line_indent <= class_indent and not line.strip().startswith('#'):
                        # Found end of class
                        insert_line = i
                        break
                    if line.lstrip().startswith('def '):
                        insert_line = i + 1
                        # Find end of this method
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
            # Ensure proper indentation
            indented_method = []
            for line in method_lines:
                if line.strip():
                    indented_method.append(' ' * (class_indent + 4) + line.lstrip())
                else:
                    indented_method.append('')

            # Add blank line before method
            lines.insert(insert_line, '')
            for i, method_line in enumerate(indented_method):
                lines.insert(insert_line + 1 + i, method_line)

            # Write back
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
