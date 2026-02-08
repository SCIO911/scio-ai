#!/usr/bin/env python3
"""
SCIO - Task Verifier
Stellt sicher, dass Aufträge zu 100% erfüllt werden

Features:
- Auftrags-Validierung
- Ergebnis-Prüfung
- Vollständigkeits-Check
- Qualitäts-Kontrolle
- Automatische Korrektur bei Unvollständigkeit
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class VerificationStatus(str, Enum):
    """Status einer Verifizierung"""
    PENDING = "pending"
    VERIFYING = "verifying"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"  # Teilweise erfüllt
    CORRECTING = "correcting"  # Wird korrigiert


class TaskType(str, Enum):
    """Typen von Aufträgen"""
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    FILE_CREATION = "file_creation"
    FILE_MODIFICATION = "file_modification"
    FEATURE_IMPLEMENTATION = "feature_implementation"
    BUG_FIX = "bug_fix"
    EVOLUTION = "evolution"
    ANALYSIS = "analysis"
    CUSTOM = "custom"


@dataclass
class VerificationResult:
    """Ergebnis einer Verifizierung"""
    status: VerificationStatus
    completion_percent: float  # 0-100
    checks_passed: int
    checks_total: int
    details: List[Dict[str, Any]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    corrections_needed: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskRequirement:
    """Eine Anforderung an einen Task"""
    name: str
    description: str
    check_function: Callable
    required: bool = True
    weight: float = 1.0  # Gewichtung für Completion-Berechnung


class TaskVerifier:
    """
    SCIO Task Verifier

    Stellt sicher, dass jeder Auftrag vollständig erfüllt wird:
    1. Definiert Anforderungen basierend auf Task-Typ
    2. Prüft jede Anforderung
    3. Berechnet Completion-Prozent
    4. Identifiziert fehlende Teile
    5. Triggert Korrekturen wenn nötig
    """

    def __init__(self):
        self.base_path = Path("C:/SCIO")
        self._verification_history: List[VerificationResult] = []
        self._correction_callbacks: List[Callable] = []

    def initialize(self) -> bool:
        """Initialisiert den Verifier"""
        return True

    def verify_task(
        self,
        task_type: TaskType,
        task_description: str,
        expected_outputs: Dict[str, Any],
        actual_outputs: Dict[str, Any] = None,
        auto_correct: bool = True
    ) -> VerificationResult:
        """
        Verifiziert einen Task

        Args:
            task_type: Typ des Tasks
            task_description: Beschreibung was getan werden sollte
            expected_outputs: Was erwartet wird (files, changes, etc.)
            actual_outputs: Was tatsächlich produziert wurde
            auto_correct: Automatisch korrigieren wenn unvollständig

        Returns:
            VerificationResult mit Details
        """
        print(f"[VERIFY] Prüfe Task: {task_description[:50]}...")

        # Get requirements for this task type
        requirements = self._get_requirements(task_type, expected_outputs)

        # Run all checks
        checks_passed = 0
        checks_total = len(requirements)
        details = []
        missing = []
        total_weight = sum(r.weight for r in requirements)
        achieved_weight = 0

        for req in requirements:
            try:
                passed = req.check_function(expected_outputs, actual_outputs)

                detail = {
                    "requirement": req.name,
                    "description": req.description,
                    "passed": passed,
                    "required": req.required,
                    "weight": req.weight,
                }
                details.append(detail)

                if passed:
                    checks_passed += 1
                    achieved_weight += req.weight
                else:
                    if req.required:
                        missing.append(req.name)

            except Exception as e:
                details.append({
                    "requirement": req.name,
                    "description": req.description,
                    "passed": False,
                    "error": str(e),
                    "required": req.required,
                })
                if req.required:
                    missing.append(req.name)

        # Calculate completion percentage
        completion_percent = (achieved_weight / total_weight * 100) if total_weight > 0 else 0

        # Determine status
        if completion_percent >= 100:
            status = VerificationStatus.PASSED
        elif completion_percent >= 80:
            status = VerificationStatus.PARTIAL
        else:
            status = VerificationStatus.FAILED

        # Identify corrections needed
        corrections_needed = []
        for m in missing:
            correction = self._suggest_correction(m, task_type, expected_outputs)
            if correction:
                corrections_needed.append(correction)

        result = VerificationResult(
            status=status,
            completion_percent=round(completion_percent, 1),
            checks_passed=checks_passed,
            checks_total=checks_total,
            details=details,
            missing=missing,
            corrections_needed=corrections_needed,
        )

        self._verification_history.append(result)

        # Log result
        if result.status == VerificationStatus.PASSED:
            print(f"[VERIFY] PASSED - 100% erfüllt")
        else:
            print(f"[VERIFY] {result.status.value.upper()} - {result.completion_percent}% erfüllt")
            print(f"[VERIFY] Fehlend: {missing}")

            # Auto-correct if enabled
            if auto_correct and corrections_needed:
                print(f"[VERIFY] Starte automatische Korrektur...")
                self._execute_corrections(corrections_needed, task_type, expected_outputs)

        return result

    def _get_requirements(
        self,
        task_type: TaskType,
        expected: Dict[str, Any]
    ) -> List[TaskRequirement]:
        """Gibt Anforderungen für einen Task-Typ zurück"""
        requirements = []

        # Common requirements
        if "files" in expected:
            for file_path in expected["files"]:
                requirements.append(TaskRequirement(
                    name=f"file_exists:{file_path}",
                    description=f"Datei {file_path} existiert",
                    check_function=lambda e, a, fp=file_path: self._check_file_exists(fp),
                    required=True,
                    weight=2.0,
                ))

                requirements.append(TaskRequirement(
                    name=f"file_not_empty:{file_path}",
                    description=f"Datei {file_path} ist nicht leer",
                    check_function=lambda e, a, fp=file_path: self._check_file_not_empty(fp),
                    required=True,
                    weight=1.0,
                ))

        if "code_files" in expected:
            for file_path in expected["code_files"]:
                requirements.append(TaskRequirement(
                    name=f"syntax_valid:{file_path}",
                    description=f"Syntax in {file_path} ist gültig",
                    check_function=lambda e, a, fp=file_path: self._check_syntax(fp),
                    required=True,
                    weight=2.0,
                ))

        if "classes" in expected:
            for class_info in expected["classes"]:
                file_path = class_info.get("file")
                class_name = class_info.get("name")
                requirements.append(TaskRequirement(
                    name=f"class_exists:{class_name}",
                    description=f"Klasse {class_name} existiert in {file_path}",
                    check_function=lambda e, a, fp=file_path, cn=class_name: self._check_class_exists(fp, cn),
                    required=True,
                    weight=2.0,
                ))

                if "methods" in class_info:
                    for method in class_info["methods"]:
                        requirements.append(TaskRequirement(
                            name=f"method_exists:{class_name}.{method}",
                            description=f"Methode {method} existiert in {class_name}",
                            check_function=lambda e, a, fp=file_path, cn=class_name, m=method: self._check_method_exists(fp, cn, m),
                            required=True,
                            weight=1.5,
                        ))

        if "functions" in expected:
            for func_info in expected["functions"]:
                file_path = func_info.get("file")
                func_name = func_info.get("name")
                requirements.append(TaskRequirement(
                    name=f"function_exists:{func_name}",
                    description=f"Funktion {func_name} existiert in {file_path}",
                    check_function=lambda e, a, fp=file_path, fn=func_name: self._check_function_exists(fp, fn),
                    required=True,
                    weight=1.5,
                ))

        if "imports" in expected:
            for import_info in expected["imports"]:
                file_path = import_info.get("file")
                module = import_info.get("module")
                requirements.append(TaskRequirement(
                    name=f"import_exists:{module}",
                    description=f"Import {module} in {file_path}",
                    check_function=lambda e, a, fp=file_path, mod=module: self._check_import_exists(fp, mod),
                    required=False,
                    weight=0.5,
                ))

        if "content_contains" in expected:
            for content_check in expected["content_contains"]:
                file_path = content_check.get("file")
                text = content_check.get("text")
                requirements.append(TaskRequirement(
                    name=f"content_contains:{text[:20]}",
                    description=f"Datei enthält '{text[:30]}'",
                    check_function=lambda e, a, fp=file_path, t=text: self._check_content_contains(fp, t),
                    required=content_check.get("required", True),
                    weight=1.0,
                ))

        if "no_errors" in expected and expected["no_errors"]:
            requirements.append(TaskRequirement(
                name="no_syntax_errors",
                description="Keine Syntax-Fehler in allen Code-Dateien",
                check_function=lambda e, a: self._check_no_errors(e.get("code_files", [])),
                required=True,
                weight=3.0,
            ))

        if "integration" in expected:
            file_path = expected["integration"].get("file")
            component = expected["integration"].get("component")
            requirements.append(TaskRequirement(
                name=f"integration:{component}",
                description=f"Komponente {component} ist integriert",
                check_function=lambda e, a, fp=file_path, c=component: self._check_integration(fp, c),
                required=True,
                weight=2.0,
            ))

        return requirements

    def _check_file_exists(self, file_path: str) -> bool:
        """Prüft ob Datei existiert"""
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        return full_path.exists()

    def _check_file_not_empty(self, file_path: str) -> bool:
        """Prüft ob Datei nicht leer ist"""
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            return False
        return full_path.stat().st_size > 0

    def _check_syntax(self, file_path: str) -> bool:
        """Prüft Python-Syntax"""
        import ast
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            return False
        try:
            content = full_path.read_text(encoding='utf-8')
            ast.parse(content)
            return True
        except SyntaxError:
            return False

    def _check_class_exists(self, file_path: str, class_name: str) -> bool:
        """Prüft ob Klasse existiert"""
        import ast
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            return False
        try:
            content = full_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return True
            return False
        except:
            return False

    def _check_method_exists(self, file_path: str, class_name: str, method_name: str) -> bool:
        """Prüft ob Methode in Klasse existiert"""
        import ast
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            return False
        try:
            content = full_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            return True
            return False
        except:
            return False

    def _check_function_exists(self, file_path: str, func_name: str) -> bool:
        """Prüft ob Funktion existiert"""
        import ast
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            return False
        try:
            content = full_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return True
            return False
        except:
            return False

    def _check_import_exists(self, file_path: str, module: str) -> bool:
        """Prüft ob Import existiert"""
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            return False
        try:
            content = full_path.read_text(encoding='utf-8')
            return module in content
        except:
            return False

    def _check_content_contains(self, file_path: str, text: str) -> bool:
        """Prüft ob Datei Text enthält"""
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            return False
        try:
            content = full_path.read_text(encoding='utf-8')
            return text in content
        except:
            return False

    def _check_no_errors(self, files: List[str]) -> bool:
        """Prüft ob keine Syntax-Fehler vorhanden sind"""
        for file_path in files:
            if not self._check_syntax(file_path):
                return False
        return True

    def _check_integration(self, file_path: str, component: str) -> bool:
        """Prüft ob Komponente integriert ist"""
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            return False
        try:
            content = full_path.read_text(encoding='utf-8')
            # Check for import and registration
            return component in content
        except:
            return False

    def _suggest_correction(
        self,
        missing: str,
        task_type: TaskType,
        expected: Dict[str, Any]
    ) -> Optional[str]:
        """Schlägt eine Korrektur für fehlende Anforderung vor"""
        if missing.startswith("file_exists:"):
            file_path = missing.split(":", 1)[1]
            return f"CREATE_FILE:{file_path}"

        if missing.startswith("class_exists:"):
            class_name = missing.split(":", 1)[1]
            return f"CREATE_CLASS:{class_name}"

        if missing.startswith("method_exists:"):
            method_info = missing.split(":", 1)[1]
            return f"CREATE_METHOD:{method_info}"

        if missing.startswith("function_exists:"):
            func_name = missing.split(":", 1)[1]
            return f"CREATE_FUNCTION:{func_name}"

        if missing.startswith("syntax_valid:"):
            file_path = missing.split(":", 1)[1]
            return f"FIX_SYNTAX:{file_path}"

        if missing.startswith("integration:"):
            component = missing.split(":", 1)[1]
            return f"INTEGRATE:{component}"

        return None

    def _execute_corrections(
        self,
        corrections: List[str],
        task_type: TaskType,
        expected: Dict[str, Any]
    ):
        """Führt Korrekturen durch"""
        for correction in corrections:
            try:
                action, target = correction.split(":", 1)
                print(f"[CORRECT] {action}: {target}")

                # Notify callbacks
                for callback in self._correction_callbacks:
                    callback(action, target, expected)

            except Exception as e:
                print(f"[ERROR] Korrektur fehlgeschlagen: {e}")

    def add_correction_callback(self, callback: Callable):
        """Registriert Callback für Korrekturen"""
        self._correction_callbacks.append(callback)

    def verify_code_generation(
        self,
        file_path: str,
        expected_classes: List[str] = None,
        expected_methods: Dict[str, List[str]] = None,
        expected_functions: List[str] = None,
        auto_correct: bool = True
    ) -> VerificationResult:
        """
        Spezialisierte Verifizierung für Code-Generierung

        Args:
            file_path: Pfad zur generierten Datei
            expected_classes: Erwartete Klassen
            expected_methods: Erwartete Methoden pro Klasse
            expected_functions: Erwartete Funktionen
            auto_correct: Automatisch korrigieren

        Returns:
            VerificationResult
        """
        expected = {
            "files": [file_path],
            "code_files": [file_path],
            "no_errors": True,
        }

        if expected_classes:
            expected["classes"] = [
                {
                    "file": file_path,
                    "name": cls,
                    "methods": expected_methods.get(cls, []) if expected_methods else [],
                }
                for cls in expected_classes
            ]

        if expected_functions:
            expected["functions"] = [
                {"file": file_path, "name": fn}
                for fn in expected_functions
            ]

        return self.verify_task(
            TaskType.CODE_GENERATION,
            f"Code-Generierung: {file_path}",
            expected,
            auto_correct=auto_correct,
        )

    def verify_evolution(
        self,
        capability: str,
        generated_file: str,
        integrated_in: str = None,
        auto_correct: bool = True
    ) -> VerificationResult:
        """
        Verifiziert eine Evolution

        Args:
            capability: Name der Fähigkeit
            generated_file: Generierte Datei
            integrated_in: Datei wo es integriert sein soll
            auto_correct: Automatisch korrigieren

        Returns:
            VerificationResult
        """
        expected = {
            "files": [generated_file],
            "code_files": [generated_file],
            "no_errors": True,
        }

        if integrated_in:
            expected["integration"] = {
                "file": integrated_in,
                "component": capability,
            }

        return self.verify_task(
            TaskType.EVOLUTION,
            f"Evolution: {capability}",
            expected,
            auto_correct=auto_correct,
        )

    def get_history(self) -> List[VerificationResult]:
        """Gibt Verifizierungs-Historie zurück"""
        return self._verification_history

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        if not self._verification_history:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "partial": 0,
                "average_completion": 0,
            }

        total = len(self._verification_history)
        passed = sum(1 for v in self._verification_history if v.status == VerificationStatus.PASSED)
        failed = sum(1 for v in self._verification_history if v.status == VerificationStatus.FAILED)
        partial = sum(1 for v in self._verification_history if v.status == VerificationStatus.PARTIAL)
        avg_completion = sum(v.completion_percent for v in self._verification_history) / total

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "partial": partial,
            "average_completion": round(avg_completion, 1),
            "success_rate": round(passed / total * 100, 1) if total > 0 else 0,
        }


# Singleton
_task_verifier: Optional[TaskVerifier] = None

def get_task_verifier() -> TaskVerifier:
    """Gibt Singleton-Instanz zurück"""
    global _task_verifier
    if _task_verifier is None:
        _task_verifier = TaskVerifier()
    return _task_verifier
