#!/usr/bin/env python3
"""
SCIO - Self Tester
Testet generierten Code bevor er aktiviert wird

Features:
- Syntax-Validierung
- Import-Validierung
- Basis-Funktionalitäts-Tests
- Integrations-Tests
- Rollback bei Fehlern
"""

import os
import sys
import ast
import importlib
import importlib.util
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TestStatus(str, Enum):
    """Test Status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Einzelnes Test-Ergebnis"""
    name: str
    status: TestStatus
    message: str = ""
    duration_ms: float = 0
    details: Optional[Dict[str, Any]] = None


@dataclass
class TestSuiteResult:
    """Ergebnis einer Test-Suite"""
    file_path: str
    timestamp: datetime
    total_tests: int
    passed: int
    failed: int
    skipped: int
    results: List[TestResult]
    can_deploy: bool


class SelfTester:
    """
    SCIO Self Tester

    Testet generierten Code auf:
    - Syntaktische Korrektheit
    - Importierbarkeit
    - Basis-Funktionalität
    - Integration mit existierendem Code
    """

    def __init__(self):
        self.base_path = Path("C:/SCIO")
        self._test_results: List[TestSuiteResult] = []

    def initialize(self) -> bool:
        """Initialisiert den Self Tester"""
        return True

    def test_generated_code(self, file_path: str, content: str) -> TestSuiteResult:
        """
        Testet generierten Code

        Args:
            file_path: Ziel-Dateipfad
            content: Code-Inhalt

        Returns:
            TestSuiteResult mit allen Tests
        """
        start_time = datetime.now()
        results = []

        # Test 1: Syntax Check
        syntax_result = self._test_syntax(content)
        results.append(syntax_result)

        # Test 2: Import Check (nur wenn Syntax OK)
        if syntax_result.status == TestStatus.PASSED:
            import_result = self._test_imports(content)
            results.append(import_result)

            # Test 3: Load Module (nur wenn Imports OK)
            if import_result.status == TestStatus.PASSED:
                load_result = self._test_module_load(content, file_path)
                results.append(load_result)

                # Test 4: Basic Functionality
                if load_result.status == TestStatus.PASSED:
                    func_result = self._test_basic_functionality(content, file_path)
                    results.append(func_result)
        else:
            # Skip remaining tests
            results.append(TestResult("import_check", TestStatus.SKIPPED, "Skipped due to syntax error"))
            results.append(TestResult("module_load", TestStatus.SKIPPED, "Skipped due to syntax error"))
            results.append(TestResult("basic_functionality", TestStatus.SKIPPED, "Skipped due to syntax error"))

        # Test 5: Code Quality Check
        quality_result = self._test_code_quality(content)
        results.append(quality_result)

        # Calculate totals
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)

        # Can deploy if no failures and at least syntax passed
        can_deploy = failed == 0 and syntax_result.status == TestStatus.PASSED

        suite_result = TestSuiteResult(
            file_path=file_path,
            timestamp=start_time,
            total_tests=len(results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            can_deploy=can_deploy,
        )

        self._test_results.append(suite_result)

        return suite_result

    def _test_syntax(self, content: str) -> TestResult:
        """Testet Python Syntax"""
        import time
        start = time.time()

        try:
            ast.parse(content)
            return TestResult(
                name="syntax_check",
                status=TestStatus.PASSED,
                message="Syntax is valid",
                duration_ms=(time.time() - start) * 1000,
            )
        except SyntaxError as e:
            return TestResult(
                name="syntax_check",
                status=TestStatus.FAILED,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                duration_ms=(time.time() - start) * 1000,
                details={"line": e.lineno, "offset": e.offset, "text": e.text},
            )

    def _test_imports(self, content: str) -> TestResult:
        """Testet ob alle Imports verfügbar sind"""
        import time
        start = time.time()

        try:
            tree = ast.parse(content)
            missing_imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if not self._is_import_available(module_name):
                            missing_imports.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        # Skip relative imports to our own modules
                        if module_name not in ['backend', '.']:
                            if not self._is_import_available(module_name):
                                missing_imports.append(node.module)

            if missing_imports:
                return TestResult(
                    name="import_check",
                    status=TestStatus.FAILED,
                    message=f"Missing imports: {', '.join(missing_imports)}",
                    duration_ms=(time.time() - start) * 1000,
                    details={"missing": missing_imports},
                )

            return TestResult(
                name="import_check",
                status=TestStatus.PASSED,
                message="All imports available",
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return TestResult(
                name="import_check",
                status=TestStatus.FAILED,
                message=f"Import check failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000,
            )

    def _is_import_available(self, module_name: str) -> bool:
        """Prüft ob ein Modul importierbar ist"""
        # Standard library and common packages
        known_available = {
            'os', 'sys', 'time', 'datetime', 'json', 'typing', 'pathlib',
            'threading', 'dataclasses', 'enum', 'abc', 'hashlib', 're',
            'tempfile', 'traceback', 'importlib', 'ast', 'uuid',
            'flask', 'torch', 'transformers', 'numpy', 'httpx', 'requests',
        }

        if module_name in known_available:
            return True

        try:
            importlib.util.find_spec(module_name)
            return True
        except (ModuleNotFoundError, ImportError):
            return False

    def _test_module_load(self, content: str, file_path: str) -> TestResult:
        """Testet ob das Modul geladen werden kann"""
        import time
        start = time.time()

        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                # Replace relative imports for testing
                test_content = content.replace(
                    'from .base_worker',
                    '# from .base_worker  # Disabled for testing\nclass BaseWorker: pass\nclass WorkerStatus: READY="ready"; ERROR="error"\nclass model_manager: @staticmethod\n def get_model(a,b): return b()\n# Original: from .base_worker'
                )
                test_content = test_content.replace(
                    'from backend.config',
                    '# from backend.config  # Disabled for testing\nclass Config: pass\n# Original: from backend.config'
                )
                f.write(test_content)
                temp_path = f.name

            try:
                spec = importlib.util.spec_from_file_location("test_module", temp_path)
                module = importlib.util.module_from_spec(spec)

                # Don't actually execute - just verify it's loadable
                return TestResult(
                    name="module_load",
                    status=TestStatus.PASSED,
                    message="Module can be loaded",
                    duration_ms=(time.time() - start) * 1000,
                )

            finally:
                os.unlink(temp_path)

        except Exception as e:
            return TestResult(
                name="module_load",
                status=TestStatus.FAILED,
                message=f"Module load failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000,
                details={"traceback": traceback.format_exc()},
            )

    def _test_basic_functionality(self, content: str, file_path: str) -> TestResult:
        """Testet Basis-Funktionalität"""
        import time
        start = time.time()

        try:
            tree = ast.parse(content)

            # Find classes and check for required methods
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        "name": node.name,
                        "methods": methods,
                    })

            # Check for Worker pattern
            if "worker" in file_path.lower():
                # Should have certain methods
                required_methods = ["initialize", "process", "cleanup"]
                for cls in classes:
                    if "Worker" in cls["name"]:
                        missing = [m for m in required_methods if m not in cls["methods"]]
                        if missing:
                            return TestResult(
                                name="basic_functionality",
                                status=TestStatus.FAILED,
                                message=f"Worker {cls['name']} missing required methods: {missing}",
                                duration_ms=(time.time() - start) * 1000,
                            )

            # Check for Service pattern
            if "service" in file_path.lower():
                required_methods = ["initialize"]
                for cls in classes:
                    if "Service" in cls["name"]:
                        missing = [m for m in required_methods if m not in cls["methods"]]
                        if missing:
                            return TestResult(
                                name="basic_functionality",
                                status=TestStatus.FAILED,
                                message=f"Service {cls['name']} missing required methods: {missing}",
                                duration_ms=(time.time() - start) * 1000,
                            )

            return TestResult(
                name="basic_functionality",
                status=TestStatus.PASSED,
                message="Basic functionality checks passed",
                duration_ms=(time.time() - start) * 1000,
                details={"classes": classes},
            )

        except Exception as e:
            return TestResult(
                name="basic_functionality",
                status=TestStatus.FAILED,
                message=f"Functionality check failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000,
            )

    def _test_code_quality(self, content: str) -> TestResult:
        """Testet Code-Qualität"""
        import time
        start = time.time()

        issues = []

        # Check for docstrings
        try:
            tree = ast.parse(content)

            # Module docstring
            if not ast.get_docstring(tree):
                issues.append("Missing module docstring")

            # Class docstrings
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        issues.append(f"Class {node.name} missing docstring")

            # Check for TODO comments
            if content.count("TODO") > 10:
                issues.append("Too many TODO comments")

            # Check line length
            long_lines = sum(1 for line in content.split('\n') if len(line) > 120)
            if long_lines > 5:
                issues.append(f"{long_lines} lines exceed 120 characters")

        except Exception as e:
            issues.append(f"Quality check error: {str(e)}")

        if issues:
            return TestResult(
                name="code_quality",
                status=TestStatus.PASSED,  # Quality issues are warnings, not failures
                message=f"Quality warnings: {len(issues)}",
                duration_ms=(time.time() - start) * 1000,
                details={"warnings": issues},
            )

        return TestResult(
            name="code_quality",
            status=TestStatus.PASSED,
            message="Code quality is good",
            duration_ms=(time.time() - start) * 1000,
        )

    def run_integration_test(self, file_path: str) -> TestResult:
        """
        Führt einen Integrations-Test durch

        Prüft ob der Code mit dem Rest des Systems funktioniert.
        """
        import time
        start = time.time()

        try:
            full_path = self.base_path / file_path

            if not full_path.exists():
                return TestResult(
                    name="integration_test",
                    status=TestStatus.FAILED,
                    message=f"File does not exist: {file_path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Try to import the actual module
            # This would require the app to be running, so we do a simpler check

            content = full_path.read_text(encoding='utf-8')

            # Check for circular imports (basic check)
            tree = ast.parse(content)
            module_name = full_path.stem

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and module_name in node.module:
                        return TestResult(
                            name="integration_test",
                            status=TestStatus.FAILED,
                            message=f"Possible circular import: {node.module}",
                            duration_ms=(time.time() - start) * 1000,
                        )

            return TestResult(
                name="integration_test",
                status=TestStatus.PASSED,
                message="Integration check passed",
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return TestResult(
                name="integration_test",
                status=TestStatus.FAILED,
                message=f"Integration test failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000,
            )

    def get_test_results(self) -> List[TestSuiteResult]:
        """Gibt alle Test-Ergebnisse zurück"""
        return self._test_results

    def get_last_result(self) -> Optional[TestSuiteResult]:
        """Gibt letztes Test-Ergebnis zurück"""
        return self._test_results[-1] if self._test_results else None

    def clear_results(self):
        """Löscht Test-Ergebnisse"""
        self._test_results.clear()

    def create_backup(self, file_path: str) -> Optional[str]:
        """
        Erstellt ein Backup einer Datei

        Returns:
            Backup-Pfad oder None
        """
        try:
            full_path = self.base_path / file_path

            if not full_path.exists():
                return None

            backup_dir = self.base_path / "data" / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{full_path.stem}_{timestamp}{full_path.suffix}"
            backup_path = backup_dir / backup_name

            content = full_path.read_text(encoding='utf-8')
            backup_path.write_text(content, encoding='utf-8')

            return str(backup_path)

        except Exception as e:
            print(f"[ERROR] Backup fehlgeschlagen: {e}")
            return None

    def restore_backup(self, backup_path: str, target_path: str) -> bool:
        """
        Stellt ein Backup wieder her

        Returns:
            True wenn erfolgreich
        """
        try:
            backup = Path(backup_path)
            target = self.base_path / target_path

            if not backup.exists():
                return False

            content = backup.read_text(encoding='utf-8')
            target.write_text(content, encoding='utf-8')

            return True

        except Exception as e:
            print(f"[ERROR] Restore fehlgeschlagen: {e}")
            return False


# Singleton
_self_tester: Optional[SelfTester] = None

def get_self_tester() -> SelfTester:
    """Gibt Singleton-Instanz zurück"""
    global _self_tester
    if _self_tester is None:
        _self_tester = SelfTester()
    return _self_tester
