#!/usr/bin/env python3
"""
SCIO - Evolution Planner
Plant und führt die kontinuierliche Selbst-Erweiterung durch

Features:
- Analyse von Verbesserungsmöglichkeiten
- Priorisierung von Erweiterungen
- Automatische Code-Generierung
- Test-Koordination
- Deployment-Management
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EvolutionStatus(str, Enum):
    """Status einer Evolution"""
    PLANNED = "planned"
    GENERATING = "generating"
    TESTING = "testing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class EvolutionError(Exception):
    """Base exception for evolution errors"""
    pass


class CodeGenerationError(EvolutionError):
    """Error during code generation"""
    pass


class TestFailureError(EvolutionError):
    """Tests failed during evolution"""
    pass


class DeploymentError(EvolutionError):
    """Error during deployment"""
    pass


@dataclass
class EvolutionPlan:
    """Ein Evolutions-Plan"""
    id: str
    name: str
    description: str
    capability: str
    priority: int
    status: EvolutionStatus = EvolutionStatus.PLANNED
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    generated_files: List[str] = field(default_factory=list)
    test_results: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class EvolutionResult:
    """Ergebnis einer Evolution"""
    plan: EvolutionPlan
    success: bool
    files_created: List[str]
    files_modified: List[str]
    backups: List[str]
    message: str


class EvolutionPlanner:
    """
    SCIO Evolution Planner

    Koordiniert die Selbst-Erweiterung von SCIO:
    1. Analysiert fehlende Fähigkeiten
    2. Erstellt Evolutions-Pläne
    3. Generiert Code
    4. Testet Code
    5. Deployed bei Erfolg
    6. Rollt zurück bei Fehler
    """

    def __init__(self):
        self.capability_analyzer = None
        self.code_generator = None
        self.self_tester = None
        self.memory = None

        self._plans: List[EvolutionPlan] = []
        self._current_plan: Optional[EvolutionPlan] = None
        self._evolution_history: List[EvolutionResult] = []

    def initialize(
        self,
        capability_analyzer,
        code_generator,
        self_tester,
        memory
    ) -> bool:
        """Initialisiert den Evolution Planner"""
        self.capability_analyzer = capability_analyzer
        self.code_generator = code_generator
        self.self_tester = self_tester
        self.memory = memory
        return True

    def suggest_next_steps(self, limit: int = 5) -> List[dict]:
        """
        Schlägt die nächsten Evolutions-Schritte vor

        Returns:
            Liste von Vorschlägen, sortiert nach Priorität
        """
        if not self.capability_analyzer:
            return []

        gaps = self.capability_analyzer.find_gaps()

        suggestions = []
        for gap in gaps[:limit]:
            suggestion = {
                "capability": gap["name"],
                "category": gap["category"],
                "description": gap["description"],
                "priority": gap["priority"],
                "suggested_action": self._get_suggested_action(gap),
                "estimated_complexity": self._estimate_complexity(gap),
                "dependencies": self._get_dependencies(gap),
            }
            suggestions.append(suggestion)

        return suggestions

    def _get_suggested_action(self, gap: dict) -> str:
        """Bestimmt die empfohlene Aktion für eine Lücke"""
        category = gap.get("category", "")

        if category in ["llm", "image", "audio", "video", "code", "vision", "embeddings", "document", "3d"]:
            return f"Erweitere existierenden {gap.get('suggested_worker', 'Worker')}"

        elif category == "automation":
            return "Erstelle neuen Automation-Service"

        elif category == "integration":
            return "Erstelle neue Integration"

        return "Analysiere und implementiere manuell"

    def _estimate_complexity(self, gap: dict) -> str:
        """Schätzt die Komplexität einer Erweiterung"""
        priority = gap.get("priority", 5)

        if priority >= 9:
            return "hoch"
        elif priority >= 6:
            return "mittel"
        else:
            return "niedrig"

    def _get_dependencies(self, gap: dict) -> List[str]:
        """Bestimmt Abhängigkeiten einer Erweiterung"""
        deps = []
        category = gap.get("category", "")

        if category in ["llm", "image", "audio", "video"]:
            deps.append("torch")
            deps.append("transformers")

        if category == "embeddings":
            deps.append("sentence-transformers")
            deps.append("faiss-gpu")

        if category == "vision":
            deps.append("ultralytics")

        return deps

    def create_plan(self, capability: str, force: bool = False) -> Optional[EvolutionPlan]:
        """
        Erstellt einen Evolutions-Plan für eine Fähigkeit

        Args:
            capability: Name der Fähigkeit
            force: Überschreibe existierenden Plan

        Returns:
            EvolutionPlan oder None
        """
        # Check if already planned
        existing = [p for p in self._plans if p.capability == capability and p.status == EvolutionStatus.PLANNED]
        if existing and not force:
            return existing[0]

        # Get capability info
        if self.capability_analyzer:
            cap_info = self.capability_analyzer.capabilities.get(capability)
            if not cap_info:
                return None

            plan = EvolutionPlan(
                id=f"evo_{int(time.time())}_{capability}",
                name=f"Add {capability}",
                description=cap_info.description,
                capability=capability,
                priority=cap_info.priority,
            )

            self._plans.append(plan)

            if self.memory:
                self.memory.log_event(
                    "evolution_planned",
                    f"Created evolution plan: {plan.name}",
                    {"capability": capability, "priority": plan.priority}
                )

            return plan

        return None

    def execute_evolution(self, target: str = None) -> EvolutionResult:
        """
        Führt eine Evolution durch

        Args:
            target: Spezifisches Ziel oder None für höchste Priorität

        Returns:
            EvolutionResult mit Ergebnis
        """
        # Get or create plan
        if target:
            plan = self.create_plan(target)
        else:
            # Get highest priority missing capability
            if self.capability_analyzer:
                gaps = self.capability_analyzer.find_gaps()
                if gaps:
                    plan = self.create_plan(gaps[0]["name"])
                else:
                    return EvolutionResult(
                        plan=EvolutionPlan(
                            id="none",
                            name="No Evolution Needed",
                            description="All capabilities implemented",
                            capability="none",
                            priority=0,
                        ),
                        success=True,
                        files_created=[],
                        files_modified=[],
                        backups=[],
                        message="No capabilities to add - system is complete",
                    )
            else:
                return EvolutionResult(
                    plan=EvolutionPlan(
                        id="error",
                        name="Error",
                        description="Capability analyzer not available",
                        capability="none",
                        priority=0,
                        status=EvolutionStatus.FAILED,
                    ),
                    success=False,
                    files_created=[],
                    files_modified=[],
                    backups=[],
                    message="Capability analyzer not initialized",
                )

        if not plan:
            return EvolutionResult(
                plan=EvolutionPlan(
                    id="error",
                    name="Error",
                    description="Could not create plan",
                    capability=target or "unknown",
                    priority=0,
                    status=EvolutionStatus.FAILED,
                ),
                success=False,
                files_created=[],
                files_modified=[],
                backups=[],
                message="Could not create evolution plan",
            )

        self._current_plan = plan
        files_created = []
        files_modified = []
        backups = []

        try:
            # Step 1: Generate Code
            plan.status = EvolutionStatus.GENERATING
            print(f"[EVOLUTION] Generiere Code für: {plan.capability}")

            cap_info = {
                "name": plan.capability,
                "category": self.capability_analyzer.capabilities[plan.capability].category.value,
                "description": plan.description,
            }

            generated = self.code_generator.generate_capability(plan.capability, cap_info)

            if not generated:
                raise CodeGenerationError("Code generation failed - no output")

            plan.generated_files.append(generated.file_path)

            # Step 2: Test Code
            plan.status = EvolutionStatus.TESTING
            print(f"[EVOLUTION] Teste generierten Code...")

            test_result = self.self_tester.test_generated_code(
                generated.file_path,
                generated.content
            )

            plan.test_results = {
                "passed": test_result.passed,
                "failed": test_result.failed,
                "can_deploy": test_result.can_deploy,
            }

            if not test_result.can_deploy:
                failed_tests = [r.name for r in test_result.results if r.status.value == "failed"]
                raise TestFailureError(f"Tests failed: {failed_tests}")

            # Step 3: Create Backup
            from backend.config import Config
            full_path = Path(getattr(Config, 'BASE_DIR', 'C:/SCIO')) / generated.file_path
            if full_path.exists():
                backup_path = self.self_tester.create_backup(generated.file_path)
                if backup_path:
                    backups.append(backup_path)
                files_modified.append(generated.file_path)
            else:
                files_created.append(generated.file_path)

            # Step 4: Deploy
            plan.status = EvolutionStatus.DEPLOYING
            print(f"[EVOLUTION] Deploye: {generated.file_path}")

            success = self.code_generator.save_generated(generated, backup=False)

            if not success:
                raise DeploymentError("Failed to save generated code")

            # Step 5: Mark as completed
            plan.status = EvolutionStatus.COMPLETED
            plan.completed_at = datetime.now()

            # Update capability status
            if plan.capability in self.capability_analyzer.capabilities:
                self.capability_analyzer.capabilities[plan.capability].implemented = True
                self.capability_analyzer.capabilities[plan.capability].quality_score = 0.5  # New code

            result = EvolutionResult(
                plan=plan,
                success=True,
                files_created=files_created,
                files_modified=files_modified,
                backups=backups,
                message=f"Successfully added capability: {plan.capability}",
            )

            self._evolution_history.append(result)

            if self.memory:
                self.memory.log_event(
                    "evolution_completed",
                    f"Evolution completed: {plan.capability}",
                    {
                        "files_created": files_created,
                        "files_modified": files_modified,
                        "test_results": plan.test_results,
                    }
                )

            print(f"[EVOLUTION] Erfolgreich: {plan.capability}")
            return result

        except Exception as e:
            # Rollback
            plan.status = EvolutionStatus.FAILED
            plan.error = str(e)

            print(f"[EVOLUTION] Fehlgeschlagen: {e}")

            # Restore backups
            for backup in backups:
                # Find original path
                for file_path in files_modified:
                    self.self_tester.restore_backup(backup, file_path)

            if self.memory:
                self.memory.log_event(
                    "evolution_failed",
                    f"Evolution failed: {plan.capability}",
                    {"error": str(e)}
                )

            return EvolutionResult(
                plan=plan,
                success=False,
                files_created=[],
                files_modified=[],
                backups=backups,
                message=f"Evolution failed: {str(e)}",
            )

        finally:
            self._current_plan = None

    def get_plans(self) -> List[EvolutionPlan]:
        """Gibt alle Pläne zurück"""
        return self._plans

    def get_history(self) -> List[EvolutionResult]:
        """Gibt Evolutions-Historie zurück"""
        return self._evolution_history

    def get_current_plan(self) -> Optional[EvolutionPlan]:
        """Gibt aktuellen Plan zurück (falls Evolution läuft)"""
        return self._current_plan

    def get_status(self) -> dict:
        """Gibt Status des Evolution Planners zurück"""
        return {
            "plans_total": len(self._plans),
            "plans_completed": sum(1 for p in self._plans if p.status == EvolutionStatus.COMPLETED),
            "plans_failed": sum(1 for p in self._plans if p.status == EvolutionStatus.FAILED),
            "plans_pending": sum(1 for p in self._plans if p.status == EvolutionStatus.PLANNED),
            "current_evolution": self._current_plan.name if self._current_plan else None,
            "total_evolutions": len(self._evolution_history),
            "successful_evolutions": sum(1 for e in self._evolution_history if e.success),
        }

    def auto_evolve(self, max_evolutions: int = 1) -> List[EvolutionResult]:
        """
        Führt automatische Evolutionen durch

        Args:
            max_evolutions: Maximale Anzahl von Evolutionen

        Returns:
            Liste von Ergebnissen
        """
        results = []

        for _ in range(max_evolutions):
            result = self.execute_evolution()

            if result.message == "No capabilities to add - system is complete":
                break

            results.append(result)

            if not result.success:
                break  # Stop on first failure

        return results


# Singleton
_evolution_planner: Optional[EvolutionPlanner] = None

def get_evolution_planner() -> EvolutionPlanner:
    """Gibt Singleton-Instanz zurück"""
    global _evolution_planner
    if _evolution_planner is None:
        _evolution_planner = EvolutionPlanner()
    return _evolution_planner
