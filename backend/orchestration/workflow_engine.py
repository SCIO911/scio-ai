#!/usr/bin/env python3
"""
SCIO - Workflow Engine
Orchestriert komplexe Workflows über alle Module hinweg
"""

import uuid
import time
import threading
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

from .event_bus import EventBus, EventType, get_event_bus

# Sichere Expression-Auswertung statt eval()
from backend.core.security import SafeExpressionEvaluator
# Plugin-System für Custom Handler
from backend.core.plugins import get_workflow_handlers, PluginNotFoundError, PluginDisabledError


class StepType(str, Enum):
    """Typen von Workflow-Schritten"""
    DECISION = "decision"       # Decision Engine
    RULE_CHECK = "rule_check"   # Rule Engine
    RL_ACTION = "rl_action"     # RL Agent
    PLAN = "plan"               # Planner
    KNOWLEDGE = "knowledge"     # Knowledge Graph
    AGENT_TASK = "agent_task"   # Multi-Agent
    TOOL = "tool"               # Capability Engine
    WORKER = "worker"           # AI Worker
    CONDITION = "condition"     # Bedingung prüfen
    PARALLEL = "parallel"       # Parallele Ausführung
    LOOP = "loop"               # Schleife
    CUSTOM = "custom"           # Custom Handler


class StepStatus(str, Enum):
    """Status eines Workflow-Schritts"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Ein Schritt in einem Workflow"""
    id: str
    name: str
    step_type: StepType
    config: Dict[str, Any] = field(default_factory=dict)

    # Abhängigkeiten
    depends_on: List[str] = field(default_factory=list)

    # Bedingte Ausführung
    condition: Optional[str] = None  # Python-Ausdruck

    # Retry-Konfiguration
    max_retries: int = 0
    retry_delay_s: float = 1.0

    # Status
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0


@dataclass
class Workflow:
    """Ein kompletter Workflow"""
    id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)

    # Kontext der zwischen Steps geteilt wird
    context: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "pending"
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    # Callbacks
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None
    on_step_complete: Optional[Callable] = None


class WorkflowEngine:
    """
    SCIO Workflow Engine

    Orchestriert komplexe Prozesse über alle Module:
    - Decision Engine für Entscheidungen
    - Rule Engine für Geschäftsregeln
    - RL Agent für optimale Aktionen
    - Planner für Planung
    - Knowledge Graph für Wissen
    - Multi-Agent für Delegation
    - Capabilities für Tool-Ausführung
    """

    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, threading.Thread] = {}
        self.step_handlers: Dict[StepType, Callable] = {}
        self.event_bus = get_event_bus()

        self._initialized = False
        self._setup_handlers()
        self._setup_predefined_workflows()

    def _setup_predefined_workflows(self):
        """Registriert vordefinierte Workflows"""
        self.predefined_workflows = {
            "image_generation": {
                "name": "Image Generation",
                "description": "Generiert ein Bild mit optimaler Konfiguration",
                "steps": [
                    {"id": "decide_model", "name": "Model auswählen", "type": "rl_action",
                     "config": {"action_space": "worker_selection"}},
                    {"id": "check_resources", "name": "Ressourcen prüfen", "type": "rule_check",
                     "config": {"rule_set": "resource"}, "depends_on": ["decide_model"]},
                    {"id": "generate", "name": "Bild generieren", "type": "tool",
                     "config": {"tool_id": "gen_img_txt2img"}, "depends_on": ["check_resources"]}
                ]
            },
            "document_processing": {
                "name": "Document Processing",
                "description": "Verarbeitet ein Dokument komplett",
                "steps": [
                    {"id": "extract_text", "name": "Text extrahieren", "type": "tool",
                     "config": {"tool_id": "doc_pdf_read"}},
                    {"id": "analyze", "name": "Inhalt analysieren", "type": "parallel",
                     "config": {"steps": [
                         {"name": "Entities", "type": "tool", "config": {"tool_id": "nlp_entity_extract"}},
                         {"name": "Summary", "type": "tool", "config": {"tool_id": "nlp_text_summarize"}}
                     ]}, "depends_on": ["extract_text"]}
                ]
            },
            "intelligent_decision": {
                "name": "Intelligent Decision",
                "description": "Trifft intelligente Entscheidung mit RL und Regeln",
                "steps": [
                    {"id": "rl_suggest", "name": "RL Vorschlag", "type": "rl_action",
                     "config": {"action_space": "worker_selection"}},
                    {"id": "rule_validate", "name": "Regel-Validierung", "type": "rule_check",
                     "config": {"rule_set": "default"}, "depends_on": ["rl_suggest"]},
                    {"id": "final_decision", "name": "Finale Entscheidung", "type": "decision",
                     "config": {"tree": "default"}, "depends_on": ["rule_validate"]}
                ]
            },
            "full_analysis": {
                "name": "Full Analysis Pipeline",
                "description": "Vollständige Analyse mit allen Modulen",
                "steps": [
                    {"id": "plan", "name": "Plan erstellen", "type": "plan",
                     "config": {"goal": "Analyze input data"}},
                    {"id": "knowledge_query", "name": "Wissen abfragen", "type": "knowledge",
                     "config": {"operation": "query"}, "depends_on": ["plan"]},
                    {"id": "delegate", "name": "An Agent delegieren", "type": "agent_task",
                     "config": {"description": "Analyze and report"}, "depends_on": ["knowledge_query"]}
                ]
            },
            "video_generation": {
                "name": "Video Generation",
                "description": "Generiert Video aus Text oder Bild",
                "steps": [
                    {"id": "select_model", "name": "Model wählen", "type": "rl_action",
                     "config": {"action_space": "worker_selection"}},
                    {"id": "generate_video", "name": "Video generieren", "type": "worker",
                     "config": {"job_type": "video_generation"}, "depends_on": ["select_model"]}
                ]
            },
            "audio_pipeline": {
                "name": "Audio Processing",
                "description": "Verarbeitet Audio mit STT, TTS, Analyse",
                "steps": [
                    {"id": "transcribe", "name": "Audio transkribieren", "type": "tool",
                     "config": {"tool_id": "audio_speech_to_text"}},
                    {"id": "analyze_text", "name": "Text analysieren", "type": "parallel",
                     "config": {"steps": [
                         {"name": "Sentiment", "type": "tool", "config": {"tool_id": "nlp_sentiment_analyze"}},
                         {"name": "Entities", "type": "tool", "config": {"tool_id": "nlp_entity_extract"}}
                     ]}, "depends_on": ["transcribe"]}
                ]
            },
            "code_generation": {
                "name": "Code Generation",
                "description": "Generiert und testet Code",
                "steps": [
                    {"id": "plan_code", "name": "Code planen", "type": "plan",
                     "config": {"goal": "Generate code for task"}},
                    {"id": "generate_code", "name": "Code generieren", "type": "tool",
                     "config": {"tool_id": "code_generate"}, "depends_on": ["plan_code"]},
                    {"id": "review_code", "name": "Code review", "type": "tool",
                     "config": {"tool_id": "code_review"}, "depends_on": ["generate_code"]}
                ]
            },
            "learning_cycle": {
                "name": "Learning Cycle",
                "description": "Führt einen Lernzyklus durch",
                "steps": [
                    {"id": "observe", "name": "Beobachtung erfassen", "type": "custom",
                     "config": {"handler": "observe"}},
                    {"id": "rl_update", "name": "RL Update", "type": "rl_action",
                     "config": {"action_space": "learning"}, "depends_on": ["observe"]},
                    {"id": "knowledge_update", "name": "Wissen aktualisieren", "type": "knowledge",
                     "config": {"operation": "add_entity"}, "depends_on": ["rl_update"]}
                ]
            }
        }

    def execute_predefined(self, workflow_name: str, context: Dict[str, Any] = None) -> Dict:
        """Führt einen vordefinierten Workflow aus"""
        if workflow_name not in self.predefined_workflows:
            return {"error": f"Workflow '{workflow_name}' not found"}

        template = self.predefined_workflows[workflow_name]

        # Workflow erstellen
        workflow = self.create_workflow(
            name=template["name"],
            description=template.get("description", ""),
            steps=template["steps"]
        )

        # Kontext setzen
        if context:
            workflow.context.update(context)

        # Ausführen
        self.execute_workflow(workflow.id, context)

        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "status": "started",
            "steps": len(workflow.steps)
        }

    def initialize(self) -> bool:
        """Initialisiert die Workflow Engine"""
        try:
            self._setup_event_subscriptions()
            self._initialized = True
            print("[OK] Workflow Engine initialisiert")
            return True
        except Exception as e:
            print(f"[ERROR] Workflow Engine Fehler: {e}")
            return False

    def _setup_handlers(self):
        """Registriert Handler für Step-Typen"""
        self.step_handlers = {
            StepType.DECISION: self._handle_decision,
            StepType.RULE_CHECK: self._handle_rule_check,
            StepType.RL_ACTION: self._handle_rl_action,
            StepType.PLAN: self._handle_plan,
            StepType.KNOWLEDGE: self._handle_knowledge,
            StepType.AGENT_TASK: self._handle_agent_task,
            StepType.TOOL: self._handle_tool,
            StepType.WORKER: self._handle_worker,
            StepType.CONDITION: self._handle_condition,
            StepType.PARALLEL: self._handle_parallel,
            StepType.LOOP: self._handle_loop,
            StepType.CUSTOM: self._handle_custom,
        }

    def _setup_event_subscriptions(self):
        """Registriert für relevante Events"""
        self.event_bus.subscribe(EventType.JOB_COMPLETED, self._on_job_completed)
        self.event_bus.subscribe(EventType.JOB_FAILED, self._on_job_failed)

    def create_workflow(self,
                        name: str,
                        description: str = "",
                        steps: List[Dict] = None) -> Workflow:
        """
        Erstellt einen neuen Workflow

        Args:
            name: Name des Workflows
            description: Beschreibung
            steps: Liste von Step-Definitionen

        Returns:
            Erstellter Workflow
        """
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description
        )

        # Steps hinzufügen
        if steps:
            for i, step_def in enumerate(steps):
                step = WorkflowStep(
                    id=step_def.get("id", f"step_{i}"),
                    name=step_def.get("name", f"Step {i+1}"),
                    step_type=StepType(step_def.get("type", "custom")),
                    config=step_def.get("config", {}),
                    depends_on=step_def.get("depends_on", []),
                    condition=step_def.get("condition"),
                    max_retries=step_def.get("max_retries", 0)
                )
                workflow.steps.append(step)

        self.workflows[workflow_id] = workflow
        return workflow

    def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> bool:
        """
        Startet Workflow-Ausführung

        Args:
            workflow_id: ID des Workflows
            context: Initialer Kontext

        Returns:
            True wenn gestartet
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return False

        if context:
            workflow.context.update(context)

        workflow.status = "running"
        workflow.started_at = datetime.now()

        # In separatem Thread ausführen
        thread = threading.Thread(
            target=self._run_workflow,
            args=(workflow,),
            daemon=True
        )
        self.running_workflows[workflow_id] = thread
        thread.start()

        # Event emittieren
        self.event_bus.emit(
            EventType.PLAN_CREATED,
            "workflow_engine",
            {"workflow_id": workflow_id, "name": workflow.name}
        )

        return True

    def _run_workflow(self, workflow: Workflow):
        """Führt Workflow aus"""
        try:
            completed_steps = set()

            while True:
                # Finde nächsten ausführbaren Step
                next_step = self._find_next_step(workflow, completed_steps)

                if next_step is None:
                    # Keine weiteren Steps
                    break

                workflow.current_step = next_step.id

                # Step ausführen
                success = self._execute_step(workflow, next_step)

                if success:
                    completed_steps.add(next_step.id)

                    # Callback aufrufen
                    if workflow.on_step_complete:
                        workflow.on_step_complete(next_step)
                else:
                    if next_step.retries < next_step.max_retries:
                        next_step.retries += 1
                        time.sleep(next_step.retry_delay_s)
                        continue
                    else:
                        # Workflow fehlgeschlagen
                        workflow.status = "failed"
                        workflow.error = next_step.error
                        workflow.completed_at = datetime.now()

                        if workflow.on_error:
                            workflow.on_error(workflow, next_step)

                        self.event_bus.emit(
                            EventType.PLAN_FAILED,
                            "workflow_engine",
                            {"workflow_id": workflow.id, "step": next_step.id, "error": next_step.error}
                        )
                        return

            # Workflow erfolgreich
            workflow.status = "completed"
            workflow.completed_at = datetime.now()

            if workflow.on_complete:
                workflow.on_complete(workflow)

            self.event_bus.emit(
                EventType.PLAN_COMPLETED,
                "workflow_engine",
                {"workflow_id": workflow.id, "context": workflow.context}
            )

        except Exception as e:
            workflow.status = "failed"
            workflow.error = str(e)
            workflow.completed_at = datetime.now()

            self.event_bus.emit(
                EventType.PLAN_FAILED,
                "workflow_engine",
                {"workflow_id": workflow.id, "error": str(e)}
            )

    def _find_next_step(self, workflow: Workflow, completed: set) -> Optional[WorkflowStep]:
        """Findet den nächsten ausführbaren Step"""
        for step in workflow.steps:
            if step.id in completed:
                continue

            if step.status == StepStatus.SKIPPED:
                continue

            # Prüfe Abhängigkeiten
            deps_met = all(dep in completed for dep in step.depends_on)
            if not deps_met:
                continue

            # Prüfe Bedingung (sicher ohne eval)
            if step.condition:
                try:
                    evaluator = SafeExpressionEvaluator({"ctx": workflow.context})
                    if not evaluator.evaluate(step.condition):
                        step.status = StepStatus.SKIPPED
                        continue
                except (ValueError, Exception):
                    continue

            return step

        return None

    def _execute_step(self, workflow: Workflow, step: WorkflowStep) -> bool:
        """Führt einen Step aus"""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()

        self.event_bus.emit(
            EventType.PLAN_STEP_STARTED,
            "workflow_engine",
            {"workflow_id": workflow.id, "step_id": step.id, "step_type": step.step_type.value}
        )

        try:
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise ValueError(f"No handler for step type: {step.step_type}")

            result = handler(workflow, step)

            step.result = result
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now()

            # Result in Kontext speichern
            workflow.context[f"step_{step.id}_result"] = result

            self.event_bus.emit(
                EventType.PLAN_STEP_COMPLETED,
                "workflow_engine",
                {"workflow_id": workflow.id, "step_id": step.id, "result": result}
            )

            return True

        except Exception as e:
            step.error = str(e)
            step.status = StepStatus.FAILED
            step.completed_at = datetime.now()
            return False

    # ═══════════════════════════════════════════════════════════════
    # STEP HANDLERS
    # ═══════════════════════════════════════════════════════════════

    def _handle_decision(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Decision Engine Step aus"""
        from backend.decision import get_decision_engine

        engine = get_decision_engine()
        tree_name = step.config.get("tree", "default")
        context = {**workflow.context, **step.config.get("context", {})}

        decision = engine.decide(tree_name, context)

        # Ergebnis in Kontext
        workflow.context["last_decision"] = decision.action
        workflow.context["decision_confidence"] = decision.confidence

        return {
            "action": decision.action,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning
        }

    def _handle_rule_check(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Rule Engine Step aus"""
        from backend.decision import get_rule_engine

        engine = get_rule_engine()
        rule_set = step.config.get("rule_set", "default")
        context = {**workflow.context, **step.config.get("context", {})}
        execute = step.config.get("execute", False)

        result = engine.evaluate(rule_set, context, execute)

        workflow.context["rules_triggered"] = result.get("triggered_rules", [])

        return result

    def _handle_rl_action(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt RL Agent Step aus"""
        from backend.learning import get_rl_agent

        agent = get_rl_agent()
        action_space = step.config.get("action_space", "worker_selection")
        context = {**workflow.context, **step.config.get("context", {})}

        action = agent.select_action(context, action_space)

        workflow.context["rl_action"] = action.name
        workflow.context["rl_params"] = action.params

        return {
            "action": action.name,
            "params": action.params
        }

    def _handle_plan(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Planner Step aus"""
        from backend.planning import get_planner

        planner = get_planner()
        goal = step.config.get("goal", "")
        decomposition = step.config.get("decomposition", {})

        plan = planner.hierarchical_plan(goal, decomposition)

        workflow.context["plan_id"] = plan.id
        workflow.context["plan_steps"] = plan.steps

        return {
            "plan_id": plan.id,
            "steps": len(plan.steps)
        }

    def _handle_knowledge(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Knowledge Graph Step aus"""
        from backend.knowledge import get_knowledge_graph

        kg = get_knowledge_graph()
        operation = step.config.get("operation", "query")

        if operation == "query":
            results = kg.query(
                entity_type=step.config.get("entity_type"),
                relation_type=step.config.get("relation_type"),
                properties=step.config.get("properties", {})
            )
            return {"results": [e.id for e in results], "count": len(results)}

        elif operation == "add_entity":
            from backend.knowledge.knowledge_graph import Entity
            entity = Entity(
                id=step.config.get("entity_id", str(uuid.uuid4())[:8]),
                type=step.config.get("entity_type", "unknown"),
                name=step.config.get("name", ""),
                properties=step.config.get("properties", {})
            )
            success = kg.add_entity(entity)
            return {"success": success, "entity_id": entity.id}

        elif operation == "infer":
            new_relations = kg.infer()
            return {"inferred": len(new_relations)}

        return {}

    def _handle_agent_task(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Multi-Agent Task aus"""
        from backend.agents import get_multi_agent_system

        mas = get_multi_agent_system()
        description = step.config.get("description", "")
        params = {**workflow.context, **step.config.get("params", {})}

        task = mas.delegate_task(description, params)

        workflow.context["agent_task_id"] = task.id
        workflow.context["assigned_agent"] = task.assigned_to

        return {
            "task_id": task.id,
            "assigned_to": task.assigned_to
        }

    def _handle_tool(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Capability Tool aus"""
        from backend.capabilities import get_tool_registry

        registry = get_tool_registry()
        tool_id = step.config.get("tool_id", "")
        params = {**workflow.context, **step.config.get("params", {})}

        result = registry.execute(tool_id, params)

        if "error" in result:
            raise Exception(result["error"])

        return result

    def _handle_worker(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Worker Job aus"""
        from backend.services.job_queue import get_job_queue

        queue = get_job_queue()
        job_type = step.config.get("job_type")
        params = {**workflow.context, **step.config.get("params", {})}

        # Job erstellen und warten
        job_id = queue.submit(job_type, params)

        # Warten auf Ergebnis (vereinfacht)
        timeout = step.config.get("timeout", 300)
        start = time.time()

        while time.time() - start < timeout:
            job = queue.get_job(job_id)
            if job and job.status in ["completed", "failed"]:
                if job.status == "failed":
                    raise Exception(job.error or "Job failed")
                return {"job_id": job_id, "result": job.result}
            time.sleep(1)

        raise Exception("Job timeout")

    def _handle_condition(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Prüft Bedingung (sicher ohne eval)"""
        condition = step.config.get("condition", "True")

        try:
            evaluator = SafeExpressionEvaluator({"ctx": workflow.context})
            result = evaluator.evaluate(condition)
        except ValueError as e:
            result = False
            workflow.context["condition_error"] = str(e)

        workflow.context["condition_result"] = result

        return {"condition": condition, "result": result}

    def _handle_parallel(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Steps parallel aus"""
        sub_steps = step.config.get("steps", [])
        results = {}
        threads = []

        def run_sub_step(sub_step_config, key):
            sub_step = WorkflowStep(
                id=f"{step.id}_{key}",
                name=sub_step_config.get("name", key),
                step_type=StepType(sub_step_config.get("type", "custom")),
                config=sub_step_config.get("config", {})
            )
            success = self._execute_step(workflow, sub_step)
            results[key] = {"success": success, "result": sub_step.result}

        for i, sub_step_config in enumerate(sub_steps):
            key = sub_step_config.get("id", f"parallel_{i}")
            t = threading.Thread(target=run_sub_step, args=(sub_step_config, key))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return {"parallel_results": results}

    def _handle_loop(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """Führt Loop aus (sicher ohne eval)"""
        iterations = step.config.get("iterations", 1)
        condition = step.config.get("while_condition")
        sub_step_config = step.config.get("step", {})
        results = []

        i = 0
        while True:
            if condition:
                try:
                    evaluator = SafeExpressionEvaluator({"ctx": workflow.context, "i": i})
                    if not evaluator.evaluate(condition):
                        break
                except ValueError:
                    break  # Bei ungültiger Bedingung abbrechen
            elif i >= iterations:
                break

            workflow.context["loop_index"] = i

            sub_step = WorkflowStep(
                id=f"{step.id}_iter_{i}",
                name=f"Loop iteration {i}",
                step_type=StepType(sub_step_config.get("type", "custom")),
                config=sub_step_config.get("config", {})
            )

            success = self._execute_step(workflow, sub_step)
            results.append({"iteration": i, "success": success, "result": sub_step.result})

            if not success:
                break

            i += 1

        return {"iterations": len(results), "results": results}

    def _handle_custom(self, workflow: Workflow, step: WorkflowStep) -> Dict:
        """
        Führt Custom Handler aus dem Plugin-System aus

        Custom Handler werden über das WorkflowHandlerRegistry registriert:

            from backend.core.plugins import get_workflow_handlers

            handlers = get_workflow_handlers()

            @handlers.register("my_custom_handler", description="Mein Handler")
            def my_handler(workflow, step):
                # Zugriff auf workflow.context
                data = workflow.context.get('input_data', {})
                # Verarbeitung...
                return {"processed": True, "result": data}

        Verwendung in Workflow-Step:
            {"type": "custom", "config": {"handler": "my_custom_handler"}}
        """
        handler_name = step.config.get("handler")
        if not handler_name:
            return {"status": "no_handler", "error": "No handler name specified"}

        # Plugin-System für Custom Handler nutzen
        handlers = get_workflow_handlers()

        try:
            # Handler über Plugin-System ausführen
            result = handlers.execute(handler_name, workflow, step)
            return {"status": "executed", "handler": handler_name, "result": result}

        except PluginNotFoundError:
            return {"status": "error", "error": f"Handler '{handler_name}' not found"}

        except PluginDisabledError:
            return {"status": "error", "error": f"Handler '{handler_name}' is disabled"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════

    def _on_job_completed(self, event):
        """Handler für Job-Completion - Async Job-Handling"""
        import logging
        logger = logging.getLogger(__name__)

        job_id = event.data.get('job_id')
        result = event.data.get('result', {})

        if not job_id:
            return

        logger.debug(f"Job completed: {job_id}")

        # Finde den Workflow der diesen Job enthält
        with self._lock:
            for workflow_id, workflow in list(self._active_workflows.items()):
                for step in workflow.steps:
                    if step.job_id == job_id:
                        step.status = StepStatus.COMPLETED
                        step.result = result
                        step.end_time = time.time()
                        logger.info(f"Workflow {workflow_id} Step '{step.name}' completed")

                        # Prüfe ob nächster Step gestartet werden kann
                        self._try_start_next_steps(workflow)
                        break

    def _on_job_failed(self, event):
        """Handler für Job-Failure - Fehlerbehandlung"""
        import logging
        logger = logging.getLogger(__name__)

        job_id = event.data.get('job_id')
        error = event.data.get('error', 'Unknown error')

        if not job_id:
            return

        logger.error(f"Job failed: {job_id} - {error}")

        # Finde den Workflow der diesen Job enthält
        with self._lock:
            for workflow_id, workflow in list(self._active_workflows.items()):
                for step in workflow.steps:
                    if step.job_id == job_id:
                        step.status = StepStatus.FAILED
                        step.error = error
                        step.end_time = time.time()
                        logger.error(f"Workflow {workflow_id} Step '{step.name}' failed: {error}")

                        # Workflow als fehlgeschlagen markieren wenn kritischer Step
                        if step.critical:
                            workflow.status = WorkflowStatus.FAILED
                            workflow.error = f"Critical step '{step.name}' failed: {error}"
                            logger.error(f"Workflow {workflow_id} failed due to critical step failure")
                        break

    def _try_start_next_steps(self, workflow: 'Workflow'):
        """Versucht die nächsten Steps eines Workflows zu starten"""
        if workflow.status != WorkflowStatus.RUNNING:
            return

        for step in workflow.steps:
            if step.status != StepStatus.PENDING:
                continue

            # Prüfe ob alle Abhängigkeiten erfüllt sind
            deps_met = True
            for dep_name in step.depends_on:
                dep_step = next((s for s in workflow.steps if s.name == dep_name), None)
                if dep_step and dep_step.status != StepStatus.COMPLETED:
                    deps_met = False
                    break

            if deps_met:
                # Starte den Step
                self._execute_step(workflow, step)

    # ═══════════════════════════════════════════════════════════════
    # PREDEFINED WORKFLOWS
    # ═══════════════════════════════════════════════════════════════

    def create_image_generation_workflow(self, prompt: str) -> Workflow:
        """Erstellt Image Generation Workflow"""
        return self.create_workflow(
            name="Image Generation",
            description="Generiert ein Bild mit optimaler Konfiguration",
            steps=[
                {
                    "id": "decide_model",
                    "name": "Model auswählen",
                    "type": "rl_action",
                    "config": {"action_space": "worker_selection"}
                },
                {
                    "id": "check_resources",
                    "name": "Ressourcen prüfen",
                    "type": "rule_check",
                    "config": {"rule_set": "resource"},
                    "depends_on": ["decide_model"]
                },
                {
                    "id": "generate",
                    "name": "Bild generieren",
                    "type": "tool",
                    "config": {"tool_id": "gen_img_txt2img", "params": {"prompt": prompt}},
                    "depends_on": ["check_resources"]
                },
                {
                    "id": "record_metrics",
                    "name": "Metriken aufzeichnen",
                    "type": "knowledge",
                    "config": {
                        "operation": "add_entity",
                        "entity_type": "generation",
                        "name": f"image_{prompt[:20]}"
                    },
                    "depends_on": ["generate"]
                }
            ]
        )

    def create_document_processing_workflow(self, document_path: str) -> Workflow:
        """Erstellt Document Processing Workflow"""
        return self.create_workflow(
            name="Document Processing",
            description="Verarbeitet ein Dokument komplett",
            steps=[
                {
                    "id": "extract_text",
                    "name": "Text extrahieren",
                    "type": "tool",
                    "config": {"tool_id": "doc_pdf_read", "params": {"file": document_path}}
                },
                {
                    "id": "analyze_content",
                    "name": "Inhalt analysieren",
                    "type": "parallel",
                    "config": {
                        "steps": [
                            {"name": "Entities", "type": "tool", "config": {"tool_id": "nlp_entity_extract"}},
                            {"name": "Summary", "type": "tool", "config": {"tool_id": "nlp_text_summarize"}},
                            {"name": "Sentiment", "type": "tool", "config": {"tool_id": "nlp_sentiment_analyze"}}
                        ]
                    },
                    "depends_on": ["extract_text"]
                },
                {
                    "id": "store_knowledge",
                    "name": "Wissen speichern",
                    "type": "knowledge",
                    "config": {"operation": "add_entity", "entity_type": "document"},
                    "depends_on": ["analyze_content"]
                }
            ]
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        status_counts = {}
        for wf in self.workflows.values():
            status_counts[wf.status] = status_counts.get(wf.status, 0) + 1

        handlers = get_workflow_handlers()

        return {
            "total_workflows": len(self.workflows),
            "running_workflows": len(self.running_workflows),
            "status_distribution": status_counts,
            "step_handlers": len(self.step_handlers),
            "custom_handlers": len(handlers.list_plugins())
        }

    # ═══════════════════════════════════════════════════════════════
    # CUSTOM HANDLER REGISTRATION
    # ═══════════════════════════════════════════════════════════════

    def register_custom_handler(self,
                                 name: str,
                                 handler_func: Callable,
                                 description: str = "",
                                 version: str = "1.0.0"):
        """
        Registriert einen Custom Handler für Workflows

        Args:
            name: Eindeutiger Handler-Name
            handler_func: Funktion(workflow, step) -> Dict
            description: Beschreibung des Handlers
            version: Version des Handlers

        Beispiel:
            def my_handler(workflow, step):
                data = workflow.context.get('data')
                return {"processed": True}

            engine.register_custom_handler("my_handler", my_handler, "Verarbeitet Daten")
        """
        handlers = get_workflow_handlers()
        handlers.register_handler(name, handler_func, version, description)

    def list_custom_handlers(self) -> List[Dict[str, Any]]:
        """Gibt Liste aller registrierten Custom Handler zurück"""
        handlers = get_workflow_handlers()
        return handlers.list_plugins()

    def get_custom_handler_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Gibt Info über einen spezifischen Handler zurück"""
        handlers = get_workflow_handlers()
        plugin = handlers.get(name)
        if plugin:
            return {
                'name': plugin.name,
                'version': plugin.version,
                'description': plugin.description,
                'author': plugin.author,
                'enabled': plugin.enabled,
                'registered_at': plugin.registered_at.isoformat(),
                'metadata': plugin.metadata
            }
        return None

    def enable_custom_handler(self, name: str) -> bool:
        """Aktiviert einen Custom Handler"""
        handlers = get_workflow_handlers()
        return handlers.enable(name)

    def disable_custom_handler(self, name: str) -> bool:
        """Deaktiviert einen Custom Handler"""
        handlers = get_workflow_handlers()
        return handlers.disable(name)


# Singleton
_workflow_engine: Optional[WorkflowEngine] = None

def get_workflow_engine() -> WorkflowEngine:
    """Gibt Singleton-Instanz zurück"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine
