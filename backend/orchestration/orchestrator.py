#!/usr/bin/env python3
"""
SCIO - Orchestrator
Zentrale Koordination aller AI-Module und Services
"""

import threading
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from .event_bus import EventBus, Event, EventType, get_event_bus
from .workflow_engine import WorkflowEngine, get_workflow_engine


class ModuleStatus(str, Enum):
    """Status eines Moduls"""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class ModuleInfo:
    """Information über ein registriertes Modul"""
    name: str
    module_type: str
    status: ModuleStatus = ModuleStatus.UNKNOWN
    instance: Any = None
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """
    SCIO Orchestrator

    Zentrale Koordinationsinstanz für alle Module:
    - Modul-Lifecycle-Management
    - Cross-Modul-Kommunikation via Event Bus
    - Automatische Feedback-Loops
    - Health Monitoring
    - Unified System Interface
    """

    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self.event_bus: EventBus = get_event_bus()
        self.workflow_engine: WorkflowEngine = get_workflow_engine()

        self._running = False
        self._health_thread: Optional[threading.Thread] = None
        self._initialized = False
        self._lock = threading.Lock()

        # Automatische Event-Handler
        self._auto_handlers: Dict[str, Callable] = {}

    def initialize(self) -> bool:
        """Initialisiert den Orchestrator und alle Module"""
        try:
            print("\n" + "=" * 60)
            print("    SCIO ORCHESTRATOR - Initialisierung")
            print("=" * 60 + "\n")

            # Event Bus starten
            if not self.event_bus._initialized:
                self.event_bus.initialize()

            # Workflow Engine initialisieren
            if not self.workflow_engine._initialized:
                self.workflow_engine.initialize()

            # Module registrieren und initialisieren
            self._register_all_modules()

            # Event-Subscriptions einrichten
            self._setup_event_subscriptions()

            # Health Monitoring starten
            self._start_health_monitoring()

            self._initialized = True
            self._running = True

            # System Startup Event
            self.event_bus.emit(
                EventType.SYSTEM_STARTUP,
                "orchestrator",
                {"modules": list(self.modules.keys())}
            )

            print("\n[OK] SCIO Orchestrator initialisiert")
            print(f"     {len(self.modules)} Module registriert")

            return True

        except Exception as e:
            print(f"[ERROR] Orchestrator Initialisierung: {e}")
            return False

    def _register_all_modules(self):
        """Registriert alle SCIO Module"""

        # AI Modules
        try:
            from backend.ai_modules import get_decision_engine
            engine = get_decision_engine()
            self.register_module("decision_engine", "decision", engine)
        except Exception as e:
            print(f"  [WARN] Decision Engine: {e}")

        try:
            from backend.ai_modules import get_rule_engine
            engine = get_rule_engine()
            self.register_module("rule_engine", "rules", engine)
        except Exception as e:
            print(f"  [WARN] Rule Engine: {e}")

        try:
            from backend.ai_modules import get_rl_agent
            agent = get_rl_agent()
            self.register_module("rl_agent", "reinforcement", agent)
        except Exception as e:
            print(f"  [WARN] RL Agent: {e}")

        try:
            from backend.ai_modules import get_continuous_learner
            learner = get_continuous_learner()
            self.register_module("continuous_learner", "learning", learner)
        except Exception as e:
            print(f"  [WARN] Continuous Learner: {e}")

        try:
            from backend.ai_modules import get_planner
            planner = get_planner()
            self.register_module("planner", "planning", planner)
        except Exception as e:
            print(f"  [WARN] Planner: {e}")

        try:
            from backend.ai_modules import get_knowledge_graph
            kg = get_knowledge_graph()
            self.register_module("knowledge_graph", "knowledge", kg)
        except Exception as e:
            print(f"  [WARN] Knowledge Graph: {e}")

        try:
            from backend.ai_modules import get_multi_agent_system
            mas = get_multi_agent_system()
            self.register_module("multi_agent_system", "agents", mas)
        except Exception as e:
            print(f"  [WARN] Multi-Agent System: {e}")

        try:
            from backend.ai_modules import get_drift_detector
            dd = get_drift_detector()
            self.register_module("drift_detector", "monitoring", dd)
        except Exception as e:
            print(f"  [WARN] Drift Detector: {e}")

        try:
            from backend.ai_modules import get_performance_tracker
            pt = get_performance_tracker()
            self.register_module("performance_tracker", "monitoring", pt)
        except Exception as e:
            print(f"  [WARN] Performance Tracker: {e}")

        # Capabilities
        try:
            from backend.capabilities import get_capability_engine, get_tool_registry
            cap_engine = get_capability_engine()
            registry = get_tool_registry()
            self.register_module("capability_engine", "capabilities", cap_engine)
            self.register_module("tool_registry", "capabilities", registry)
        except Exception as e:
            print(f"  [WARN] Capabilities: {e}")

        # Workers
        try:
            from backend.workers import get_worker_manager
            wm = get_worker_manager()
            self.register_module("worker_manager", "workers", wm)
        except Exception as e:
            print(f"  [WARN] Worker Manager: {e}")

    def register_module(self, name: str, module_type: str, instance: Any):
        """Registriert ein Modul"""
        with self._lock:
            self.modules[name] = ModuleInfo(
                name=name,
                module_type=module_type,
                status=ModuleStatus.READY,
                instance=instance,
                last_health_check=datetime.now()
            )
            print(f"  [+] {name} ({module_type})")

    def _setup_event_subscriptions(self):
        """Richtet automatische Event-Handler ein"""

        # Job Completion -> RL Feedback
        self.event_bus.subscribe(EventType.JOB_COMPLETED, self._on_job_completed)
        self.event_bus.subscribe(EventType.JOB_FAILED, self._on_job_failed)

        # Decision Made -> Knowledge Graph Update
        self.event_bus.subscribe(EventType.DECISION_MADE, self._on_decision_made)

        # Learning Updates -> Performance Tracking
        self.event_bus.subscribe(EventType.LEARNING_UPDATE, self._on_learning_update)

        # Drift Detection -> Alerts
        self.event_bus.subscribe(EventType.DRIFT_DETECTED, self._on_drift_detected)

        # Tool Execution -> Metrics
        self.event_bus.subscribe(EventType.TOOL_EXECUTED, self._on_tool_executed)
        self.event_bus.subscribe(EventType.TOOL_FAILED, self._on_tool_failed)

        # Worker Events -> Resource Management
        self.event_bus.subscribe(EventType.WORKER_BUSY, self._on_worker_busy)
        self.event_bus.subscribe(EventType.WORKER_IDLE, self._on_worker_idle)

        # Pattern-basierte Subscriptions
        self.event_bus.subscribe_pattern("job.*", self._on_any_job_event)
        self.event_bus.subscribe_pattern("monitoring.*", self._on_any_monitoring_event)

        print("  [OK] Event Subscriptions eingerichtet")

    def _on_job_completed(self, event: Event):
        """Handler für Job-Completion -> RL Feedback"""
        try:
            if "rl_agent" in self.modules:
                agent = self.modules["rl_agent"].instance
                job_data = event.data

                # Reward basierend auf Ergebnis
                reward = job_data.get("quality_score", 0.8)
                state = job_data.get("state", {})
                action = job_data.get("action", "process")

                # RL Update
                if hasattr(agent, 'update'):
                    agent.update(state, action, reward, {}, False)

                self.modules["rl_agent"].success_count += 1

        except Exception as e:
            print(f"[WARN] RL Feedback Error: {e}")

    def _on_job_failed(self, event: Event):
        """Handler für Job-Failure -> Negative Feedback"""
        try:
            if "rl_agent" in self.modules:
                agent = self.modules["rl_agent"].instance
                job_data = event.data

                # Negative Reward
                reward = -0.5
                state = job_data.get("state", {})
                action = job_data.get("action", "process")

                if hasattr(agent, 'update'):
                    agent.update(state, action, reward, {}, True)

                self.modules["rl_agent"].error_count += 1

        except Exception as e:
            print(f"[WARN] RL Failure Feedback Error: {e}")

    def _on_decision_made(self, event: Event):
        """Handler für Decision -> Knowledge Graph"""
        try:
            if "knowledge_graph" in self.modules:
                kg = self.modules["knowledge_graph"].instance
                decision_data = event.data

                # Entität für Decision erstellen
                if hasattr(kg, 'add_entity'):
                    kg.add_entity(
                        entity_id=f"decision_{uuid.uuid4().hex[:8]}",
                        entity_type="decision",
                        properties={
                            "name": decision_data.get("name", "unknown"),
                            "action": decision_data.get("action", "unknown"),
                            "confidence": decision_data.get("confidence", 0.0),
                            "timestamp": datetime.now().isoformat()
                        }
                    )

        except Exception as e:
            print(f"[WARN] Knowledge Graph Update Error: {e}")

    def _on_learning_update(self, event: Event):
        """Handler für Learning -> Performance Tracking"""
        try:
            if "performance_tracker" in self.modules:
                tracker = self.modules["performance_tracker"].instance
                learning_data = event.data

                if hasattr(tracker, 'record_metric'):
                    tracker.record_metric(
                        name="learning_update",
                        value=learning_data.get("improvement", 0.0),
                        tags={"source": event.source}
                    )

        except Exception as e:
            print(f"[WARN] Performance Tracking Error: {e}")

    def _on_drift_detected(self, event: Event):
        """Handler für Drift Detection -> Alerts"""
        try:
            drift_data = event.data
            severity = drift_data.get("severity", "medium")

            # Alert erstellen
            self.event_bus.emit(
                EventType.ALERT_CREATED,
                "orchestrator",
                {
                    "type": "drift_detected",
                    "severity": severity,
                    "message": drift_data.get("message", "Data drift detected"),
                    "details": drift_data
                },
                priority=2 if severity == "high" else 1
            )

            # Bei kritischem Drift: Continuous Learner triggern
            if severity == "high" and "continuous_learner" in self.modules:
                learner = self.modules["continuous_learner"].instance
                if hasattr(learner, 'trigger_retraining'):
                    learner.trigger_retraining(drift_data)

        except Exception as e:
            print(f"[WARN] Drift Alert Error: {e}")

    def _on_tool_executed(self, event: Event):
        """Handler für Tool-Execution -> Metrics"""
        try:
            if "performance_tracker" in self.modules:
                tracker = self.modules["performance_tracker"].instance
                tool_data = event.data

                if hasattr(tracker, 'record_metric'):
                    tracker.record_metric(
                        name="tool_execution",
                        value=tool_data.get("execution_time_ms", 0),
                        tags={
                            "tool_id": tool_data.get("tool_id", "unknown"),
                            "success": "true"
                        }
                    )

        except Exception as e:
            print(f"[WARN] Tool Metrics Error: {e}")

    def _on_tool_failed(self, event: Event):
        """Handler für Tool-Failure"""
        try:
            if "performance_tracker" in self.modules:
                tracker = self.modules["performance_tracker"].instance
                tool_data = event.data

                if hasattr(tracker, 'record_metric'):
                    tracker.record_metric(
                        name="tool_failure",
                        value=1,
                        tags={
                            "tool_id": tool_data.get("tool_id", "unknown"),
                            "error": tool_data.get("error", "unknown")
                        }
                    )

        except Exception as e:
            print(f"[WARN] Tool Failure Metrics Error: {e}")

    def _on_worker_busy(self, event: Event):
        """Handler für Worker Busy Events"""
        worker_data = event.data
        worker_type = worker_data.get("worker_type", "unknown")

        # Update Module Status
        for name, info in self.modules.items():
            if info.module_type == "workers":
                info.status = ModuleStatus.BUSY

    def _on_worker_idle(self, event: Event):
        """Handler für Worker Idle Events"""
        for name, info in self.modules.items():
            if info.module_type == "workers":
                info.status = ModuleStatus.READY

    def _on_any_job_event(self, event: Event):
        """Handler für alle Job-Events"""
        # Logging für Debugging
        pass

    def _on_any_monitoring_event(self, event: Event):
        """Handler für alle Monitoring-Events"""
        # Kann für zentrale Monitoring-Logik verwendet werden
        pass

    def _start_health_monitoring(self):
        """Startet Health-Monitoring Thread"""
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self._health_thread.start()
        print("  [OK] Health Monitoring gestartet")

    def _health_monitor_loop(self):
        """Health Check Loop"""
        while self._running:
            try:
                self._check_module_health()
                time.sleep(30)  # Alle 30 Sekunden
            except Exception as e:
                print(f"[WARN] Health Monitor Error: {e}")

    def _check_module_health(self):
        """Prüft Gesundheit aller Module"""
        for name, info in self.modules.items():
            try:
                instance = info.instance

                # Generischer Health Check
                if hasattr(instance, 'get_statistics'):
                    stats = instance.get_statistics()
                    info.last_health_check = datetime.now()
                    info.status = ModuleStatus.READY
                elif hasattr(instance, 'health_check'):
                    healthy = instance.health_check()
                    info.status = ModuleStatus.READY if healthy else ModuleStatus.ERROR
                    info.last_health_check = datetime.now()

            except Exception as e:
                info.status = ModuleStatus.ERROR
                info.error_count += 1

        # Health Event emittieren
        self.event_bus.emit(
            EventType.SYSTEM_HEALTH,
            "orchestrator",
            self.get_health_summary()
        )

    def get_health_summary(self) -> Dict[str, Any]:
        """Gibt Health-Zusammenfassung zurück"""
        module_status = {}
        for name, info in self.modules.items():
            module_status[name] = {
                "status": info.status.value,
                "type": info.module_type,
                "errors": info.error_count,
                "successes": info.success_count,
                "last_check": info.last_health_check.isoformat() if info.last_health_check else None
            }

        healthy_count = sum(1 for info in self.modules.values() if info.status == ModuleStatus.READY)

        return {
            "overall_status": "healthy" if healthy_count == len(self.modules) else "degraded",
            "total_modules": len(self.modules),
            "healthy_modules": healthy_count,
            "modules": module_status
        }

    def get_module(self, name: str) -> Optional[Any]:
        """Gibt Modul-Instanz zurück"""
        if name in self.modules:
            return self.modules[name].instance
        return None

    def execute_workflow(self, workflow_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Führt einen vordefinierten Workflow aus"""
        return self.workflow_engine.execute_predefined(workflow_name, context or {})

    def process_request(self, request_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verarbeitet eine Anfrage mit automatischer Modul-Koordination

        Args:
            request_type: Art der Anfrage (z.B. "analyze", "generate", "transform")
            data: Anfrage-Daten

        Returns:
            Verarbeitungsergebnis
        """
        correlation_id = str(uuid.uuid4())

        # Event für Request-Start
        self.event_bus.emit(
            EventType.JOB_CREATED,
            "orchestrator",
            {"request_type": request_type, "data": data},
            correlation_id=correlation_id
        )

        try:
            result = {}

            # 1. Decision Engine konsultieren
            if "decision_engine" in self.modules:
                engine = self.modules["decision_engine"].instance
                if hasattr(engine, 'make_decision'):
                    decision = engine.make_decision(
                        f"Process {request_type} request",
                        data
                    )
                    result["decision"] = {
                        "action": decision.action,
                        "confidence": decision.confidence
                    }

            # 2. Passende Tools finden
            if "capability_engine" in self.modules:
                cap_engine = self.modules["capability_engine"].instance
                if hasattr(cap_engine, 'find_tools'):
                    tools = cap_engine.find_tools(request_type, limit=5)
                    result["suggested_tools"] = [
                        {"tool": t.tool.id, "confidence": t.confidence}
                        for t in tools
                    ]

            # 3. Plan erstellen
            if "planner" in self.modules:
                planner = self.modules["planner"].instance
                if hasattr(planner, 'create_plan'):
                    plan = planner.create_plan(
                        goal=f"Execute {request_type}",
                        context=data
                    )
                    result["plan"] = {
                        "steps": len(plan.steps) if hasattr(plan, 'steps') else 0,
                        "status": plan.status.value if hasattr(plan, 'status') else "created"
                    }

            # Event für Request-Completion
            self.event_bus.emit(
                EventType.JOB_COMPLETED,
                "orchestrator",
                {"request_type": request_type, "result": result},
                correlation_id=correlation_id
            )

            return {
                "success": True,
                "correlation_id": correlation_id,
                "result": result
            }

        except Exception as e:
            self.event_bus.emit(
                EventType.JOB_FAILED,
                "orchestrator",
                {"request_type": request_type, "error": str(e)},
                correlation_id=correlation_id
            )

            return {
                "success": False,
                "correlation_id": correlation_id,
                "error": str(e)
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Orchestrator-Statistiken zurück"""
        return {
            "running": self._running,
            "initialized": self._initialized,
            "total_modules": len(self.modules),
            "modules_by_type": self._count_modules_by_type(),
            "event_bus": self.event_bus.get_statistics(),
            "workflow_engine": self.workflow_engine.get_statistics(),
            "health": self.get_health_summary()
        }

    def _count_modules_by_type(self) -> Dict[str, int]:
        """Zählt Module nach Typ"""
        counts = {}
        for info in self.modules.values():
            counts[info.module_type] = counts.get(info.module_type, 0) + 1
        return counts

    def stop(self):
        """Stoppt den Orchestrator"""
        print("\n[INFO] SCIO Orchestrator wird beendet...")

        self._running = False

        # Shutdown Event
        self.event_bus.emit(
            EventType.SYSTEM_SHUTDOWN,
            "orchestrator",
            {"modules": list(self.modules.keys())}
        )

        # Event Bus stoppen
        self.event_bus.stop()

        # Workflow Engine stoppen
        self.workflow_engine.stop()

        print("[OK] SCIO Orchestrator beendet")


# Singleton
_orchestrator: Optional[Orchestrator] = None

def get_orchestrator() -> Orchestrator:
    """Gibt Singleton-Instanz zurück"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
