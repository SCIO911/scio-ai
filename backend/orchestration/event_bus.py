#!/usr/bin/env python3
"""
SCIO - Event Bus
Zentrales Event-System für Kommunikation zwischen allen Modulen
"""

import threading
import queue
import time
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict


class EventType(str, Enum):
    """Typen von Events im System"""
    # System Events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_HEALTH = "system.health"

    # Job Events
    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_PROGRESS = "job.progress"

    # Decision Events
    DECISION_MADE = "decision.made"
    DECISION_FEEDBACK = "decision.feedback"
    RULE_TRIGGERED = "rule.triggered"

    # Learning Events
    LEARNING_OBSERVATION = "learning.observation"
    LEARNING_UPDATE = "learning.update"
    REWARD_CALCULATED = "reward.calculated"

    # Planning Events
    PLAN_CREATED = "plan.created"
    PLAN_STEP_STARTED = "plan.step.started"
    PLAN_STEP_COMPLETED = "plan.step.completed"
    PLAN_COMPLETED = "plan.completed"
    PLAN_FAILED = "plan.failed"

    # Knowledge Events
    ENTITY_CREATED = "knowledge.entity.created"
    RELATION_CREATED = "knowledge.relation.created"
    INFERENCE_COMPLETED = "knowledge.inference.completed"

    # Agent Events
    AGENT_REGISTERED = "agent.registered"
    AGENT_MESSAGE = "agent.message"
    TASK_DELEGATED = "task.delegated"
    TASK_COMPLETED = "task.completed"

    # Monitoring Events
    METRIC_RECORDED = "monitoring.metric"
    DRIFT_DETECTED = "monitoring.drift"
    SLA_VIOLATION = "monitoring.sla.violation"
    ALERT_CREATED = "monitoring.alert"

    # Capability Events
    TOOL_EXECUTED = "capability.tool.executed"
    TOOL_FAILED = "capability.tool.failed"
    CHAIN_STARTED = "capability.chain.started"
    CHAIN_COMPLETED = "capability.chain.completed"

    # Worker Events
    WORKER_BUSY = "worker.busy"
    WORKER_IDLE = "worker.idle"
    WORKER_ERROR = "worker.error"
    MODEL_LOADED = "worker.model.loaded"
    MODEL_UNLOADED = "worker.model.unloaded"

    # Custom
    CUSTOM = "custom"


@dataclass
class Event:
    """Ein Event im System"""
    id: str
    type: EventType
    source: str  # Modul das Event erzeugt hat
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # Für zusammenhängende Events
    priority: int = 0  # Höher = wichtiger

    def __lt__(self, other):
        """Vergleich für PriorityQueue (nach timestamp)"""
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp < other.timestamp

    def __le__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp <= other.timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority
        }


class EventBus:
    """
    SCIO Event Bus (Optimiert)

    Zentrales Pub/Sub-System für alle Module:
    - Asynchrone Event-Verarbeitung mit Thread Pool
    - Pattern-basierte Subscriptions
    - Begrenzte Event-History
    - Prioritäts-basierte Verarbeitung
    - Non-blocking Callback-Ausführung
    """

    # Konstanten
    HISTORY_LIMIT = 1000
    CALLBACK_POOL_SIZE = 10
    CALLBACK_TIMEOUT = 2.0  # Sekunden
    QUEUE_MAX_SIZE = 10000

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.pattern_subscribers: List[tuple] = []  # (pattern, callback)
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.QUEUE_MAX_SIZE)
        self.event_history: List[Event] = []
        self.history_limit = self.HISTORY_LIMIT

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback_threads: List[threading.Thread] = []
        self._callback_queue: queue.Queue = queue.Queue()
        self._event_counter = 0
        self._lock = threading.Lock()
        self._initialized = False

        # Statistiken
        self._events_processed = 0
        self._callbacks_executed = 0
        self._callbacks_failed = 0
        self._queue_full_count = 0

    def initialize(self) -> bool:
        """Initialisiert den Event Bus"""
        try:
            self._initialized = True
            self.start()
            print("[OK] Event Bus initialisiert (Thread Pool aktiviert)")
            return True
        except Exception as e:
            print(f"[ERROR] Event Bus Fehler: {e}")
            return False

    def start(self):
        """Startet die Event-Verarbeitung mit Thread Pool"""
        if self._running:
            return

        self._running = True

        # Haupt-Event-Processing-Thread
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._thread.start()

        # Callback-Thread-Pool starten
        for i in range(self.CALLBACK_POOL_SIZE):
            t = threading.Thread(target=self._callback_worker, daemon=True)
            t.start()
            self._callback_threads.append(t)

    def stop(self):
        """Stoppt die Event-Verarbeitung"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        for t in self._callback_threads:
            t.join(timeout=1)
        self._callback_threads.clear()

    def subscribe(self, event_type: EventType, callback: Callable):
        """
        Registriert Callback für Event-Typ

        Args:
            event_type: Der Event-Typ
            callback: Funktion die aufgerufen wird (erhält Event)
        """
        self.subscribers[event_type.value].append(callback)

    def subscribe_pattern(self, pattern: str, callback: Callable):
        """
        Registriert Callback für Event-Pattern (z.B. "job.*")

        Args:
            pattern: Pattern mit Wildcards
            callback: Funktion die aufgerufen wird
        """
        self.pattern_subscribers.append((pattern, callback))

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Entfernt Subscription"""
        if event_type.value in self.subscribers:
            self.subscribers[event_type.value] = [
                cb for cb in self.subscribers[event_type.value]
                if cb != callback
            ]

    def publish(self, event: Event):
        """
        Veröffentlicht ein Event

        Args:
            event: Das zu veröffentlichende Event
        """
        # Priorität negieren für PriorityQueue (höher = früher)
        self.event_queue.put((-event.priority, time.time(), event))

    def emit(self,
             event_type: EventType,
             source: str,
             data: Dict[str, Any] = None,
             correlation_id: str = None,
             priority: int = 0) -> str:
        """
        Erzeugt und veröffentlicht ein Event (Convenience-Methode)

        Returns:
            Event-ID
        """
        with self._lock:
            self._event_counter += 1
            event_id = f"evt_{self._event_counter}"

        event = Event(
            id=event_id,
            type=event_type,
            source=source,
            data=data or {},
            correlation_id=correlation_id,
            priority=priority
        )

        self.publish(event)
        return event_id

    def _process_events(self):
        """Event-Processing Loop (non-blocking)"""
        while self._running:
            try:
                # Timeout damit wir shutdown erkennen können
                try:
                    _, _, event = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self._events_processed += 1

                # Event in History speichern
                self._store_event(event)

                # Callbacks an Thread Pool delegieren (non-blocking!)
                callbacks = list(self.subscribers.get(event.type.value, []))

                # Pattern-Subscribers hinzufügen
                for pattern, callback in self.pattern_subscribers:
                    if self._matches_pattern(event.type.value, pattern):
                        callbacks.append(callback)

                # Alle Callbacks zur Queue hinzufügen
                for callback in callbacks:
                    try:
                        self._callback_queue.put_nowait((callback, event))
                    except queue.Full:
                        # Queue voll - Callback überspringen aber loggen
                        self._callbacks_failed += 1

            except Exception as e:
                print(f"[ERROR] Event processing error: {e}")

    def _callback_worker(self):
        """Worker-Thread für Callback-Ausführung"""
        while self._running:
            try:
                try:
                    callback, event = self._callback_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    callback(event)
                    self._callbacks_executed += 1
                except Exception as e:
                    self._callbacks_failed += 1
                    # Nur debug-loggen um Spam zu vermeiden
                    import logging
                    logging.getLogger(__name__).debug(f"Callback error: {e}")

            except Exception as e:
                print(f"[ERROR] Callback worker error: {e}")

    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Prüft ob Event-Typ zu Pattern passt"""
        if pattern == "*":
            return True

        pattern_parts = pattern.split(".")
        type_parts = event_type.split(".")

        for i, part in enumerate(pattern_parts):
            if part == "*":
                continue
            if i >= len(type_parts):
                return False
            if part != type_parts[i]:
                return False

        return True

    def _store_event(self, event: Event):
        """Speichert Event in History"""
        self.event_history.append(event)
        if len(self.event_history) > self.history_limit:
            self.event_history = self.event_history[-self.history_limit:]

    def get_history(self,
                    event_type: EventType = None,
                    source: str = None,
                    limit: int = 100) -> List[Event]:
        """Gibt Event-History zurück"""
        events = self.event_history

        if event_type:
            events = [e for e in events if e.type == event_type]

        if source:
            events = [e for e in events if e.source == source]

        return events[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt erweiterte Statistiken zurück"""
        type_counts = defaultdict(int)
        source_counts = defaultdict(int)

        for event in self.event_history:
            type_counts[event.type.value] += 1
            source_counts[event.source] += 1

        return {
            "total_events": self._event_counter,
            "events_processed": self._events_processed,
            "history_size": len(self.event_history),
            "history_limit": self.history_limit,
            "subscribers": len(self.subscribers),
            "pattern_subscribers": len(self.pattern_subscribers),
            "queue_size": self.event_queue.qsize(),
            "callback_queue_size": self._callback_queue.qsize(),
            "callback_threads": len(self._callback_threads),
            "callbacks_executed": self._callbacks_executed,
            "callbacks_failed": self._callbacks_failed,
            "running": self._running,
            "events_by_type": dict(type_counts),
            "events_by_source": dict(source_counts)
        }


# Singleton
_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """Gibt Singleton-Instanz zurück"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
