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
    SCIO Event Bus

    Zentrales Pub/Sub-System für alle Module:
    - Asynchrone Event-Verarbeitung
    - Pattern-basierte Subscriptions
    - Event-History
    - Prioritäts-basierte Verarbeitung
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.pattern_subscribers: List[tuple] = []  # (pattern, callback)
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.event_history: List[Event] = []
        self.history_limit = 1000

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_counter = 0
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert den Event Bus"""
        try:
            self._initialized = True
            self.start()
            print("[OK] Event Bus initialisiert")
            return True
        except Exception as e:
            print(f"[ERROR] Event Bus Fehler: {e}")
            return False

    def start(self):
        """Startet die Event-Verarbeitung"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._thread.start()

    def stop(self):
        """Stoppt die Event-Verarbeitung"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

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
        """Event-Processing Loop"""
        while self._running:
            try:
                # Timeout damit wir shutdown erkennen können
                try:
                    _, _, event = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Event in History speichern
                self._store_event(event)

                # Direkte Subscribers benachrichtigen
                for callback in self.subscribers.get(event.type.value, []):
                    try:
                        callback(event)
                    except Exception as e:
                        print(f"[WARN] Event callback error: {e}")

                # Pattern-Subscribers prüfen
                for pattern, callback in self.pattern_subscribers:
                    if self._matches_pattern(event.type.value, pattern):
                        try:
                            callback(event)
                        except Exception as e:
                            print(f"[WARN] Pattern callback error: {e}")

            except Exception as e:
                print(f"[ERROR] Event processing error: {e}")

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
        """Gibt Statistiken zurück"""
        type_counts = defaultdict(int)
        source_counts = defaultdict(int)

        for event in self.event_history:
            type_counts[event.type.value] += 1
            source_counts[event.source] += 1

        return {
            "total_events": self._event_counter,
            "history_size": len(self.event_history),
            "subscribers": len(self.subscribers),
            "pattern_subscribers": len(self.pattern_subscribers),
            "queue_size": self.event_queue.qsize(),
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
