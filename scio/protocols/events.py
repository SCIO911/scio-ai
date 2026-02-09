"""
Event Protocol
==============

Event-basierte Kommunikation und Pub/Sub-System.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum, auto
from datetime import datetime
import uuid
import threading
import asyncio
from collections import defaultdict


class EventType(Enum):
    """Standard-Eventtypen"""
    # Lifecycle Events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    SYSTEM_ERROR = auto()

    # Agent Events
    AGENT_CREATED = auto()
    AGENT_STARTED = auto()
    AGENT_STOPPED = auto()
    AGENT_ERROR = auto()
    AGENT_MESSAGE = auto()

    # Execution Events
    EXECUTION_START = auto()
    EXECUTION_PROGRESS = auto()
    EXECUTION_COMPLETE = auto()
    EXECUTION_ERROR = auto()
    EXECUTION_CANCELLED = auto()

    # Data Events
    DATA_RECEIVED = auto()
    DATA_PROCESSED = auto()
    DATA_ERROR = auto()

    # Tool Events
    TOOL_INVOKED = auto()
    TOOL_COMPLETE = auto()
    TOOL_ERROR = auto()

    # Custom Events
    CUSTOM = auto()


@dataclass
class Event:
    """Basis-Event"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.CUSTOM
    source: str = ""
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    propagate: bool = True  # Ob Event weitergeleitet werden soll

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "id": self.id,
            "type": self.type.name,
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "tags": list(self.tags),
            "propagate": self.propagate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Erstellt Event aus Dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=EventType[data.get("type", "CUSTOM")],
            source=data.get("source", ""),
            data=data.get("data"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            tags=set(data.get("tags", [])),
            propagate=data.get("propagate", True),
        )

    def with_tag(self, tag: str) -> 'Event':
        """Fuegt Tag hinzu und gibt Event zurueck"""
        self.tags.add(tag)
        return self

    def has_tag(self, tag: str) -> bool:
        """Prueft ob Tag vorhanden"""
        return tag in self.tags


@dataclass
class EventFilter:
    """Filter fuer Events"""

    event_types: Optional[Set[EventType]] = None
    sources: Optional[Set[str]] = None
    tags: Optional[Set[str]] = None
    custom_filter: Optional[Callable[[Event], bool]] = None

    def matches(self, event: Event) -> bool:
        """Prueft ob Event dem Filter entspricht"""
        if self.event_types and event.type not in self.event_types:
            return False

        if self.sources and event.source not in self.sources:
            return False

        if self.tags and not self.tags.intersection(event.tags):
            return False

        if self.custom_filter and not self.custom_filter(event):
            return False

        return True

    @classmethod
    def for_types(cls, *types: EventType) -> 'EventFilter':
        """Erstellt Filter fuer bestimmte Eventtypen"""
        return cls(event_types=set(types))

    @classmethod
    def for_source(cls, source: str) -> 'EventFilter':
        """Erstellt Filter fuer bestimmte Quelle"""
        return cls(sources={source})

    @classmethod
    def for_tags(cls, *tags: str) -> 'EventFilter':
        """Erstellt Filter fuer bestimmte Tags"""
        return cls(tags=set(tags))


# Type alias fuer Event Handler
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]


class EventBus:
    """Zentraler Event-Bus fuer Pub/Sub"""

    def __init__(self):
        self._handlers: Dict[EventType, List[tuple[EventHandler, Optional[EventFilter]]]] = defaultdict(list)
        self._async_handlers: Dict[EventType, List[tuple[AsyncEventHandler, Optional[EventFilter]]]] = defaultdict(list)
        self._global_handlers: List[tuple[EventHandler, Optional[EventFilter]]] = []
        self._history: List[Event] = []
        self._history_size = 1000
        self._lock = threading.Lock()

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        filter: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """Abonniert Events eines Typs"""
        with self._lock:
            self._handlers[event_type].append((handler, filter))

        def unsubscribe():
            with self._lock:
                self._handlers[event_type].remove((handler, filter))

        return unsubscribe

    def subscribe_all(
        self,
        handler: EventHandler,
        filter: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """Abonniert alle Events"""
        with self._lock:
            self._global_handlers.append((handler, filter))

        def unsubscribe():
            with self._lock:
                self._global_handlers.remove((handler, filter))

        return unsubscribe

    def subscribe_async(
        self,
        event_type: EventType,
        handler: AsyncEventHandler,
        filter: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """Abonniert Events asynchron"""
        with self._lock:
            self._async_handlers[event_type].append((handler, filter))

        def unsubscribe():
            with self._lock:
                self._async_handlers[event_type].remove((handler, filter))

        return unsubscribe

    def publish(self, event: Event) -> None:
        """Veroeffentlicht ein Event"""
        # Zur Historie hinzufuegen
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history.pop(0)

        # Typ-spezifische Handler
        for handler, filter in self._handlers.get(event.type, []):
            if filter is None or filter.matches(event):
                try:
                    handler(event)
                except Exception as e:
                    self._handle_error(event, handler, e)

        # Globale Handler
        for handler, filter in self._global_handlers:
            if filter is None or filter.matches(event):
                try:
                    handler(event)
                except Exception as e:
                    self._handle_error(event, handler, e)

    async def publish_async(self, event: Event) -> None:
        """Veroeffentlicht ein Event asynchron"""
        # Zur Historie hinzufuegen
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history.pop(0)

        # Async Handler
        tasks = []
        for handler, filter in self._async_handlers.get(event.type, []):
            if filter is None or filter.matches(event):
                tasks.append(handler(event))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Sync Handler in Thread
        self.publish(event)

    def _handle_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Behandelt Fehler in Handlern"""
        error_event = Event(
            type=EventType.SYSTEM_ERROR,
            source="EventBus",
            data={
                "original_event_id": event.id,
                "handler": str(handler),
                "error": str(error),
            },
        )
        # Vermeide Rekursion bei Fehler-Events
        if event.type != EventType.SYSTEM_ERROR:
            self.publish(error_event)

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """Holt Event-Historie"""
        with self._lock:
            events = self._history.copy()

        if event_type:
            events = [e for e in events if e.type == event_type]

        return events[-limit:]

    def clear_history(self) -> None:
        """Loescht Historie"""
        with self._lock:
            self._history.clear()

    def get_subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """Zaehlt Abonnenten"""
        if event_type:
            return len(self._handlers.get(event_type, []))
        return sum(len(handlers) for handlers in self._handlers.values()) + len(self._global_handlers)


# Globale EventBus-Instanz
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Holt globalen EventBus"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def emit(event: Event) -> None:
    """Kurzform zum Veroeffentlichen"""
    get_event_bus().publish(event)


def on(event_type: EventType, filter: Optional[EventFilter] = None):
    """Decorator fuer Event-Handler"""
    def decorator(func: EventHandler) -> EventHandler:
        get_event_bus().subscribe(event_type, func, filter)
        return func
    return decorator


__all__ = [
    'Event',
    'EventType',
    'EventFilter',
    'EventHandler',
    'AsyncEventHandler',
    'EventBus',
    'get_event_bus',
    'emit',
    'on',
]
