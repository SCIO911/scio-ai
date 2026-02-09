"""
Message Protocol
================

Standardisiertes Nachrichtenformat fuer Agent-zu-Agent Kommunikation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum, auto
from datetime import datetime
import uuid
import json


class MessageType(Enum):
    """Nachrichtentypen"""
    REQUEST = auto()
    RESPONSE = auto()
    NOTIFICATION = auto()
    ERROR = auto()
    HEARTBEAT = auto()
    ACK = auto()
    BROADCAST = auto()


class MessagePriority(Enum):
    """Nachrichtenprioritaet"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Message:
    """Basis-Nachricht fuer alle Kommunikation"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.NOTIFICATION
    sender: str = ""
    receiver: str = ""
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "id": self.id,
            "type": self.type.name,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Erstellt Message aus Dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType[data.get("type", "NOTIFICATION")],
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            payload=data.get("payload"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            priority=MessagePriority(data.get("priority", 5)),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl=data.get("ttl"),
        )

    def to_json(self) -> str:
        """Serialisiert zu JSON"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Deserialisiert von JSON"""
        return cls.from_dict(json.loads(json_str))

    def is_expired(self) -> bool:
        """Prueft ob Nachricht abgelaufen ist"""
        if self.ttl is None:
            return False
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.ttl

    def create_reply(self, payload: Any = None) -> 'Message':
        """Erstellt Antwort-Nachricht"""
        return Message(
            type=MessageType.RESPONSE,
            sender=self.receiver,
            receiver=self.sender,
            payload=payload,
            correlation_id=self.id,
            priority=self.priority,
        )


@dataclass
class Request(Message):
    """Anfrage-Nachricht"""

    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None

    def __post_init__(self):
        self.type = MessageType.REQUEST

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "action": self.action,
            "parameters": self.parameters,
            "timeout": self.timeout,
        })
        return data


@dataclass
class Response(Message):
    """Antwort-Nachricht"""

    success: bool = True
    result: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None

    def __post_init__(self):
        self.type = MessageType.RESPONSE

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        })
        return data

    @classmethod
    def success_response(cls, request: Request, result: Any, duration_ms: float = None) -> 'Response':
        """Erstellt erfolgreiche Antwort"""
        return cls(
            sender=request.receiver,
            receiver=request.sender,
            correlation_id=request.id,
            success=True,
            result=result,
            duration_ms=duration_ms,
        )

    @classmethod
    def error_response(cls, request: Request, error: str) -> 'Response':
        """Erstellt Fehler-Antwort"""
        return cls(
            sender=request.receiver,
            receiver=request.sender,
            correlation_id=request.id,
            success=False,
            error=error,
        )


@dataclass
class Notification(Message):
    """Benachrichtigung (keine Antwort erwartet)"""

    topic: str = ""
    data: Any = None

    def __post_init__(self):
        self.type = MessageType.NOTIFICATION

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "topic": self.topic,
            "data": self.data,
        })
        return data


@dataclass
class ErrorMessage(Message):
    """Fehlermeldung"""

    error_code: str = ""
    error_message: str = ""
    error_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    recoverable: bool = True

    def __post_init__(self):
        self.type = MessageType.ERROR
        self.priority = MessagePriority.HIGH

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "stack_trace": self.stack_trace,
            "recoverable": self.recoverable,
        })
        return data


class MessageQueue:
    """Einfache Nachrichten-Warteschlange"""

    def __init__(self, max_size: int = 1000):
        self._queue: List[Message] = []
        self._max_size = max_size

    def push(self, message: Message) -> bool:
        """Fuegt Nachricht hinzu"""
        if len(self._queue) >= self._max_size:
            return False

        # Nach Prioritaet einsortieren
        inserted = False
        for i, msg in enumerate(self._queue):
            if message.priority.value > msg.priority.value:
                self._queue.insert(i, message)
                inserted = True
                break

        if not inserted:
            self._queue.append(message)

        return True

    def pop(self) -> Optional[Message]:
        """Holt naechste Nachricht"""
        if not self._queue:
            return None

        # Abgelaufene Nachrichten entfernen
        self._queue = [m for m in self._queue if not m.is_expired()]

        return self._queue.pop(0) if self._queue else None

    def peek(self) -> Optional[Message]:
        """Zeigt naechste Nachricht ohne sie zu entfernen"""
        return self._queue[0] if self._queue else None

    def size(self) -> int:
        """Anzahl Nachrichten"""
        return len(self._queue)

    def is_empty(self) -> bool:
        """Ist Warteschlange leer"""
        return len(self._queue) == 0

    def clear(self) -> None:
        """Leert Warteschlange"""
        self._queue.clear()


__all__ = [
    'Message',
    'MessageType',
    'MessagePriority',
    'Request',
    'Response',
    'Notification',
    'ErrorMessage',
    'MessageQueue',
]
