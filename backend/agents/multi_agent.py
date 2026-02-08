#!/usr/bin/env python3
"""
SCIO - Multi-Agent System
Kollaboration zwischen spezialisierten AI-Agenten
"""

import uuid
import time
import logging
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Typen von Agent-Nachrichten"""
    REQUEST = 'request'          # Anfrage an anderen Agenten
    RESPONSE = 'response'        # Antwort auf Anfrage
    BROADCAST = 'broadcast'      # Nachricht an alle
    DELEGATE = 'delegate'        # Aufgabe delegieren
    RESULT = 'result'           # Ergebnis einer Aufgabe
    STATUS = 'status'           # Status-Update
    SUBSCRIBE = 'subscribe'      # Für Events registrieren
    EVENT = 'event'             # Event-Benachrichtigung


class AgentStatus(str, Enum):
    """Status eines Agenten"""
    IDLE = 'idle'
    BUSY = 'busy'
    WAITING = 'waiting'
    ERROR = 'error'
    OFFLINE = 'offline'


@dataclass
class Message:
    """Eine Nachricht zwischen Agenten"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: str = ""  # Leer = Broadcast
    content: Dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None  # ID der Nachricht auf die geantwortet wird
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: int = 60  # Time-to-live in Sekunden


@dataclass
class Task:
    """Eine Aufgabe für einen Agenten"""
    id: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    assigned_to: Optional[str] = None
    created_by: str = ""
    priority: int = 0
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)


class Agent(ABC):
    """
    Basis-Klasse für einen AI-Agenten
    """

    def __init__(self, agent_id: str, name: str, capabilities: List[str] = None):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities or []
        self.status = AgentStatus.IDLE
        self.message_queue: Queue = Queue()
        self.pending_responses: Dict[str, Message] = {}
        self._pending_responses_lock = threading.Lock()
        self.subscriptions: Set[str] = set()
        self._subscriptions_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @abstractmethod
    def process_task(self, task: Task) -> Dict[str, Any]:
        """Verarbeitet eine Aufgabe"""
        pass

    def handle_message(self, message: Message) -> Optional[Message]:
        """Verarbeitet eine eingehende Nachricht"""
        if message.type == MessageType.REQUEST:
            # Anfrage verarbeiten
            result = self._handle_request(message)
            if result:
                return Message(
                    type=MessageType.RESPONSE,
                    sender=self.agent_id,
                    recipient=message.sender,
                    content=result,
                    reply_to=message.id
                )

        elif message.type == MessageType.DELEGATE:
            # Delegierte Aufgabe
            task = Task(
                id=message.content.get("task_id", str(uuid.uuid4())[:8]),
                description=message.content.get("description", ""),
                params=message.content.get("params", {}),
                created_by=message.sender
            )
            result = self.process_task(task)
            return Message(
                type=MessageType.RESULT,
                sender=self.agent_id,
                recipient=message.sender,
                content={"task_id": task.id, "result": result},
                reply_to=message.id
            )

        elif message.type == MessageType.RESPONSE:
            # Antwort auf unsere Anfrage
            with self._pending_responses_lock:
                if message.reply_to in self.pending_responses:
                    self.pending_responses[message.reply_to] = message

        elif message.type == MessageType.EVENT:
            # Event-Benachrichtigung
            self._handle_event(message)

        return None

    def _handle_request(self, message: Message) -> Optional[Dict]:
        """Verarbeitet eine Anfrage"""
        action = message.content.get("action")

        if action == "get_status":
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "status": self.status.value,
                "capabilities": self.capabilities
            }

        elif action == "get_capabilities":
            return {"capabilities": self.capabilities}

        elif action == "can_handle":
            task_type = message.content.get("task_type")
            return {"can_handle": task_type in self.capabilities}

        return None

    def _handle_event(self, message: Message):
        """Verarbeitet ein Event"""
        pass  # Kann überschrieben werden

    def start(self):
        """Startet den Agenten"""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stoppt den Agenten"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        """Hauptschleife des Agenten"""
        while self._running:
            try:
                message = self.message_queue.get(timeout=1)
                response = self.handle_message(message)
                if response:
                    # Response zurück an System senden
                    self._send_response(response)
            except Empty:
                continue
            except Exception as e:
                self.status = AgentStatus.ERROR
                logger.error(f"Agent {self.agent_id}: {e}")

    def _send_response(self, message: Message):
        """Sendet Antwort (wird vom MultiAgentSystem überschrieben)"""
        pass

    def receive(self, message: Message):
        """Empfängt eine Nachricht"""
        self.message_queue.put(message)


class SpecialistAgent(Agent):
    """Ein spezialisierter Agent für bestimmte Aufgaben"""

    def __init__(self, agent_id: str, name: str, worker_type: str):
        super().__init__(agent_id, name, [worker_type])
        self.worker_type = worker_type
        self.worker = None

    def set_worker(self, worker):
        """Setzt den zugehörigen Worker"""
        self.worker = worker

    def process_task(self, task: Task) -> Dict[str, Any]:
        """Verarbeitet eine Aufgabe mit dem Worker"""
        if not self.worker:
            return {"error": "No worker assigned"}

        self.status = AgentStatus.BUSY
        try:
            result = self.worker.process(task.id, task.params)
            self.status = AgentStatus.IDLE
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {"error": str(e)}


class CoordinatorAgent(Agent):
    """Koordinator-Agent der Aufgaben verteilt"""

    def __init__(self, agent_id: str = "coordinator"):
        super().__init__(agent_id, "Coordinator", ["coordinate", "delegate", "monitor"])
        self.agent_registry: Dict[str, Dict] = {}
        self.task_queue: List[Task] = []

    def register_agent(self, agent_id: str, capabilities: List[str]):
        """Registriert einen Agenten"""
        self.agent_registry[agent_id] = {
            "capabilities": capabilities,
            "status": AgentStatus.IDLE,
            "tasks_completed": 0
        }

    def process_task(self, task: Task) -> Dict[str, Any]:
        """Koordiniert die Aufgabenverteilung"""
        # Finde passenden Agenten
        best_agent = self._find_best_agent(task)

        if not best_agent:
            return {"error": "No suitable agent found"}

        # Delegiere Aufgabe
        task.assigned_to = best_agent
        return {
            "delegated_to": best_agent,
            "task_id": task.id
        }

    def _find_best_agent(self, task: Task) -> Optional[str]:
        """Findet den besten Agenten für eine Aufgabe"""
        task_type = task.params.get("type", "")

        candidates = []
        for agent_id, info in self.agent_registry.items():
            if task_type in info["capabilities"]:
                if info["status"] == AgentStatus.IDLE:
                    candidates.append((agent_id, info["tasks_completed"]))

        if not candidates:
            # Alle busy - nimm den mit wenigsten Tasks
            for agent_id, info in self.agent_registry.items():
                if task_type in info["capabilities"]:
                    candidates.append((agent_id, info["tasks_completed"]))

        if candidates:
            # Wähle Agent mit wenigsten abgeschlossenen Tasks (Load Balancing)
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None


class MultiAgentSystem:
    """
    SCIO Multi-Agent System

    Koordiniert mehrere spezialisierte AI-Agenten:
    - Registrierung und Discovery
    - Nachrichtenaustausch
    - Aufgabenverteilung
    - Kollaborative Problemlösung
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self._agents_lock = threading.Lock()
        self.message_bus: Queue = Queue()
        self.coordinator: Optional[CoordinatorAgent] = None
        self.event_subscribers: Dict[str, Set[str]] = {}  # event_type -> agent_ids
        self._subscribers_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert das Multi-Agent System"""
        try:
            # Coordinator erstellen
            self.coordinator = CoordinatorAgent()
            self.register_agent(self.coordinator)

            self._initialized = True
            logger.info("Multi-Agent System initialisiert")
            return True
        except Exception as e:
            logger.error(f"Multi-Agent System Fehler: {e}")
            return False

    def register_agent(self, agent: Agent):
        """Registriert einen Agenten"""
        self.agents[agent.agent_id] = agent

        # Bei Coordinator registrieren
        if self.coordinator and agent.agent_id != self.coordinator.agent_id:
            self.coordinator.register_agent(agent.agent_id, agent.capabilities)

        # Response-Callback setzen
        agent._send_response = lambda msg: self.send_message(msg)

    def unregister_agent(self, agent_id: str):
        """Entfernt einen Agenten"""
        if agent_id in self.agents:
            self.agents[agent_id].stop()
            del self.agents[agent_id]

    def send_message(self, message: Message):
        """Sendet eine Nachricht"""
        self.message_bus.put(message)

    def broadcast(self, content: Dict[str, Any], sender: str = "system"):
        """Sendet Broadcast an alle Agenten"""
        message = Message(
            type=MessageType.BROADCAST,
            sender=sender,
            content=content
        )
        for agent_id in self.agents:
            if agent_id != sender:
                msg_copy = Message(
                    id=message.id,
                    type=message.type,
                    sender=message.sender,
                    recipient=agent_id,
                    content=message.content.copy()
                )
                self.agents[agent_id].receive(msg_copy)

    def request(self,
                recipient: str,
                action: str,
                params: Dict = None,
                timeout: float = 30.0) -> Optional[Dict]:
        """
        Sendet synchrone Anfrage an Agenten

        Args:
            recipient: Ziel-Agent
            action: Gewünschte Aktion
            params: Parameter
            timeout: Timeout in Sekunden

        Returns:
            Antwort oder None bei Timeout
        """
        if recipient not in self.agents:
            return None

        message = Message(
            type=MessageType.REQUEST,
            sender="system",
            recipient=recipient,
            content={"action": action, **(params or {})}
        )

        # Pending Response registrieren
        response_event = threading.Event()
        response_holder = {"response": None}

        def on_response(msg: Message):
            if msg.reply_to == message.id:
                response_holder["response"] = msg.content
                response_event.set()

        # Temporär Response-Handler
        original_handler = self.agents[recipient].handle_message

        def wrapped_handler(msg):
            result = original_handler(msg)
            if result and result.reply_to == message.id:
                on_response(result)
            return result

        self.agents[recipient].receive(message)

        # Warten auf Antwort
        if response_event.wait(timeout):
            return response_holder["response"]

        return None

    def delegate_task(self,
                      description: str,
                      params: Dict[str, Any],
                      priority: int = 0) -> Task:
        """
        Delegiert eine Aufgabe an passenden Agenten

        Args:
            description: Aufgabenbeschreibung
            params: Parameter
            priority: Priorität

        Returns:
            Die erstellte Task
        """
        task = Task(
            id=str(uuid.uuid4())[:8],
            description=description,
            params=params,
            priority=priority,
            created_by="system"
        )

        if self.coordinator:
            result = self.coordinator.process_task(task)
            task.assigned_to = result.get("delegated_to")

            if task.assigned_to and task.assigned_to in self.agents:
                # Aufgabe an Agent senden
                message = Message(
                    type=MessageType.DELEGATE,
                    sender="coordinator",
                    recipient=task.assigned_to,
                    content={
                        "task_id": task.id,
                        "description": description,
                        "params": params
                    },
                    priority=priority
                )
                self.agents[task.assigned_to].receive(message)

        return task

    def subscribe(self, agent_id: str, event_type: str):
        """Registriert Agent für Event-Typ"""
        with self._subscribers_lock:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = set()
            self.event_subscribers[event_type].add(agent_id)

    def emit_event(self, event_type: str, data: Dict[str, Any], sender: str = "system"):
        """Emittiert ein Event an alle Subscriber"""
        with self._subscribers_lock:
            subscribers = self.event_subscribers.get(event_type, set()).copy()

        for agent_id in subscribers:
            if agent_id in self.agents:
                message = Message(
                    type=MessageType.EVENT,
                    sender=sender,
                    recipient=agent_id,
                    content={"event_type": event_type, "data": data}
                )
                self.agents[agent_id].receive(message)

    def start(self):
        """Startet das System"""
        self._running = True

        # Alle Agenten starten
        for agent in self.agents.values():
            agent.start()

        # Message Router starten
        self._thread = threading.Thread(target=self._message_router, daemon=True)
        self._thread.start()

    def stop(self):
        """Stoppt das System"""
        self._running = False

        for agent in self.agents.values():
            agent.stop()

        if self._thread:
            self._thread.join(timeout=5)

    def _message_router(self):
        """Verteilt Nachrichten an Agenten"""
        while self._running:
            try:
                message = self.message_bus.get(timeout=1)

                if message.recipient:
                    # Direkte Nachricht
                    if message.recipient in self.agents:
                        self.agents[message.recipient].receive(message)
                else:
                    # Broadcast
                    for agent_id, agent in self.agents.items():
                        if agent_id != message.sender:
                            agent.receive(message)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Message Router: {e}")

    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Gibt Status eines Agenten zurück"""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "status": agent.status.value,
            "capabilities": agent.capabilities,
            "queue_size": agent.message_queue.qsize()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = {
                "status": agent.status.value,
                "capabilities": agent.capabilities,
                "queue_size": agent.message_queue.qsize()
            }

        return {
            "agents_count": len(self.agents),
            "running": self._running,
            "message_bus_size": self.message_bus.qsize(),
            "event_subscriptions": {k: len(v) for k, v in self.event_subscribers.items()},
            "agents": agent_stats
        }


# Singleton
_multi_agent_system: Optional[MultiAgentSystem] = None

def get_multi_agent_system() -> MultiAgentSystem:
    """Gibt Singleton-Instanz zurück"""
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system
