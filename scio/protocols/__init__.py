"""
SCIO Protocols
==============

Standardisierte Protokoll-Definitionen fuer Kommunikation,
Datenaustausch und Interoperabilitaet zwischen SCIO-Komponenten.

Protokolle:
- MessageProtocol: Nachrichtenformat zwischen Agents
- DataProtocol: Datenaustausch-Formate
- EventProtocol: Event-basierte Kommunikation
- ToolProtocol: Tool-Schnittstellen
- ExecutionProtocol: Ausfuehrungsprotokolle
"""

from scio.protocols.messages import (
    Message,
    MessageType,
    MessagePriority,
    Request,
    Response,
    Notification,
    ErrorMessage,
)

from scio.protocols.data import (
    DataPacket,
    DataFormat,
    DataSchema,
    serialize,
    deserialize,
)

from scio.protocols.events import (
    Event,
    EventType,
    EventBus,
    EventHandler,
    EventFilter,
)

from scio.protocols.tools import (
    ToolInterface,
    ToolInput,
    ToolOutput,
    ToolMetadata,
    ToolResult,
)

from scio.protocols.execution import (
    ExecutionContext,
    ExecutionState,
    ExecutionResult,
    Checkpoint,
    Progress,
)

__all__ = [
    # Messages
    'Message',
    'MessageType',
    'MessagePriority',
    'Request',
    'Response',
    'Notification',
    'ErrorMessage',
    # Data
    'DataPacket',
    'DataFormat',
    'DataSchema',
    'serialize',
    'deserialize',
    # Events
    'Event',
    'EventType',
    'EventBus',
    'EventHandler',
    'EventFilter',
    # Tools
    'ToolInterface',
    'ToolInput',
    'ToolOutput',
    'ToolMetadata',
    'ToolResult',
    # Execution
    'ExecutionContext',
    'ExecutionState',
    'ExecutionResult',
    'Checkpoint',
    'Progress',
]

__version__ = '1.0.0'
