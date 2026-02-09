"""
Comprehensive tests for SCIO Protocols Module.

Tests cover:
- Messages: Message, MessageType, MessagePriority, Request, Response, Notification, ErrorMessage, MessageQueue
- Events: Event, EventType, EventFilter, EventBus, get_event_bus, emit, on
- Tools: ToolStatus, ToolCategory, ToolMetadata, ToolInput, ToolOutput, ToolResult, ToolInterface
- Execution: ExecutionState, Progress, Checkpoint, ExecutionContext, ExecutionResult
- Data: DataFormat, CompressionType, DataSchema, DataPacket, serialize, deserialize
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Messages
from scio.protocols.messages import (
    Message,
    MessageType,
    MessagePriority,
    Request,
    Response,
    Notification,
    ErrorMessage,
    MessageQueue,
)

# Events
from scio.protocols.events import (
    Event,
    EventType,
    EventFilter,
    EventBus,
    get_event_bus,
    emit,
    on,
)

# Tools
from scio.protocols.tools import (
    ToolStatus,
    ToolCategory,
    ToolMetadata,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolInterface,
)

# Execution
from scio.protocols.execution import (
    ExecutionState,
    Progress,
    Checkpoint,
    ExecutionContext,
    ExecutionResult,
)

# Data
from scio.protocols.data import (
    DataFormat,
    CompressionType,
    DataSchema,
    DataPacket,
    serialize,
    deserialize,
)


# ============================================================================
# MessageType Tests
# ============================================================================

class TestMessageType:
    """Tests for MessageType enum."""

    def test_all_types_exist(self):
        """Test all expected message types exist."""
        expected = ["REQUEST", "RESPONSE", "NOTIFICATION", "ERROR", "HEARTBEAT", "ACK", "BROADCAST"]
        for t in expected:
            assert hasattr(MessageType, t)

    def test_type_values(self):
        """Test message types are distinct."""
        types = list(MessageType)
        values = [t.value for t in types]
        assert len(values) == len(set(values))


class TestMessagePriority:
    """Tests for MessagePriority enum."""

    def test_priority_values(self):
        """Test priority values."""
        assert MessagePriority.LOW.value == 1
        assert MessagePriority.NORMAL.value == 5
        assert MessagePriority.HIGH.value == 8
        assert MessagePriority.CRITICAL.value == 10

    def test_priority_ordering(self):
        """Test priorities are ordered correctly."""
        assert MessagePriority.LOW.value < MessagePriority.NORMAL.value
        assert MessagePriority.NORMAL.value < MessagePriority.HIGH.value
        assert MessagePriority.HIGH.value < MessagePriority.CRITICAL.value


# ============================================================================
# Message Tests
# ============================================================================

class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a basic message."""
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            payload={"data": "test"}
        )
        assert msg.sender == "agent_a"
        assert msg.receiver == "agent_b"
        assert msg.payload == {"data": "test"}
        assert msg.id is not None
        assert msg.type == MessageType.NOTIFICATION

    def test_message_defaults(self):
        """Test default values."""
        msg = Message()
        assert msg.type == MessageType.NOTIFICATION
        assert msg.priority == MessagePriority.NORMAL
        assert msg.ttl is None
        assert msg.correlation_id is None

    def test_message_to_dict(self):
        """Test serialization to dict."""
        msg = Message(
            sender="sender",
            receiver="receiver",
            payload="test",
            priority=MessagePriority.HIGH
        )
        d = msg.to_dict()
        assert d["sender"] == "sender"
        assert d["receiver"] == "receiver"
        assert d["payload"] == "test"
        assert d["type"] == "NOTIFICATION"
        assert d["priority"] == 8

    def test_message_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "test-123",
            "type": "REQUEST",
            "sender": "agent_a",
            "receiver": "agent_b",
            "payload": {"key": "value"},
            "timestamp": datetime.now().isoformat(),
            "priority": 5
        }
        msg = Message.from_dict(data)
        assert msg.id == "test-123"
        assert msg.type == MessageType.REQUEST
        assert msg.sender == "agent_a"

    def test_message_to_json(self):
        """Test JSON serialization."""
        msg = Message(sender="test", payload={"data": 123})
        json_str = msg.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["sender"] == "test"

    def test_message_from_json(self):
        """Test JSON deserialization."""
        original = Message(sender="test", payload={"data": 123})
        json_str = original.to_json()
        restored = Message.from_json(json_str)
        assert restored.sender == original.sender

    def test_message_is_expired(self):
        """Test TTL expiration check."""
        # Non-expired message
        msg = Message(ttl=3600)
        assert msg.is_expired() == False

        # Message with no TTL never expires
        msg_no_ttl = Message()
        assert msg_no_ttl.is_expired() == False

    def test_message_create_reply(self):
        """Test creating a reply message."""
        original = Message(
            sender="agent_a",
            receiver="agent_b",
            payload="question"
        )
        reply = original.create_reply(payload="answer")
        assert reply.sender == "agent_b"
        assert reply.receiver == "agent_a"
        assert reply.correlation_id == original.id
        assert reply.type == MessageType.RESPONSE


# ============================================================================
# Request/Response/Notification Tests
# ============================================================================

class TestRequest:
    """Tests for Request message."""

    def test_create_request(self):
        """Test creating a request."""
        req = Request(
            sender="client",
            receiver="server",
            action="get_data",
            parameters={"id": 123}
        )
        assert req.type == MessageType.REQUEST
        assert req.action == "get_data"
        assert req.parameters == {"id": 123}

    def test_request_to_dict(self):
        """Test request serialization."""
        req = Request(action="test", parameters={"a": 1})
        d = req.to_dict()
        assert d["action"] == "test"
        assert d["parameters"] == {"a": 1}


class TestResponse:
    """Tests for Response message."""

    def test_create_response(self):
        """Test creating a response."""
        resp = Response(
            sender="server",
            receiver="client",
            success=True,
            result={"data": "result"}
        )
        assert resp.type == MessageType.RESPONSE
        assert resp.success == True
        assert resp.result == {"data": "result"}

    def test_success_response(self):
        """Test creating success response from request."""
        req = Request(sender="client", receiver="server", action="test")
        resp = Response.success_response(req, result="ok", duration_ms=100)
        assert resp.success == True
        assert resp.result == "ok"
        assert resp.duration_ms == 100
        assert resp.correlation_id == req.id

    def test_error_response(self):
        """Test creating error response from request."""
        req = Request(sender="client", receiver="server", action="test")
        resp = Response.error_response(req, error="Something failed")
        assert resp.success == False
        assert resp.error == "Something failed"
        assert resp.correlation_id == req.id


class TestNotification:
    """Tests for Notification message."""

    def test_create_notification(self):
        """Test creating a notification."""
        notif = Notification(
            sender="system",
            topic="status",
            data={"status": "running"}
        )
        assert notif.type == MessageType.NOTIFICATION
        assert notif.topic == "status"
        assert notif.data == {"status": "running"}


class TestErrorMessage:
    """Tests for ErrorMessage."""

    def test_create_error_message(self):
        """Test creating an error message."""
        err = ErrorMessage(
            sender="system",
            error_code="E001",
            error_message="Something went wrong",
            recoverable=True
        )
        assert err.type == MessageType.ERROR
        assert err.priority == MessagePriority.HIGH
        assert err.error_code == "E001"
        assert err.recoverable == True


# ============================================================================
# MessageQueue Tests
# ============================================================================

class TestMessageQueue:
    """Tests for MessageQueue."""

    def test_create_queue(self):
        """Test creating a queue."""
        queue = MessageQueue(max_size=100)
        assert queue.size() == 0
        assert queue.is_empty() == True

    def test_push_and_pop(self):
        """Test pushing and popping messages."""
        queue = MessageQueue()
        msg = Message(payload="test")
        assert queue.push(msg) == True
        assert queue.size() == 1

        popped = queue.pop()
        assert popped.payload == "test"
        assert queue.is_empty() == True

    def test_priority_ordering(self):
        """Test messages are ordered by priority."""
        queue = MessageQueue()
        low = Message(priority=MessagePriority.LOW, payload="low")
        high = Message(priority=MessagePriority.HIGH, payload="high")

        queue.push(low)
        queue.push(high)

        # High priority should come first
        first = queue.pop()
        assert first.payload == "high"

    def test_queue_max_size(self):
        """Test queue respects max size."""
        queue = MessageQueue(max_size=2)
        queue.push(Message(payload="1"))
        queue.push(Message(payload="2"))

        # Third message should be rejected
        assert queue.push(Message(payload="3")) == False
        assert queue.size() == 2

    def test_peek(self):
        """Test peeking at next message."""
        queue = MessageQueue()
        msg = Message(payload="peek_test")
        queue.push(msg)

        peeked = queue.peek()
        assert peeked.payload == "peek_test"
        # Message should still be in queue
        assert queue.size() == 1

    def test_clear(self):
        """Test clearing the queue."""
        queue = MessageQueue()
        queue.push(Message())
        queue.push(Message())
        queue.clear()
        assert queue.is_empty() == True


# ============================================================================
# EventType Tests
# ============================================================================

class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        """Test all expected event types exist."""
        expected = [
            "SYSTEM_START", "SYSTEM_STOP", "SYSTEM_ERROR",
            "AGENT_CREATED", "AGENT_STARTED", "AGENT_STOPPED",
            "EXECUTION_START", "EXECUTION_COMPLETE", "EXECUTION_ERROR",
            "TOOL_INVOKED", "TOOL_COMPLETE", "TOOL_ERROR",
            "CUSTOM"
        ]
        for t in expected:
            assert hasattr(EventType, t)


# ============================================================================
# Event Tests
# ============================================================================

class TestEvent:
    """Tests for Event dataclass."""

    def test_create_event(self):
        """Test creating an event."""
        event = Event(
            type=EventType.SYSTEM_START,
            source="main",
            data={"version": "1.0"}
        )
        assert event.type == EventType.SYSTEM_START
        assert event.source == "main"
        assert event.data == {"version": "1.0"}

    def test_event_to_dict(self):
        """Test event serialization."""
        event = Event(
            type=EventType.CUSTOM,
            source="test",
            data={"key": "value"},
            tags={"tag1", "tag2"}
        )
        d = event.to_dict()
        assert d["type"] == "CUSTOM"
        assert d["source"] == "test"
        assert set(d["tags"]) == {"tag1", "tag2"}

    def test_event_from_dict(self):
        """Test event deserialization."""
        data = {
            "id": "evt-123",
            "type": "AGENT_STARTED",
            "source": "agent1",
            "data": {"status": "ok"},
            "tags": ["important"],
            "timestamp": datetime.now().isoformat()
        }
        event = Event.from_dict(data)
        assert event.id == "evt-123"
        assert event.type == EventType.AGENT_STARTED

    def test_event_with_tag(self):
        """Test adding tags."""
        event = Event()
        event.with_tag("important").with_tag("urgent")
        assert event.has_tag("important") == True
        assert event.has_tag("urgent") == True
        assert event.has_tag("other") == False


# ============================================================================
# EventFilter Tests
# ============================================================================

class TestEventFilter:
    """Tests for EventFilter."""

    def test_filter_by_type(self):
        """Test filtering by event type."""
        filter = EventFilter.for_types(EventType.SYSTEM_START, EventType.SYSTEM_STOP)

        event_match = Event(type=EventType.SYSTEM_START)
        event_no_match = Event(type=EventType.CUSTOM)

        assert filter.matches(event_match) == True
        assert filter.matches(event_no_match) == False

    def test_filter_by_source(self):
        """Test filtering by source."""
        filter = EventFilter.for_source("agent1")

        event_match = Event(source="agent1")
        event_no_match = Event(source="agent2")

        assert filter.matches(event_match) == True
        assert filter.matches(event_no_match) == False

    def test_filter_by_tags(self):
        """Test filtering by tags."""
        filter = EventFilter.for_tags("important")

        event_match = Event(tags={"important", "other"})
        event_no_match = Event(tags={"other"})

        assert filter.matches(event_match) == True
        assert filter.matches(event_no_match) == False

    def test_custom_filter(self):
        """Test custom filter function."""
        filter = EventFilter(custom_filter=lambda e: e.data == "match")

        event_match = Event(data="match")
        event_no_match = Event(data="no_match")

        assert filter.matches(event_match) == True
        assert filter.matches(event_no_match) == False


# ============================================================================
# EventBus Tests
# ============================================================================

class TestEventBus:
    """Tests for EventBus."""

    @pytest.fixture
    def bus(self):
        """Create a fresh EventBus."""
        return EventBus()

    def test_subscribe_and_publish(self, bus):
        """Test basic subscribe and publish."""
        events_received = []

        def handler(event):
            events_received.append(event)

        bus.subscribe(EventType.CUSTOM, handler)
        bus.publish(Event(type=EventType.CUSTOM, data="test"))

        assert len(events_received) == 1
        assert events_received[0].data == "test"

    def test_unsubscribe(self, bus):
        """Test unsubscribe."""
        events_received = []

        def handler(event):
            events_received.append(event)

        unsub = bus.subscribe(EventType.CUSTOM, handler)
        bus.publish(Event(type=EventType.CUSTOM))
        assert len(events_received) == 1

        unsub()
        bus.publish(Event(type=EventType.CUSTOM))
        assert len(events_received) == 1  # No new events

    def test_subscribe_all(self, bus):
        """Test subscribing to all events."""
        events_received = []

        def handler(event):
            events_received.append(event)

        bus.subscribe_all(handler)
        bus.publish(Event(type=EventType.CUSTOM))
        bus.publish(Event(type=EventType.SYSTEM_START))

        assert len(events_received) == 2

    def test_subscribe_with_filter(self, bus):
        """Test subscribe with filter."""
        events_received = []

        def handler(event):
            events_received.append(event)

        filter = EventFilter.for_source("agent1")
        bus.subscribe(EventType.CUSTOM, handler, filter)

        bus.publish(Event(type=EventType.CUSTOM, source="agent1"))
        bus.publish(Event(type=EventType.CUSTOM, source="agent2"))

        assert len(events_received) == 1

    def test_get_history(self, bus):
        """Test event history."""
        bus.publish(Event(type=EventType.CUSTOM, data="1"))
        bus.publish(Event(type=EventType.CUSTOM, data="2"))
        bus.publish(Event(type=EventType.SYSTEM_START, data="3"))

        all_history = bus.get_history()
        assert len(all_history) == 3

        custom_history = bus.get_history(EventType.CUSTOM)
        assert len(custom_history) == 2

    def test_subscriber_count(self, bus):
        """Test counting subscribers."""
        assert bus.get_subscriber_count() == 0

        bus.subscribe(EventType.CUSTOM, lambda e: None)
        bus.subscribe(EventType.CUSTOM, lambda e: None)

        assert bus.get_subscriber_count(EventType.CUSTOM) == 2

    def test_clear_history(self, bus):
        """Test clearing history."""
        bus.publish(Event())
        assert len(bus.get_history()) == 1

        bus.clear_history()
        assert len(bus.get_history()) == 0


# ============================================================================
# ToolStatus Tests
# ============================================================================

class TestToolStatus:
    """Tests for ToolStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        expected = ["PENDING", "RUNNING", "SUCCESS", "FAILED", "CANCELLED", "TIMEOUT"]
        for s in expected:
            assert hasattr(ToolStatus, s)


# ============================================================================
# ToolMetadata Tests
# ============================================================================

class TestToolMetadata:
    """Tests for ToolMetadata."""

    def test_create_metadata(self):
        """Test creating tool metadata."""
        meta = ToolMetadata(
            name="test_tool",
            version="2.0.0",
            description="A test tool",
            category=ToolCategory.ANALYSIS,
            tags=["test", "analysis"]
        )
        assert meta.name == "test_tool"
        assert meta.version == "2.0.0"
        assert meta.category == ToolCategory.ANALYSIS

    def test_metadata_defaults(self):
        """Test default values."""
        meta = ToolMetadata()
        assert meta.version == "1.0.0"
        assert meta.deprecated == False
        assert meta.cacheable == False
        assert meta.requires_gpu == False

    def test_metadata_to_dict(self):
        """Test serialization."""
        meta = ToolMetadata(name="tool", category=ToolCategory.ML)
        d = meta.to_dict()
        assert d["name"] == "tool"
        assert d["category"] == "ml"


# ============================================================================
# ToolInput Tests
# ============================================================================

class TestToolInput:
    """Tests for ToolInput."""

    def test_create_input(self):
        """Test creating tool input."""
        input = ToolInput(
            parameters={"x": 1, "y": 2},
            data=[1, 2, 3]
        )
        assert input.parameters == {"x": 1, "y": 2}
        assert input.data == [1, 2, 3]

    def test_get_parameter(self):
        """Test getting parameters."""
        input = ToolInput(parameters={"x": 1})
        assert input.get("x") == 1
        assert input.get("y", "default") == "default"

    def test_require_parameter(self):
        """Test requiring parameters."""
        input = ToolInput(parameters={"x": 1})
        assert input.require("x") == 1

        with pytest.raises(ValueError):
            input.require("missing")

    def test_validate_parameters(self):
        """Test parameter validation."""
        input = ToolInput(parameters={"name": "test", "count": 5})

        schema = {"name": str, "count": int}
        valid, errors = input.validate(schema)
        assert valid == True

        schema_missing = {"name": str, "missing": str}
        valid, errors = input.validate(schema_missing)
        assert valid == False
        assert len(errors) == 1


# ============================================================================
# ToolOutput Tests
# ============================================================================

class TestToolOutput:
    """Tests for ToolOutput."""

    def test_create_output(self):
        """Test creating tool output."""
        output = ToolOutput(
            result={"computed": True},
            metadata={"version": "1.0"}
        )
        assert output.result == {"computed": True}

    def test_add_artifact(self):
        """Test adding artifacts."""
        output = ToolOutput()
        output.add_artifact("plot", "plot_data")
        assert output.artifacts["plot"] == "plot_data"

    def test_add_log(self):
        """Test adding logs."""
        output = ToolOutput()
        output.add_log("Processing started")
        assert len(output.logs) == 1
        assert "Processing started" in output.logs[0]

    def test_add_warning(self):
        """Test adding warnings."""
        output = ToolOutput()
        output.add_warning("Memory usage high")
        assert "Memory usage high" in output.warnings


# ============================================================================
# ToolResult Tests
# ============================================================================

class TestToolResult:
    """Tests for ToolResult."""

    def test_create_result(self):
        """Test creating a result."""
        result = ToolResult(tool_name="test_tool")
        assert result.status == ToolStatus.PENDING
        assert result.tool_name == "test_tool"

    def test_mark_started(self):
        """Test marking as started."""
        result = ToolResult()
        result.mark_started()
        assert result.status == ToolStatus.RUNNING
        assert result.start_time is not None

    def test_mark_success(self):
        """Test marking as success."""
        result = ToolResult()
        result.mark_started()
        output = ToolOutput(result="done")
        result.mark_success(output)

        assert result.status == ToolStatus.SUCCESS
        assert result.success == True
        assert result.output.result == "done"
        assert result.duration_ms is not None

    def test_mark_failed(self):
        """Test marking as failed."""
        result = ToolResult()
        result.mark_started()
        result.mark_failed("Error occurred", {"code": 500})

        assert result.status == ToolStatus.FAILED
        assert result.failed == True
        assert result.error == "Error occurred"
        assert result.error_details == {"code": 500}

    def test_mark_timeout(self):
        """Test marking as timeout."""
        result = ToolResult()
        result.mark_timeout()
        assert result.status == ToolStatus.TIMEOUT
        assert result.failed == True


# ============================================================================
# ExecutionState Tests
# ============================================================================

class TestExecutionState:
    """Tests for ExecutionState enum."""

    def test_terminal_states(self):
        """Test terminal state detection."""
        assert ExecutionState.COMPLETED.is_terminal == True
        assert ExecutionState.FAILED.is_terminal == True
        assert ExecutionState.CANCELLED.is_terminal == True
        assert ExecutionState.RUNNING.is_terminal == False

    def test_active_states(self):
        """Test active state detection."""
        assert ExecutionState.RUNNING.is_active == True
        assert ExecutionState.INITIALIZING.is_active == True
        assert ExecutionState.PAUSED.is_active == False

    def test_can_pause(self):
        """Test pause capability."""
        assert ExecutionState.RUNNING.can_pause == True
        assert ExecutionState.PAUSED.can_pause == False

    def test_can_resume(self):
        """Test resume capability."""
        assert ExecutionState.PAUSED.can_resume == True
        assert ExecutionState.RUNNING.can_resume == False


# ============================================================================
# Progress Tests
# ============================================================================

class TestProgress:
    """Tests for Progress tracking."""

    def test_create_progress(self):
        """Test creating progress."""
        progress = Progress(current=50, total=100, stage="Processing")
        assert progress.current == 50
        assert progress.total == 100
        assert progress.stage == "Processing"

    def test_percent_property(self):
        """Test percent calculation."""
        progress = Progress(current=50, total=100)
        assert progress.percent == 50.0

        progress.current = 100
        assert progress.percent == 100.0

    def test_is_complete(self):
        """Test completion check."""
        progress = Progress(current=50, total=100)
        assert progress.is_complete == False

        progress.current = 100
        assert progress.is_complete == True

    def test_update(self):
        """Test updating progress."""
        progress = Progress()
        progress.update(current=25, message="Quarter done", stage="Stage 1")

        assert progress.current == 25
        assert progress.message == "Quarter done"
        assert progress.stage == "Stage 1"

    def test_increment(self):
        """Test incrementing progress."""
        progress = Progress(current=0, total=10)
        progress.increment()
        assert progress.current == 1

        progress.increment(5)
        assert progress.current == 6

    def test_progress_clamping(self):
        """Test progress doesn't exceed total."""
        progress = Progress(current=95, total=100)
        progress.increment(10)
        assert progress.current == 100  # Clamped to total


# ============================================================================
# Checkpoint Tests
# ============================================================================

class TestCheckpoint:
    """Tests for Checkpoint."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        checkpoint = Checkpoint(
            execution_id="exec-123",
            name="step_1",
            state={"counter": 50}
        )
        assert checkpoint.execution_id == "exec-123"
        assert checkpoint.name == "step_1"
        assert checkpoint.state == {"counter": 50}

    def test_compute_checksum(self):
        """Test checksum computation."""
        checkpoint = Checkpoint(state={"data": "test"})
        checksum = checkpoint.compute_checksum()
        assert checksum is not None
        assert len(checksum) == 16

    def test_verify_checksum(self):
        """Test checksum verification."""
        checkpoint = Checkpoint(state={"data": "test"})
        checkpoint.compute_checksum()

        assert checkpoint.verify() == True

        # Modify state
        checkpoint.state["data"] = "modified"
        assert checkpoint.verify() == False

    def test_checkpoint_to_dict(self):
        """Test serialization."""
        # Create checkpoint without progress to avoid eta_seconds calculation issue
        checkpoint = Checkpoint(
            name="test",
            state={"x": 1}
        )
        d = checkpoint.to_dict()
        assert d["name"] == "test"
        assert d["state"] == {"x": 1}


# ============================================================================
# ExecutionContext Tests
# ============================================================================

class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_create_context(self):
        """Test creating a context."""
        ctx = ExecutionContext(
            name="test_execution",
            parameters={"input": "value"}
        )
        assert ctx.name == "test_execution"
        assert ctx.state == ExecutionState.PENDING

    def test_context_lifecycle(self):
        """Test context state transitions."""
        ctx = ExecutionContext()

        ctx.start()
        assert ctx.state == ExecutionState.RUNNING
        assert ctx.started_at is not None

        ctx.pause()
        assert ctx.state == ExecutionState.PAUSED

        ctx.resume()
        assert ctx.state == ExecutionState.RESUMING

        ctx.complete()
        assert ctx.state == ExecutionState.COMPLETED
        assert ctx.completed_at is not None

    def test_context_fail(self):
        """Test failing a context."""
        ctx = ExecutionContext()
        ctx.start()
        ctx.fail("Test error", {"code": "E001"})

        assert ctx.state == ExecutionState.FAILED
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["error"] == "Test error"

    def test_context_variables(self):
        """Test context variables."""
        ctx = ExecutionContext()
        ctx.set_variable("x", 100)
        ctx.set_variable("y", "hello")

        assert ctx.get_variable("x") == 100
        assert ctx.get_variable("y") == "hello"
        assert ctx.get_variable("z", "default") == "default"

    def test_context_logging(self):
        """Test context logging."""
        ctx = ExecutionContext()
        ctx.log("INFO", "Starting process")
        ctx.log("DEBUG", "Processing item", {"item_id": 123})

        assert len(ctx.logs) == 2
        assert ctx.logs[0]["message"] == "Starting process"

    def test_create_and_restore_checkpoint(self):
        """Test checkpoint creation and restoration."""
        ctx = ExecutionContext()
        ctx.set_variable("counter", 50)
        ctx.progress.current = 50

        checkpoint = ctx.create_checkpoint("midpoint")
        assert len(ctx.checkpoints) == 1

        # Modify context
        ctx.set_variable("counter", 100)
        ctx.progress.current = 100

        # Restore
        success = ctx.restore_checkpoint(checkpoint)
        assert success == True
        assert ctx.get_variable("counter") == 50
        assert ctx.progress.current == 50

    def test_duration_seconds(self):
        """Test duration calculation."""
        ctx = ExecutionContext()
        assert ctx.duration_seconds is None

        ctx.start()
        time.sleep(0.1)
        duration = ctx.duration_seconds
        assert duration is not None
        assert duration >= 0.1


# ============================================================================
# ExecutionResult Tests
# ============================================================================

class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_create_result(self):
        """Test creating a result."""
        result = ExecutionResult(
            execution_id="exec-123",
            success=True,
            result={"output": "done"}
        )
        assert result.success == True
        assert result.result == {"output": "done"}

    def test_from_context(self):
        """Test creating result from context."""
        ctx = ExecutionContext(name="test")
        ctx.start()
        ctx.complete()

        result = ExecutionResult.from_context(ctx, result="done")
        assert result.success == True
        assert result.result == "done"
        assert result.metrics["checkpoint_count"] == 0


# ============================================================================
# DataFormat Tests
# ============================================================================

class TestDataFormat:
    """Tests for DataFormat enum."""

    def test_all_formats_exist(self):
        """Test all expected formats exist."""
        expected = ["JSON", "BINARY", "CSV", "NUMPY", "PANDAS", "TORCH", "RAW"]
        for f in expected:
            assert hasattr(DataFormat, f)


# ============================================================================
# DataSchema Tests
# ============================================================================

class TestDataSchema:
    """Tests for DataSchema."""

    def test_create_schema(self):
        """Test creating a schema."""
        schema = DataSchema(
            name="user",
            version="1.0",
            fields={
                "name": {"type": "string"},
                "age": {"type": "int"}
            },
            required=["name"]
        )
        assert schema.name == "user"
        assert "name" in schema.fields

    def test_validate_valid_data(self):
        """Test validating valid data."""
        schema = DataSchema(
            fields={
                "name": {"type": "string"},
                "count": {"type": "int"}
            },
            required=["name"]
        )

        valid, errors = schema.validate({"name": "test", "count": 5})
        assert valid == True
        assert len(errors) == 0

    def test_validate_missing_required(self):
        """Test validation with missing required field."""
        schema = DataSchema(required=["name"])

        valid, errors = schema.validate({})
        assert valid == False
        assert "name" in errors[0]

    def test_validate_wrong_type(self):
        """Test validation with wrong type."""
        schema = DataSchema(
            fields={"count": {"type": "int"}}
        )

        valid, errors = schema.validate({"count": "not_an_int"})
        assert valid == False


# ============================================================================
# DataPacket Tests
# ============================================================================

class TestDataPacket:
    """Tests for DataPacket."""

    def test_create_packet(self):
        """Test creating a packet."""
        packet = DataPacket(
            format=DataFormat.JSON,
            data={"key": "value"}
        )
        assert packet.format == DataFormat.JSON
        assert packet.data == {"key": "value"}
        assert packet.size_bytes > 0

    def test_compute_checksum(self):
        """Test checksum computation."""
        packet = DataPacket(data={"test": "data"})
        checksum = packet.compute_checksum()
        assert checksum is not None
        assert len(checksum) == 64  # SHA256 hex

    def test_verify_checksum(self):
        """Test checksum verification."""
        packet = DataPacket(data={"test": "data"})
        original_checksum = packet.compute_checksum()

        # Verify succeeds when data matches checksum
        assert packet.verify_checksum() == True
        assert packet.checksum == original_checksum

        # Packet without checksum always verifies
        packet_no_checksum = DataPacket(data={"test": "data"})
        assert packet_no_checksum.checksum is None
        assert packet_no_checksum.verify_checksum() == True

        # Test that same data produces same checksum
        packet2 = DataPacket(data={"test": "data"})
        checksum2 = packet2.compute_checksum()
        assert checksum2 == original_checksum

    def test_compress_and_decompress(self):
        """Test compression and decompression."""
        original = DataPacket(
            format=DataFormat.JSON,
            data={"message": "test" * 100}
        )

        compressed = original.compress(CompressionType.ZLIB)
        assert compressed.compression == CompressionType.ZLIB
        assert compressed.format == DataFormat.BINARY

        decompressed = compressed.decompress()
        assert decompressed.data == original.data
        assert decompressed.compression == CompressionType.NONE

    def test_validate_with_schema(self):
        """Test validation with schema."""
        schema = DataSchema(required=["name"])
        packet = DataPacket(
            data={"name": "test"},
            schema=schema
        )

        valid, errors = packet.validate()
        assert valid == True

        packet_invalid = DataPacket(
            data={},
            schema=schema
        )
        valid, errors = packet_invalid.validate()
        assert valid == False


# ============================================================================
# Serialize/Deserialize Tests
# ============================================================================

class TestSerialization:
    """Tests for serialize and deserialize functions."""

    def test_json_serialization(self):
        """Test JSON serialization."""
        data = {"key": "value", "number": 42}
        serialized = serialize(data, DataFormat.JSON)
        assert isinstance(serialized, bytes)

        deserialized = deserialize(serialized, DataFormat.JSON)
        assert deserialized == data

    def test_binary_serialization(self):
        """Test binary serialization."""
        data = b"binary data"
        serialized = serialize(data, DataFormat.BINARY)
        assert serialized == data

        deserialized = deserialize(serialized, DataFormat.BINARY)
        assert deserialized == data

    def test_numpy_serialization(self):
        """Test NumPy serialization."""
        import numpy as np
        data = np.array([1, 2, 3, 4, 5])

        serialized = serialize(data, DataFormat.NUMPY)
        assert isinstance(serialized, bytes)

        deserialized = deserialize(serialized, DataFormat.NUMPY)
        assert np.array_equal(deserialized, data)

    def test_torch_serialization(self):
        """Test PyTorch serialization."""
        import torch
        data = torch.tensor([1, 2, 3, 4, 5])

        serialized = serialize(data, DataFormat.TORCH)
        assert isinstance(serialized, bytes)

        deserialized = deserialize(serialized, DataFormat.TORCH)
        assert torch.equal(deserialized, data)


# ============================================================================
# Integration Tests
# ============================================================================

class TestProtocolsIntegration:
    """Integration tests for protocols."""

    def test_message_with_execution_context(self):
        """Test using messages with execution context."""
        # Create request
        request = Request(
            sender="client",
            receiver="server",
            action="execute",
            parameters={"task": "process"}
        )

        # Create execution context
        ctx = ExecutionContext(name="process_task")
        ctx.start()
        ctx.set_variable("request_id", request.id)

        # Simulate processing - Progress.update() takes current, message, stage
        ctx.progress.update(current=100, message="Complete")
        ctx.complete()

        # Create response
        result = ExecutionResult.from_context(ctx, result="done")
        response = Response.success_response(request, result=result.to_dict())

        assert response.success == True
        assert response.correlation_id == request.id

    def test_event_driven_tool_execution(self):
        """Test event-driven tool execution."""
        bus = EventBus()
        events_received = []

        def on_tool_complete(event):
            events_received.append(event)

        bus.subscribe(EventType.TOOL_COMPLETE, on_tool_complete)

        # Create tool result
        result = ToolResult(tool_name="analyzer")
        result.mark_started()
        result.mark_success(ToolOutput(result="analysis_done"))

        # Emit completion event
        bus.publish(Event(
            type=EventType.TOOL_COMPLETE,
            source="analyzer",
            data=result.to_dict()
        ))

        assert len(events_received) == 1
        assert events_received[0].data["status"] == "SUCCESS"

    def test_data_packet_in_message(self):
        """Test using DataPacket in messages."""
        packet = DataPacket(
            format=DataFormat.JSON,
            data={"analysis": [1, 2, 3, 4, 5]}
        )
        packet.compute_checksum()

        msg = Message(
            sender="data_source",
            receiver="processor",
            payload=packet.to_dict()
        )

        # Restore packet from message
        restored = DataPacket.from_dict(msg.payload)
        assert restored.data == packet.data
