"""Tests für das Memory-System."""

import pytest
from scio.memory.store import MemoryStore, MemoryEntry
from scio.memory.context import ContextManager, ExecutionContext


class TestMemoryStore:
    """Tests für MemoryStore."""

    def test_set_and_get(self):
        """Testet Speichern und Abrufen."""
        store = MemoryStore()
        store.set("key1", "value1")

        assert store.get("key1") == "value1"
        assert store.exists("key1")

    def test_get_default(self):
        """Testet Standardwert bei fehlendem Schlüssel."""
        store = MemoryStore()
        assert store.get("nonexistent", "default") == "default"

    def test_delete(self):
        """Testet Löschen."""
        store = MemoryStore()
        store.set("key", "value")
        assert store.delete("key")
        assert not store.exists("key")

    def test_ttl(self):
        """Testet Time-to-Live (vereinfacht)."""
        import time
        store = MemoryStore()
        store.set("key", "value", ttl_seconds=1)  # 1 Sekunde TTL

        # Sollte noch existieren
        assert store.get("key") == "value"

        # Warte bis TTL abgelaufen
        time.sleep(1.1)

        # Jetzt sollte es weg sein
        assert store.get("key") is None

    def test_find_by_tag(self):
        """Testet Suche nach Tags."""
        store = MemoryStore()
        store.set("key1", "value1", tags=["important"])
        store.set("key2", "value2", tags=["important", "other"])
        store.set("key3", "value3", tags=["other"])

        results = store.find_by_tag("important")
        assert len(results) == 2

    def test_find_by_prefix(self):
        """Testet Suche nach Prefix."""
        store = MemoryStore()
        store.set("user:1", "alice")
        store.set("user:2", "bob")
        store.set("item:1", "thing")

        results = store.find_by_prefix("user:")
        assert len(results) == 2

    def test_clear(self):
        """Testet Löschen aller Einträge."""
        store = MemoryStore()
        store.set("key1", "value1")
        store.set("key2", "value2")

        store.clear()
        assert store.size() == 0


class TestExecutionContext:
    """Tests für ExecutionContext."""

    def test_create_context(self):
        """Testet Context-Erstellung."""
        ctx = ExecutionContext(
            execution_id="exec-1",
            experiment_name="test_exp",
            parameters={"param1": "value1"},
        )

        assert ctx.execution_id == "exec-1"
        assert ctx.experiment_name == "test_exp"

    def test_get_set_variable(self):
        """Testet Variablen-Zugriff."""
        ctx = ExecutionContext(
            execution_id="exec-1",
            experiment_name="test",
        )

        ctx.set_variable("x", 42)
        assert ctx.get_variable("x") == 42

    def test_parameter_fallback(self):
        """Testet Fallback auf Parameter."""
        ctx = ExecutionContext(
            execution_id="exec-1",
            experiment_name="test",
            parameters={"default_value": 100},
        )

        # Variable nicht gesetzt, sollte Parameter zurückgeben
        assert ctx.get_variable("default_value") == 100

    def test_step_output(self):
        """Testet Step-Output-Zugriff."""
        ctx = ExecutionContext(
            execution_id="exec-1",
            experiment_name="test",
        )

        ctx.set_step_output("step1", {"result": "ok", "count": 10})

        assert ctx.get_step_output("step1", "result") == "ok"
        assert ctx.get_step_output("step1", "count") == 10
        assert ctx.get_step_output("step1", "nonexistent") is None

    def test_resolve_reference(self):
        """Testet Referenz-Auflösung."""
        ctx = ExecutionContext(
            execution_id="exec-1",
            experiment_name="test",
            parameters={"seed": 42},
        )
        ctx.set_variable("counter", 5)
        ctx.set_step_output("load", {"data": [1, 2, 3]})

        # Parameter-Referenz
        assert ctx.resolve_reference("${seed}") == 42

        # Variable-Referenz
        assert ctx.resolve_reference("${var.counter}") == 5

        # Step-Output-Referenz
        assert ctx.resolve_reference("${step.load.data}") == [1, 2, 3]


class TestContextManager:
    """Tests für ContextManager."""

    def test_create_context(self):
        """Testet Context-Erstellung über Manager."""
        manager = ContextManager()
        ctx = manager.create_context("test_experiment", {"param": "value"})

        assert ctx is not None
        assert ctx.experiment_name == "test_experiment"
        assert ctx.parameters["param"] == "value"

    def test_get_context(self):
        """Testet Context-Abruf."""
        manager = ContextManager()
        ctx = manager.create_context("test", {})

        retrieved = manager.get_context(ctx.execution_id)
        assert retrieved == ctx

    def test_checkpoint(self):
        """Testet Checkpoint-Funktionalität."""
        manager = ContextManager()
        ctx = manager.create_context("test", {"x": 1})
        ctx.set_variable("progress", 50)

        # Speichere Checkpoint
        checkpoint_id = manager.save_checkpoint(ctx)
        assert checkpoint_id is not None

        # Modifiziere Context
        ctx.set_variable("progress", 100)

        # Der gespeicherte Checkpoint enthält den Zustand zum Zeitpunkt des Speicherns
        # Prüfe dass der Checkpoint im Store existiert
        key = f"checkpoint:{ctx.execution_id}:{checkpoint_id}"
        stored = manager.memory.get(key)
        assert stored is not None
        assert stored["variables"]["progress"] == 50
