"""Tests für die Execution Engine."""

import pytest
from scio.execution.engine import ExecutionEngine, ExecutionStatus, StepResult
from scio.execution.sandbox import Sandbox, SandboxConfig
from scio.execution.scheduler import Scheduler, TaskPriority
from scio.parser.schema import ExperimentSchema


class TestExecutionEngine:
    """Tests für ExecutionEngine."""

    def test_create_engine(self):
        """Testet Engine-Erstellung."""
        engine = ExecutionEngine()
        assert engine is not None

    def test_register_handler(self):
        """Testet Handler-Registrierung."""
        engine = ExecutionEngine()

        async def dummy_handler(step, context):
            return {"result": "ok"}

        engine.register_step_handler("test", dummy_handler)
        assert "test" in engine._step_handlers


class TestSandbox:
    """Tests für Sandbox."""

    def test_sandbox_disabled(self):
        """Testet deaktivierte Sandbox."""
        config = SandboxConfig(enabled=False)
        sandbox = Sandbox(config)

        # Alles sollte erlaubt sein
        assert sandbox.check_path_access("/any/path")
        assert sandbox.check_module_import("os")

    def test_path_access_allowed(self, tmp_path):
        """Testet erlaubten Pfad-Zugriff."""
        from pathlib import Path

        config = SandboxConfig(
            enabled=True,
            allowed_paths=[tmp_path],
        )
        sandbox = Sandbox(config)

        assert sandbox.check_path_access(tmp_path / "file.txt")

    def test_path_access_denied(self, tmp_path):
        """Testet verweigerten Pfad-Zugriff."""
        from pathlib import Path
        from scio.core.exceptions import SecurityError

        config = SandboxConfig(
            enabled=True,
            allowed_paths=[tmp_path],
        )
        sandbox = Sandbox(config)

        # Ein anderer Pfad sollte blockiert werden
        other_path = tmp_path.parent / "other_folder" / "file.txt"
        with pytest.raises(SecurityError):
            sandbox.check_path_access(other_path)

    def test_module_blocked(self):
        """Testet blockierte Module."""
        from scio.core.exceptions import SecurityError

        config = SandboxConfig(
            enabled=True,
            blocked_modules=["subprocess", "os.system"],
        )
        sandbox = Sandbox(config)

        with pytest.raises(SecurityError):
            sandbox.check_module_import("subprocess")

        with pytest.raises(SecurityError):
            sandbox.check_module_import("os.system")

    def test_network_blocked(self):
        """Testet blockierten Netzwerkzugriff."""
        from scio.core.exceptions import SecurityError

        config = SandboxConfig(enabled=True, network_enabled=False)
        sandbox = Sandbox(config)

        with pytest.raises(SecurityError):
            sandbox.check_network_access("example.com", 80)

    def test_restricted_globals(self):
        """Testet eingeschränkte Globals."""
        sandbox = Sandbox()
        globals_dict = sandbox.create_restricted_globals()

        assert "print" in globals_dict["__builtins__"]
        assert "len" in globals_dict["__builtins__"]
        # Gefährliche Funktionen sollten fehlen
        assert "eval" not in globals_dict["__builtins__"]
        assert "exec" not in globals_dict["__builtins__"]
        assert "open" not in globals_dict["__builtins__"]


@pytest.mark.asyncio
class TestScheduler:
    """Async Tests für Scheduler."""

    async def test_submit_and_run(self):
        """Testet Task-Submission und -Ausführung."""
        scheduler = Scheduler(max_concurrent=2)
        await scheduler.start()

        results = []

        async def task(x):
            return x * 2

        await scheduler.submit(task, 5)
        await scheduler.submit(task, 10)

        all_results = await scheduler.run_all()

        assert len(all_results) == 2
        assert 10 in all_results
        assert 20 in all_results

        await scheduler.stop()

    async def test_priority_ordering(self):
        """Testet Prioritäts-Reihenfolge."""
        scheduler = Scheduler(max_concurrent=1)
        await scheduler.start()

        order = []

        async def task(name):
            order.append(name)
            return name

        # Submitten in umgekehrter Prioritäts-Reihenfolge
        await scheduler.submit(task, "low", priority=TaskPriority.LOW)
        await scheduler.submit(task, "high", priority=TaskPriority.HIGH)
        await scheduler.submit(task, "normal", priority=TaskPriority.NORMAL)

        await scheduler.run_all()

        # Höchste Priorität sollte zuerst kommen
        assert order[0] == "high"

        await scheduler.stop()

    async def test_status(self):
        """Testet Status-Abfrage."""
        scheduler = Scheduler(max_concurrent=4)
        await scheduler.start()

        async def task():
            return "done"

        await scheduler.submit(task)
        await scheduler.submit(task)

        status = scheduler.get_status()
        assert status["queue_size"] == 2
        assert status["max_concurrent"] == 4

        await scheduler.stop()
