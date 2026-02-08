"""
SCIO Scheduler

Task-Scheduling und Ressourcen-Management.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Optional
from uuid import uuid4

from scio.core.config import get_config
from scio.core.logging import get_logger
from scio.core.utils import now_utc

logger = get_logger(__name__)


class TaskPriority(IntEnum):
    """Priorität für Tasks."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ScheduledTask:
    """Ein geplanter Task."""

    id: str
    name: str
    callable: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=now_utc)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None

    def __lt__(self, other: "ScheduledTask") -> bool:
        """Für Priority Queue - höhere Priorität zuerst."""
        return self.priority > other.priority


class Scheduler:
    """
    Task Scheduler für parallele Ausführung.

    Features:
    - Prioritäts-basierte Ausführung
    - Ressourcen-Limits
    - Concurrent Execution mit Limits
    """

    def __init__(self, max_concurrent: int | None = None):
        config = get_config()
        self.max_concurrent = max_concurrent or config.execution.max_concurrent_agents
        self.logger = get_logger(__name__, component="scheduler")

        self._queue: asyncio.PriorityQueue[ScheduledTask] = asyncio.PriorityQueue()
        self._running: dict[str, ScheduledTask] = {}
        self._completed: dict[str, ScheduledTask] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._shutdown = False

    async def start(self) -> None:
        """Startet den Scheduler."""
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._shutdown = False
        self.logger.info("Scheduler started", max_concurrent=self.max_concurrent)

    async def stop(self) -> None:
        """Stoppt den Scheduler."""
        self._shutdown = True
        self.logger.info("Scheduler stopped")

    async def submit(
        self,
        callable: Callable,
        *args: Any,
        name: str | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs: Any,
    ) -> str:
        """
        Fügt einen Task zur Queue hinzu.

        Args:
            callable: Auszuführende Funktion (async oder sync)
            *args: Positionale Argumente
            name: Optionaler Task-Name
            priority: Task-Priorität
            **kwargs: Keyword-Argumente

        Returns:
            Task-ID
        """
        task_id = str(uuid4())[:8]

        task = ScheduledTask(
            id=task_id,
            name=name or callable.__name__,
            callable=callable,
            args=args,
            kwargs=kwargs,
            priority=priority,
        )

        await self._queue.put(task)
        self.logger.debug("Task submitted", task_id=task_id, name=task.name)

        return task_id

    async def run_task(self, task: ScheduledTask) -> Any:
        """Führt einen einzelnen Task aus."""
        if self._semaphore is None:
            raise RuntimeError("Scheduler not started")

        async with self._semaphore:
            self._running[task.id] = task
            task.started_at = now_utc()

            self.logger.debug("Task starting", task_id=task.id, name=task.name)

            try:
                # Führe aus (async oder sync)
                if asyncio.iscoroutinefunction(task.callable):
                    result = await task.callable(*task.args, **task.kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: task.callable(*task.args, **task.kwargs)
                    )

                task.result = result
                task.completed_at = now_utc()

                self.logger.debug(
                    "Task completed",
                    task_id=task.id,
                    name=task.name,
                )

                return result

            except Exception as e:
                task.error = str(e)
                task.completed_at = now_utc()
                self.logger.error(
                    "Task failed",
                    task_id=task.id,
                    name=task.name,
                    error=str(e),
                )
                raise

            finally:
                del self._running[task.id]
                self._completed[task.id] = task

    async def process_queue(self) -> None:
        """Verarbeitet die Task-Queue."""
        while not self._shutdown:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                asyncio.create_task(self.run_task(task))
            except asyncio.TimeoutError:
                continue

    async def run_all(self) -> list[Any]:
        """Führt alle Tasks in der Queue aus und wartet auf Abschluss."""
        tasks = []

        while not self._queue.empty():
            task = await self._queue.get()
            tasks.append(asyncio.create_task(self.run_task(task)))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        return []

    def get_status(self) -> dict[str, Any]:
        """Gibt den aktuellen Status zurück."""
        return {
            "queue_size": self._queue.qsize(),
            "running": len(self._running),
            "completed": len(self._completed),
            "max_concurrent": self.max_concurrent,
        }
