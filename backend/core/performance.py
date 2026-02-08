#!/usr/bin/env python3
"""
SCIO Performance Module
Optimierte Datenstrukturen und Caching für bessere Performance
"""

import time
import threading
import heapq
from typing import Any, Dict, Optional, Callable, TypeVar, Generic, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════
# LRU CACHE MIT TTL
# ═══════════════════════════════════════════════════════════════════════════

class LRUCache(Generic[T]):
    """
    Thread-safe LRU Cache mit optionalem TTL

    Verwendung:
        cache = LRUCache(max_size=1000, ttl_seconds=300)
        cache.set('key', value)
        value = cache.get('key')
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str, default: T = None) -> Optional[T]:
        """Holt Wert aus Cache"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default

            value, timestamp = self._cache[key]

            # TTL prüfen
            if self.ttl_seconds and time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return default

            # An Ende verschieben (Most Recently Used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: T):
        """Setzt Wert in Cache"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                # Ältesten Eintrag entfernen
                self._cache.popitem(last=False)

            self._cache[key] = (value, time.time())

    def delete(self, key: str) -> bool:
        """Löscht Eintrag"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Leert Cache"""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Entfernt abgelaufene Einträge, gibt Anzahl zurück"""
        if not self.ttl_seconds:
            return 0

        with self._lock:
            now = time.time()
            expired = [
                key for key, (_, timestamp) in self._cache.items()
                if now - timestamp > self.ttl_seconds
            ]
            for key in expired:
                del self._cache[key]
            return len(expired)

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate
        }


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDED QUEUE MIT EVICTION
# ═══════════════════════════════════════════════════════════════════════════

class BoundedQueue(Generic[T]):
    """
    Thread-safe Queue mit maximaler Größe und automatischer Eviction

    Verwendung:
        queue = BoundedQueue(max_size=1000)
        queue.append(item)
        items = queue.get_all()
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._items: List[T] = []
        self._lock = threading.Lock()
        self._evicted_count = 0

    def append(self, item: T):
        """Fügt Item hinzu, evicted ältestes wenn voll"""
        with self._lock:
            if len(self._items) >= self.max_size:
                # Älteste 10% entfernen
                evict_count = max(1, self.max_size // 10)
                self._items = self._items[evict_count:]
                self._evicted_count += evict_count

            self._items.append(item)

    def get_all(self) -> List[T]:
        """Gibt alle Items zurück"""
        with self._lock:
            return list(self._items)

    def get_latest(self, n: int = 100) -> List[T]:
        """Gibt die letzten n Items zurück"""
        with self._lock:
            return list(self._items[-n:])

    def clear(self):
        """Leert Queue"""
        with self._lock:
            self._items.clear()

    @property
    def size(self) -> int:
        return len(self._items)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'max_size': self.max_size,
            'evicted_count': self._evicted_count
        }


# ═══════════════════════════════════════════════════════════════════════════
# INDEXED COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

class IndexedCollection(Generic[T]):
    """
    Collection mit automatischer Indexierung für schnelle Suche

    Verwendung:
        coll = IndexedCollection(['type', 'category'])
        coll.add('id1', {'type': 'foo', 'category': 'bar', 'data': ...})
        results = coll.find_by_index('type', 'foo')
    """

    def __init__(self, index_fields: List[str], max_size: int = 100000):
        self.index_fields = index_fields
        self.max_size = max_size
        self._items: Dict[str, T] = {}
        self._indices: Dict[str, Dict[Any, set]] = {
            field: {} for field in index_fields
        }
        self._lock = threading.Lock()

    def add(self, item_id: str, item: T) -> bool:
        """Fügt Item hinzu und aktualisiert Indices"""
        with self._lock:
            # Größenlimit prüfen
            if len(self._items) >= self.max_size:
                return False

            # Altes Item entfernen falls existiert
            if item_id in self._items:
                self._remove_from_indices(item_id)

            self._items[item_id] = item

            # Indices aktualisieren
            for field in self.index_fields:
                value = self._get_field_value(item, field)
                if value is not None:
                    if value not in self._indices[field]:
                        self._indices[field][value] = set()
                    self._indices[field][value].add(item_id)

            return True

    def get(self, item_id: str) -> Optional[T]:
        """Holt Item nach ID"""
        with self._lock:
            return self._items.get(item_id)

    def remove(self, item_id: str) -> bool:
        """Entfernt Item"""
        with self._lock:
            if item_id not in self._items:
                return False
            self._remove_from_indices(item_id)
            del self._items[item_id]
            return True

    def find_by_index(self, field: str, value: Any) -> List[T]:
        """Sucht nach Index-Wert (O(1) Lookup)"""
        with self._lock:
            if field not in self._indices:
                return []
            ids = self._indices[field].get(value, set())
            return [self._items[id] for id in ids if id in self._items]

    def find_by_multiple(self, criteria: Dict[str, Any]) -> List[T]:
        """Sucht nach mehreren Index-Werten (Intersection)"""
        with self._lock:
            result_ids = None

            for field, value in criteria.items():
                if field not in self._indices:
                    continue
                ids = self._indices[field].get(value, set())
                if result_ids is None:
                    result_ids = ids.copy()
                else:
                    result_ids &= ids

            if result_ids is None:
                return []
            return [self._items[id] for id in result_ids if id in self._items]

    def _remove_from_indices(self, item_id: str):
        """Entfernt Item aus allen Indices"""
        item = self._items.get(item_id)
        if not item:
            return

        for field in self.index_fields:
            value = self._get_field_value(item, field)
            if value is not None and value in self._indices[field]:
                self._indices[field][value].discard(item_id)
                if not self._indices[field][value]:
                    del self._indices[field][value]

    def _get_field_value(self, item: T, field: str) -> Any:
        """Holt Feldwert von Item"""
        if isinstance(item, dict):
            return item.get(field)
        return getattr(item, field, None)

    @property
    def size(self) -> int:
        return len(self._items)

    def get_all(self) -> List[T]:
        with self._lock:
            return list(self._items.values())


# ═══════════════════════════════════════════════════════════════════════════
# THREAD POOL EXECUTOR MIT CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ThreadPoolTask:
    """Task für Thread Pool"""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    priority: int = 0
    submitted_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        # Höhere Priorität = früher ausführen
        return (-self.priority, self.submitted_at) < (-other.priority, other.submitted_at)


class BoundedThreadPool:
    """
    Thread Pool mit Prioritäten und Callbacks

    Verwendung:
        pool = BoundedThreadPool(max_workers=10, queue_size=1000)
        pool.submit(func, args, callback=on_done, priority=1)
    """

    def __init__(self, max_workers: int = 10, queue_size: int = 1000):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self._task_queue: List[ThreadPoolTask] = []
        self._queue_lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._task_counter = 0
        self._completed = 0
        self._failed = 0

    def start(self):
        """Startet Worker-Threads"""
        if self._running:
            return

        self._running = True
        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self, timeout: float = 5.0):
        """Stoppt alle Worker"""
        self._running = False
        for w in self._workers:
            w.join(timeout=timeout)
        self._workers.clear()

    def submit(self, func: Callable,
               args: tuple = (),
               kwargs: Dict = None,
               callback: Callable = None,
               error_callback: Callable = None,
               priority: int = 0) -> Optional[str]:
        """
        Submittet Task zur Ausführung

        Returns:
            Task-ID oder None wenn Queue voll
        """
        with self._queue_lock:
            if len(self._task_queue) >= self.queue_size:
                return None

            self._task_counter += 1
            task = ThreadPoolTask(
                id=f"task_{self._task_counter}",
                func=func,
                args=args,
                kwargs=kwargs or {},
                callback=callback,
                error_callback=error_callback,
                priority=priority
            )
            heapq.heappush(self._task_queue, task)
            return task.id

    def _worker_loop(self):
        """Worker-Thread Hauptschleife"""
        while self._running:
            task = None
            with self._queue_lock:
                if self._task_queue:
                    task = heapq.heappop(self._task_queue)

            if task:
                try:
                    result = task.func(*task.args, **task.kwargs)
                    self._completed += 1
                    if task.callback:
                        try:
                            task.callback(result)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                except Exception as e:
                    self._failed += 1
                    if task.error_callback:
                        try:
                            task.error_callback(e)
                        except Exception as cb_e:
                            logger.error(f"Error callback error: {cb_e}")
            else:
                time.sleep(0.01)  # Kurz warten wenn keine Tasks

    @property
    def pending_tasks(self) -> int:
        return len(self._task_queue)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'workers': len(self._workers),
            'pending': self.pending_tasks,
            'completed': self._completed,
            'failed': self._failed,
            'running': self._running
        }


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class MemoryTracker:
    """
    Überwacht Speicherverbrauch von Objekten

    Verwendung:
        tracker = MemoryTracker(warning_threshold_mb=500)
        tracker.track('cache', cache_object)
        tracker.check()  # Loggt Warnung wenn Limit überschritten
    """

    def __init__(self, warning_threshold_mb: float = 500):
        self.warning_threshold_bytes = warning_threshold_mb * 1024 * 1024
        self._tracked: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def track(self, name: str, obj: Any):
        """Trackt ein Objekt"""
        with self._lock:
            self._tracked[name] = obj

    def untrack(self, name: str):
        """Entfernt Tracking"""
        with self._lock:
            if name in self._tracked:
                del self._tracked[name]

    def get_size(self, obj: Any) -> int:
        """Schätzt Speicherverbrauch eines Objekts"""
        import sys
        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            size += sum(self.get_size(k) + self.get_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(self.get_size(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += self.get_size(obj.__dict__)

        return size

    def check(self) -> Dict[str, int]:
        """Prüft Speicherverbrauch und loggt Warnungen"""
        with self._lock:
            sizes = {}
            total = 0

            for name, obj in self._tracked.items():
                size = self.get_size(obj)
                sizes[name] = size
                total += size

                if size > self.warning_threshold_bytes:
                    logger.warning(
                        f"Memory warning: {name} uses {size / 1024 / 1024:.1f}MB"
                    )

            if total > self.warning_threshold_bytes * 2:
                logger.warning(
                    f"Total tracked memory: {total / 1024 / 1024:.1f}MB"
                )

            return sizes


# ═══════════════════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════════════════

def memoize(ttl_seconds: float = 60.0, max_size: int = 1000):
    """
    Decorator für Memoization mit TTL

    Verwendung:
        @memoize(ttl_seconds=300)
        def expensive_function(arg):
            ...
    """
    cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Cache-Key generieren
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator


def timeout(seconds: float = 30.0, default: Any = None):
    """
    Decorator für Timeout bei Funktionsausführung

    Verwendung:
        @timeout(seconds=5.0)
        def slow_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [default]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                logger.warning(f"Timeout in {func.__name__} after {seconds}s")
                return default

            if exception[0]:
                raise exception[0]

            return result[0]
        return wrapper
    return decorator
