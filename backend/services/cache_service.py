#!/usr/bin/env python3
"""
SCIO - Cache Service (MEGA-UPGRADE)
High-Performance Caching Layer mit LRU, TTL und Redis Support

Features:
- LRU-Cache für häufige Prompts (TTL: 5min)
- Redis-Integration für verteiltes Caching
- HTTP Cache-Control Headers
- ETag-basierte Invalidierung
- Semantic Caching für LLM-Responses
"""

import time
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Callable, Tuple
from functools import wraps

from backend.config import Config

logger = logging.getLogger(__name__)

# Try Redis
REDIS_AVAILABLE = False
redis_client = None
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# CACHE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_TTL = 300  # 5 minutes
LLM_CACHE_TTL = 600  # 10 minutes for LLM responses
IMAGE_CACHE_TTL = 3600  # 1 hour for image metadata
MAX_CACHE_SIZE = 10000  # Maximum entries in LRU cache
REDIS_PREFIX = "scio:cache:"


@dataclass
class CacheEntry:
    """Einzelner Cache-Eintrag"""
    value: Any
    created_at: float
    expires_at: float
    etag: str
    hits: int = 0
    size_bytes: int = 0


@dataclass
class CacheStats:
    """Cache-Statistiken"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0


class LRUCache:
    """
    Thread-safe LRU Cache mit TTL Support

    Features:
    - O(1) Lookup und Insertion
    - Automatische Expiration
    - Size-basierte Eviction
    - Thread-safe
    """

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def _generate_etag(self, value: Any) -> str:
        """Generiert ETag für einen Wert"""
        content = json.dumps(value, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _calculate_size(self, value: Any) -> int:
        """Schätzt die Größe eines Wertes in Bytes"""
        try:
            return len(json.dumps(value, default=str).encode())
        except Exception:
            return 1000  # Default estimate

    def get(self, key: str, check_etag: str = None) -> Tuple[Optional[Any], Optional[str]]:
        """
        Holt Wert aus dem Cache

        Args:
            key: Cache-Schlüssel
            check_etag: Optionaler ETag zum Vergleichen

        Returns:
            Tuple von (Wert, ETag) oder (None, None) wenn nicht gefunden
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None, None

            entry = self._cache[key]

            # Check expiration
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None, None

            # Check ETag (for 304 Not Modified)
            if check_etag and entry.etag == check_etag:
                entry.hits += 1
                self._stats.hits += 1
                return "NOT_MODIFIED", entry.etag

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1

            return entry.value, entry.etag

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = DEFAULT_TTL,
    ) -> str:
        """
        Speichert Wert im Cache

        Args:
            key: Cache-Schlüssel
            value: Zu speichernder Wert
            ttl: Time-to-Live in Sekunden

        Returns:
            ETag des gespeicherten Wertes
        """
        with self._lock:
            now = time.time()
            etag = self._generate_etag(value)
            size = self._calculate_size(value)

            entry = CacheEntry(
                value=value,
                created_at=now,
                expires_at=now + ttl,
                etag=etag,
                hits=0,
                size_bytes=size,
            )

            # Wenn Key existiert, entfernen (um am Ende neu einzufügen)
            if key in self._cache:
                self._stats.total_size_bytes -= self._cache[key].size_bytes
                del self._cache[key]

            # Eviction wenn Cache voll
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                self._stats.total_size_bytes -= self._cache[oldest_key].size_bytes
                del self._cache[oldest_key]
                self._stats.evictions += 1

            self._cache[key] = entry
            self._stats.total_size_bytes += size
            self._stats.entry_count = len(self._cache)

            return etag

    def delete(self, key: str) -> bool:
        """Löscht einen Eintrag"""
        with self._lock:
            if key in self._cache:
                self._stats.total_size_bytes -= self._cache[key].size_bytes
                del self._cache[key]
                self._stats.entry_count = len(self._cache)
                return True
            return False

    def clear(self):
        """Leert den gesamten Cache"""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    def cleanup_expired(self) -> int:
        """Entfernt abgelaufene Einträge"""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._cache.items()
                if v.expires_at < now
            ]
            for key in expired_keys:
                self._stats.total_size_bytes -= self._cache[key].size_bytes
                del self._cache[key]
                self._stats.evictions += 1

            self._stats.entry_count = len(self._cache)
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Cache-Statistiken zurück"""
        with self._lock:
            total_requests = self._stats.hits + self._stats.misses
            hit_rate = (
                self._stats.hits / total_requests * 100
                if total_requests > 0 else 0.0
            )

            return {
                'hits': self._stats.hits,
                'misses': self._stats.misses,
                'hit_rate_percent': round(hit_rate, 2),
                'evictions': self._stats.evictions,
                'entry_count': len(self._cache),
                'max_size': self._max_size,
                'total_size_bytes': self._stats.total_size_bytes,
                'total_size_mb': round(self._stats.total_size_bytes / 1024 / 1024, 2),
            }


class RedisCache:
    """
    Redis-basierter Cache für verteiltes Caching

    Features:
    - Verteiltes Caching über mehrere Server
    - Automatische Expiration (Redis TTL)
    - Serialisierung mit JSON
    """

    def __init__(self, host: str = None, port: int = None, db: int = 0):
        self._host = host or getattr(Config, 'REDIS_HOST', 'localhost')
        self._port = port or getattr(Config, 'REDIS_PORT', 6379)
        self._db = db
        self._client = None
        self._available = False
        self._connect()

    def _connect(self):
        """Verbindet zu Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis nicht verfügbar (redis-py nicht installiert)")
            return

        try:
            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            self._client.ping()
            self._available = True
            logger.info(f"Redis verbunden: {self._host}:{self._port}")
        except Exception as e:
            logger.warning(f"Redis Verbindung fehlgeschlagen: {e}")
            self._available = False

    def get(self, key: str) -> Tuple[Optional[Any], Optional[str]]:
        """Holt Wert aus Redis"""
        if not self._available:
            return None, None

        try:
            full_key = REDIS_PREFIX + key
            data = self._client.get(full_key)
            if data:
                entry = json.loads(data)
                return entry['value'], entry.get('etag')
            return None, None
        except Exception as e:
            logger.error(f"Redis get Fehler: {e}")
            return None, None

    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> Optional[str]:
        """Speichert Wert in Redis"""
        if not self._available:
            return None

        try:
            full_key = REDIS_PREFIX + key
            etag = hashlib.md5(json.dumps(value, default=str).encode()).hexdigest()[:16]
            entry = {'value': value, 'etag': etag}
            self._client.setex(full_key, ttl, json.dumps(entry, default=str))
            return etag
        except Exception as e:
            logger.error(f"Redis set Fehler: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Löscht Wert aus Redis"""
        if not self._available:
            return False

        try:
            full_key = REDIS_PREFIX + key
            return bool(self._client.delete(full_key))
        except Exception as e:
            logger.error(f"Redis delete Fehler: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Löscht alle Schlüssel die dem Pattern entsprechen"""
        if not self._available:
            return 0

        try:
            keys = self._client.keys(REDIS_PREFIX + pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear_pattern Fehler: {e}")
            return 0

    @property
    def available(self) -> bool:
        return self._available


class CacheService:
    """
    Haupt-Cache-Service mit Multi-Layer Caching

    Features:
    - L1: In-Memory LRU Cache (schnell, lokal)
    - L2: Redis Cache (verteilt, persistent)
    - Automatische Fallback-Strategie
    - Spezialisierte Cache-Methoden für verschiedene Datentypen
    """

    def __init__(self, use_redis: bool = True):
        self._l1 = LRUCache(max_size=MAX_CACHE_SIZE)
        self._l2 = RedisCache() if use_redis else None
        self._use_redis = use_redis and self._l2 and self._l2.available

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._running = True
        self._cleanup_thread.start()

        logger.info(f"CacheService initialisiert (Redis: {self._use_redis})")

    def _cleanup_loop(self):
        """Periodisches Cleanup von abgelaufenen Einträgen"""
        while self._running:
            try:
                time.sleep(60)  # Every minute
                cleaned = self._l1.cleanup_expired()
                if cleaned > 0:
                    logger.debug(f"Cache cleanup: {cleaned} Einträge entfernt")
            except Exception as e:
                logger.error(f"Cache cleanup Fehler: {e}")

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generiert einen Cache-Schlüssel aus den Argumenten"""
        key_parts = [prefix] + list(str(a) for a in args)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        key_str = ":".join(key_parts)

        # Hash wenn zu lang
        if len(key_str) > 200:
            return prefix + ":" + hashlib.md5(key_str.encode()).hexdigest()

        return key_str

    def get(
        self,
        key: str,
        check_etag: str = None,
    ) -> Tuple[Optional[Any], Optional[str], bool]:
        """
        Holt Wert aus dem Cache (L1 -> L2 Fallback)

        Returns:
            Tuple von (Wert, ETag, ist_not_modified)
        """
        # L1 Check
        value, etag = self._l1.get(key, check_etag)
        if value == "NOT_MODIFIED":
            return None, etag, True
        if value is not None:
            return value, etag, False

        # L2 Check
        if self._use_redis:
            value, etag = self._l2.get(key)
            if value is not None:
                # Populate L1
                self._l1.set(key, value, DEFAULT_TTL)
                return value, etag, False

        return None, None, False

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = DEFAULT_TTL,
        l1_only: bool = False,
    ) -> str:
        """
        Speichert Wert im Cache

        Args:
            key: Cache-Schlüssel
            value: Zu speichernder Wert
            ttl: Time-to-Live in Sekunden
            l1_only: Nur im lokalen Cache speichern

        Returns:
            ETag des Wertes
        """
        etag = self._l1.set(key, value, ttl)

        if self._use_redis and not l1_only:
            self._l2.set(key, value, ttl)

        return etag

    def delete(self, key: str) -> bool:
        """Löscht Wert aus beiden Cache-Ebenen"""
        result = self._l1.delete(key)
        if self._use_redis:
            result = self._l2.delete(key) or result
        return result

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidiert alle Schlüssel die dem Pattern entsprechen"""
        count = 0
        if self._use_redis:
            count = self._l2.clear_pattern(pattern)
        # L1 hat keine Pattern-Unterstützung, daher komplett leeren wenn Pattern "*"
        if pattern == "*":
            self._l1.clear()
        return count

    # ═══════════════════════════════════════════════════════════
    # SPEZIALISIERTE CACHE-METHODEN
    # ═══════════════════════════════════════════════════════════

    def cache_llm_response(
        self,
        model: str,
        messages: list,
        response: str,
        ttl: int = LLM_CACHE_TTL,
    ) -> str:
        """Cached LLM Response basierend auf Prompt"""
        # Normalisiere Messages für konsistenten Key
        normalized = json.dumps(messages, sort_keys=True)
        key = self._generate_key("llm", model, normalized)
        return self.set(key, response, ttl)

    def get_llm_response(
        self,
        model: str,
        messages: list,
    ) -> Optional[str]:
        """Holt gecachte LLM Response"""
        normalized = json.dumps(messages, sort_keys=True)
        key = self._generate_key("llm", model, normalized)
        value, _, _ = self.get(key)
        return value

    def cache_image_metadata(
        self,
        prompt: str,
        model: str,
        metadata: dict,
        ttl: int = IMAGE_CACHE_TTL,
    ) -> str:
        """Cached Image Generation Metadata"""
        key = self._generate_key("image", model, prompt[:100])
        return self.set(key, metadata, ttl)

    def cache_embedding(
        self,
        text: str,
        model: str,
        embedding: list,
        ttl: int = 3600,
    ) -> str:
        """Cached Embedding"""
        # Hash für lange Texte
        text_hash = hashlib.md5(text.encode()).hexdigest()
        key = self._generate_key("embedding", model, text_hash)
        return self.set(key, embedding, ttl)

    def get_embedding(
        self,
        text: str,
        model: str,
    ) -> Optional[list]:
        """Holt gecachtes Embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        key = self._generate_key("embedding", model, text_hash)
        value, _, _ = self.get(key)
        return value

    # ═══════════════════════════════════════════════════════════
    # HTTP CACHE HEADERS
    # ═══════════════════════════════════════════════════════════

    def get_cache_headers(
        self,
        etag: str,
        ttl: int = DEFAULT_TTL,
        private: bool = True,
    ) -> Dict[str, str]:
        """
        Generiert HTTP Cache-Control Headers

        Args:
            etag: ETag des Inhalts
            ttl: Max-Age in Sekunden
            private: Ob Cache privat sein soll

        Returns:
            Dict mit HTTP Headers
        """
        cache_control = f"{'private' if private else 'public'}, max-age={ttl}"

        return {
            'Cache-Control': cache_control,
            'ETag': f'"{etag}"',
            'Vary': 'Accept, Accept-Encoding',
        }

    def check_if_none_match(self, request_etag: str, current_etag: str) -> bool:
        """
        Prüft If-None-Match Header für 304 Response

        Returns:
            True wenn 304 Not Modified zurückgegeben werden sollte
        """
        if not request_etag:
            return False

        # Remove quotes if present
        request_etag = request_etag.strip('"')
        current_etag = current_etag.strip('"')

        return request_etag == current_etag

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Cache-Statistiken zurück"""
        return {
            'l1_cache': self._l1.get_stats(),
            'redis_available': self._use_redis,
            'default_ttl': DEFAULT_TTL,
            'llm_cache_ttl': LLM_CACHE_TTL,
            'image_cache_ttl': IMAGE_CACHE_TTL,
        }

    def stop(self):
        """Stoppt den Cache-Service"""
        self._running = False


# ═══════════════════════════════════════════════════════════════
# DECORATOR FÜR AUTOMATISCHES CACHING
# ═══════════════════════════════════════════════════════════════

def cached(
    ttl: int = DEFAULT_TTL,
    key_prefix: str = None,
    cache_service: CacheService = None,
):
    """
    Decorator für automatisches Caching von Funktionsergebnissen

    Usage:
        @cached(ttl=300, key_prefix="my_func")
        def my_expensive_function(arg1, arg2):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache service
            cs = cache_service or get_cache_service()

            # Generate key
            prefix = key_prefix or func.__name__
            key = cs._generate_key(prefix, *args, **kwargs)

            # Check cache
            value, _, _ = cs.get(key)
            if value is not None:
                return value

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cs.set(key, result, ttl)

            return result

        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════

_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Gibt Singleton-Instanz zurück"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
