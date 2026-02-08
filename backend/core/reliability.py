#!/usr/bin/env python3
"""
SCIO Reliability Module
Circuit Breaker, Retry-Logik, Graceful Degradation
"""

import time
import threading
import logging
from typing import Any, Callable, Optional, TypeVar, Generic, List, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════

class CircuitState(str, Enum):
    """Zustände des Circuit Breakers"""
    CLOSED = "closed"      # Normal - Requests werden durchgelassen
    OPEN = "open"          # Fehler - Requests werden blockiert
    HALF_OPEN = "half_open"  # Test - Ein Request wird durchgelassen


@dataclass
class CircuitBreakerConfig:
    """Konfiguration für Circuit Breaker"""
    failure_threshold: int = 5        # Anzahl Fehler bis OPEN
    success_threshold: int = 3        # Anzahl Erfolge in HALF_OPEN bis CLOSED
    timeout_seconds: float = 30.0     # Zeit in OPEN bis HALF_OPEN
    half_open_max_calls: int = 1      # Max Calls in HALF_OPEN


class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation

    Verhindert kaskadierende Fehler durch temporäres Blockieren
    von Aufrufen zu fehlerhaften Services.

    Verwendung:
        breaker = CircuitBreaker("my_service")

        try:
            with breaker:
                result = call_external_service()
        except CircuitOpenError:
            result = fallback_value

        # Oder als Decorator:
        @breaker.protect
        def call_service():
            ...
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

        # Statistiken
        self._total_calls = 0
        self._total_failures = 0
        self._total_blocked = 0

    @property
    def state(self) -> CircuitState:
        """Aktueller Zustand (mit automatischem Timeout-Check)"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_try_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
            return self._state

    def _should_try_reset(self) -> bool:
        """Prüft ob Timeout für Reset erreicht ist"""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.config.timeout_seconds

    def __enter__(self):
        """Context Manager Entry"""
        self._before_call()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit"""
        if exc_type is None:
            self._on_success()
        else:
            self._on_failure()
        return False  # Exception nicht unterdrücken

    def _before_call(self):
        """Wird vor jedem Aufruf geprüft"""
        state = self.state  # Triggert Timeout-Check

        with self._lock:
            self._total_calls += 1

            if state == CircuitState.OPEN:
                self._total_blocked += 1
                raise CircuitOpenError(
                    f"Circuit {self.name} is OPEN - calls blocked"
                )

            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._total_blocked += 1
                    raise CircuitOpenError(
                        f"Circuit {self.name} is HALF_OPEN - max calls reached"
                    )
                self._half_open_calls += 1

    def _on_success(self):
        """Wird bei erfolgreichem Aufruf aufgerufen"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset bei Erfolg

    def _on_failure(self):
        """Wird bei fehlgeschlagenem Aufruf aufgerufen"""
        with self._lock:
            self._total_failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Sofort zurück zu OPEN
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (failure)")
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name}: CLOSED -> OPEN ({self._failure_count} failures)")

    def protect(self, func: Callable) -> Callable:
        """Decorator für Circuit Breaker Protection"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    def reset(self):
        """Manueller Reset des Circuit Breakers"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            logger.info(f"Circuit {self.name}: Manual reset to CLOSED")

    def get_stats(self) -> Dict[str, Any]:
        """Statistiken"""
        return {
            'name': self.name,
            'state': self._state.value,
            'failure_count': self._failure_count,
            'total_calls': self._total_calls,
            'total_failures': self._total_failures,
            'total_blocked': self._total_blocked,
            'success_rate': 1 - (self._total_failures / max(1, self._total_calls))
        }


class CircuitOpenError(Exception):
    """Exception wenn Circuit offen ist"""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# RETRY LOGIC
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetryConfig:
    """Konfiguration für Retry-Logik"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Zufällige Variation
    retry_exceptions: tuple = (Exception,)  # Welche Exceptions wiederholen


def retry(config: RetryConfig = None):
    """
    Decorator für automatische Wiederholungsversuche

    Verwendet exponentielles Backoff mit optionalem Jitter.

    Verwendung:
        @retry(RetryConfig(max_retries=5))
        def flaky_function():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retry_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        logger.error(f"Retry exhausted for {func.__name__} after {attempt + 1} attempts")
                        raise

                    # Delay berechnen
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )

                    # Jitter hinzufügen (±25%)
                    if config.jitter:
                        delay *= 0.75 + random.random() * 0.5

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator


def retry_with_fallback(fallback_value: Any = None,
                        config: RetryConfig = None):
    """
    Retry-Decorator mit Fallback-Wert bei Erschöpfung

    Verwendung:
        @retry_with_fallback(fallback_value=[], config=RetryConfig(max_retries=3))
        def get_items():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return retry(config)(func)(*args, **kwargs)
            except Exception as e:
                logger.error(f"All retries failed for {func.__name__}, using fallback: {e}")
                return fallback_value
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# GRACEFUL DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════

class DegradationLevel(str, Enum):
    """Level der Degradation"""
    NORMAL = "normal"
    REDUCED = "reduced"
    MINIMAL = "minimal"
    EMERGENCY = "emergency"


class GracefulDegradation:
    """
    Graceful Degradation Manager

    Ermöglicht kontrollierte Reduktion von Features bei Problemen.

    Verwendung:
        degradation = GracefulDegradation()

        if degradation.is_feature_enabled("advanced_analytics"):
            # Volle Funktion
            result = full_analysis()
        else:
            # Reduzierte Funktion
            result = basic_analysis()
    """

    def __init__(self):
        self._level = DegradationLevel.NORMAL
        self._disabled_features: set = set()
        self._lock = threading.Lock()

        # Feature-Level Mapping
        self._feature_levels = {
            DegradationLevel.NORMAL: set(),  # Keine Einschränkungen
            DegradationLevel.REDUCED: {
                "advanced_analytics",
                "background_tasks",
                "expensive_queries"
            },
            DegradationLevel.MINIMAL: {
                "advanced_analytics",
                "background_tasks",
                "expensive_queries",
                "image_generation",
                "video_generation",
                "training_jobs"
            },
            DegradationLevel.EMERGENCY: {
                "advanced_analytics",
                "background_tasks",
                "expensive_queries",
                "image_generation",
                "video_generation",
                "training_jobs",
                "non_essential_apis",
                "websocket_updates"
            }
        }

    @property
    def level(self) -> DegradationLevel:
        return self._level

    def set_level(self, level: DegradationLevel):
        """Setzt Degradation-Level"""
        with self._lock:
            old_level = self._level
            self._level = level
            logger.warning(f"Degradation level: {old_level.value} -> {level.value}")

    def is_feature_enabled(self, feature: str) -> bool:
        """Prüft ob Feature aktiviert ist"""
        with self._lock:
            if feature in self._disabled_features:
                return False
            disabled = self._feature_levels.get(self._level, set())
            return feature not in disabled

    def disable_feature(self, feature: str):
        """Deaktiviert einzelnes Feature"""
        with self._lock:
            self._disabled_features.add(feature)
            logger.warning(f"Feature disabled: {feature}")

    def enable_feature(self, feature: str):
        """Aktiviert einzelnes Feature"""
        with self._lock:
            self._disabled_features.discard(feature)
            logger.info(f"Feature enabled: {feature}")

    def get_status(self) -> Dict[str, Any]:
        """Status-Übersicht"""
        return {
            'level': self._level.value,
            'disabled_features': list(self._disabled_features),
            'level_disabled': list(self._feature_levels.get(self._level, set()))
        }


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HealthStatus:
    """Status einer Health-Prüfung"""
    name: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0
    checked_at: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """
    Zentrale Health-Check Verwaltung

    Verwendung:
        health = HealthChecker()

        @health.register("database")
        def check_db():
            # True wenn healthy, False wenn nicht
            return db.ping()

        status = health.check_all()
    """

    def __init__(self):
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._last_results: Dict[str, HealthStatus] = {}
        self._lock = threading.Lock()

    def register(self, name: str, timeout_seconds: float = 5.0):
        """Decorator zum Registrieren einer Health-Check-Funktion"""
        def decorator(func: Callable[[], bool]) -> Callable[[], bool]:
            with self._lock:
                self._checks[name] = (func, timeout_seconds)
            return func
        return decorator

    def check(self, name: str) -> HealthStatus:
        """Führt einzelnen Health-Check aus"""
        if name not in self._checks:
            return HealthStatus(name=name, healthy=False, message="Check not found")

        func, timeout = self._checks[name]
        start = time.time()

        try:
            result = [False]
            exception = [None]

            def run_check():
                try:
                    result[0] = func()
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=run_check)
            thread.start()
            thread.join(timeout=timeout)

            latency = (time.time() - start) * 1000

            if thread.is_alive():
                status = HealthStatus(
                    name=name,
                    healthy=False,
                    message="Timeout",
                    latency_ms=latency
                )
            elif exception[0]:
                status = HealthStatus(
                    name=name,
                    healthy=False,
                    message=str(exception[0]),
                    latency_ms=latency
                )
            else:
                status = HealthStatus(
                    name=name,
                    healthy=result[0],
                    message="OK" if result[0] else "Unhealthy",
                    latency_ms=latency
                )

        except Exception as e:
            status = HealthStatus(
                name=name,
                healthy=False,
                message=str(e),
                latency_ms=(time.time() - start) * 1000
            )

        with self._lock:
            self._last_results[name] = status

        return status

    def check_all(self) -> Dict[str, HealthStatus]:
        """Führt alle Health-Checks aus"""
        results = {}
        for name in self._checks:
            results[name] = self.check(name)
        return results

    def is_healthy(self) -> bool:
        """Prüft ob alle Checks healthy sind"""
        results = self.check_all()
        return all(status.healthy for status in results.values())

    def get_last_results(self) -> Dict[str, HealthStatus]:
        """Gibt letzte Ergebnisse zurück"""
        with self._lock:
            return dict(self._last_results)


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES
# ═══════════════════════════════════════════════════════════════════════════

# Globale Circuit Breaker Registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(name: str,
                        config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Holt oder erstellt Circuit Breaker"""
    with _breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


# Globale Health Checker Instanz
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Gibt globale HealthChecker-Instanz zurück"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# Globale Degradation Instanz
_degradation: Optional[GracefulDegradation] = None


def get_degradation() -> GracefulDegradation:
    """Gibt globale GracefulDegradation-Instanz zurück"""
    global _degradation
    if _degradation is None:
        _degradation = GracefulDegradation()
    return _degradation
