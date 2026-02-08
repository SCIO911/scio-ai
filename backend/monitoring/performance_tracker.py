#!/usr/bin/env python3
"""
SCIO - Performance Tracker
Umfassendes Performance-Monitoring und Reporting
"""

import time
import logging
import statistics
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metriken für einen einzelnen Request"""
    request_id: str
    model: str
    operation: str
    start_time: float
    end_time: float = 0
    tokens_input: int = 0
    tokens_output: int = 0
    success: bool = True
    error: Optional[str] = None
    cost_cents: float = 0
    gpu_seconds: float = 0

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0

    @property
    def tokens_per_second(self) -> float:
        duration = self.end_time - self.start_time
        return self.tokens_output / duration if duration > 0 else 0


@dataclass
class AggregatedMetrics:
    """Aggregierte Metriken für einen Zeitraum"""
    period_start: datetime
    period_end: datetime
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_latency_ms: float = 0
    total_cost_cents: float = 0
    total_gpu_seconds: float = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=10000))

    @property
    def success_rate(self) -> float:
        return self.success_count / self.request_count if self.request_count > 0 else 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.request_count if self.request_count > 0 else 0

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(list(self.latencies))
        idx = len(sorted_lat) // 2
        return sorted_lat[idx]

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(list(self.latencies))
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(list(self.latencies))
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


class PerformanceTracker:
    """
    SCIO Performance Tracker

    Funktionen:
    - Request-Level Tracking
    - Aggregierte Statistiken
    - Zeitraum-basierte Reports
    - SLA-Monitoring
    - Cost Tracking
    """

    def __init__(self):
        self.requests: List[RequestMetrics] = []
        self.active_requests: Dict[str, RequestMetrics] = {}

        # Aggregationen
        self.hourly_metrics: Dict[str, AggregatedMetrics] = {}
        self.daily_metrics: Dict[str, AggregatedMetrics] = {}

        # Per-Model Tracking (using deque for bounded latency storage)
        self.model_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "requests": 0,
            "tokens": 0,
            "cost_cents": 0,
            "errors": 0,
            "latencies": deque(maxlen=1000)
        })

        # SLA Thresholds
        self.sla_latency_ms = 5000  # 5 Sekunden
        self.sla_success_rate = 0.99  # 99%

        # Callbacks
        self.on_sla_violation: List[Callable] = []

        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert den Performance Tracker"""
        try:
            self._initialized = True
            logger.info("Performance Tracker initialisiert")
            return True
        except Exception as e:
            logger.error(f"Performance Tracker Fehler: {e}")
            return False

    @contextmanager
    def track_request(self, request_id: str, model: str, operation: str):
        """
        Context Manager für Request-Tracking

        Usage:
            with tracker.track_request("req_123", "gpt-4", "chat") as metrics:
                # ... request verarbeiten ...
                metrics.tokens_output = 100
        """
        metrics = RequestMetrics(
            request_id=request_id,
            model=model,
            operation=operation,
            start_time=time.time()
        )
        self.active_requests[request_id] = metrics

        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            raise
        finally:
            metrics.end_time = time.time()
            self._record_completed(metrics)
            del self.active_requests[request_id]

    def start_request(self, request_id: str, model: str, operation: str) -> RequestMetrics:
        """Startet Request-Tracking (manuell)"""
        metrics = RequestMetrics(
            request_id=request_id,
            model=model,
            operation=operation,
            start_time=time.time()
        )
        self.active_requests[request_id] = metrics
        return metrics

    def end_request(self, request_id: str,
                    tokens_input: int = 0,
                    tokens_output: int = 0,
                    success: bool = True,
                    error: str = None,
                    cost_cents: float = 0,
                    gpu_seconds: float = 0):
        """Beendet Request-Tracking (manuell)"""
        if request_id not in self.active_requests:
            return

        metrics = self.active_requests[request_id]
        metrics.end_time = time.time()
        metrics.tokens_input = tokens_input
        metrics.tokens_output = tokens_output
        metrics.success = success
        metrics.error = error
        metrics.cost_cents = cost_cents
        metrics.gpu_seconds = gpu_seconds

        self._record_completed(metrics)
        del self.active_requests[request_id]

    def _record_completed(self, metrics: RequestMetrics):
        """Zeichnet abgeschlossenen Request auf"""
        self.requests.append(metrics)

        # Nur letzte 10000 behalten
        if len(self.requests) > 10000:
            self.requests = self.requests[-10000:]

        # Model-Metriken aktualisieren
        mm = self.model_metrics[metrics.model]
        mm["requests"] += 1
        mm["tokens"] += metrics.tokens_input + metrics.tokens_output
        mm["cost_cents"] += metrics.cost_cents
        if not metrics.success:
            mm["errors"] += 1
        mm["latencies"].append(metrics.latency_ms)
        # deque with maxlen handles truncation automatically

        # Aggregationen aktualisieren
        self._update_aggregations(metrics)

        # SLA-Check
        self._check_sla(metrics)

    def _update_aggregations(self, metrics: RequestMetrics):
        """Aktualisiert Aggregationen"""
        now = datetime.now()

        # Stündlich
        hour_key = now.strftime("%Y-%m-%d-%H")
        if hour_key not in self.hourly_metrics:
            self.hourly_metrics[hour_key] = AggregatedMetrics(
                period_start=now.replace(minute=0, second=0, microsecond=0),
                period_end=now.replace(minute=59, second=59, microsecond=999999)
            )

        hourly = self.hourly_metrics[hour_key]
        hourly.request_count += 1
        if metrics.success:
            hourly.success_count += 1
        else:
            hourly.error_count += 1
        hourly.total_tokens_input += metrics.tokens_input
        hourly.total_tokens_output += metrics.tokens_output
        hourly.total_latency_ms += metrics.latency_ms
        hourly.total_cost_cents += metrics.cost_cents
        hourly.total_gpu_seconds += metrics.gpu_seconds
        hourly.latencies.append(metrics.latency_ms)

        # Täglich
        day_key = now.strftime("%Y-%m-%d")
        if day_key not in self.daily_metrics:
            self.daily_metrics[day_key] = AggregatedMetrics(
                period_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
                period_end=now.replace(hour=23, minute=59, second=59, microsecond=999999)
            )

        daily = self.daily_metrics[day_key]
        daily.request_count += 1
        if metrics.success:
            daily.success_count += 1
        else:
            daily.error_count += 1
        daily.total_tokens_input += metrics.tokens_input
        daily.total_tokens_output += metrics.tokens_output
        daily.total_latency_ms += metrics.latency_ms
        daily.total_cost_cents += metrics.cost_cents
        daily.total_gpu_seconds += metrics.gpu_seconds
        daily.latencies.append(metrics.latency_ms)

        # Alte Daten bereinigen
        self._cleanup_old_data()

    def _cleanup_old_data(self):
        """Bereinigt alte Aggregationen"""
        cutoff_hourly = datetime.now() - timedelta(days=7)
        cutoff_daily = datetime.now() - timedelta(days=90)

        self.hourly_metrics = {
            k: v for k, v in self.hourly_metrics.items()
            if v.period_start > cutoff_hourly
        }

        self.daily_metrics = {
            k: v for k, v in self.daily_metrics.items()
            if v.period_start > cutoff_daily
        }

    def _check_sla(self, metrics: RequestMetrics):
        """Prüft SLA-Verletzungen"""
        violations = []

        # Latenz-Check
        if metrics.latency_ms > self.sla_latency_ms:
            violations.append({
                "type": "latency",
                "value": metrics.latency_ms,
                "threshold": self.sla_latency_ms
            })

        # Für Callbacks
        if violations:
            for callback in self.on_sla_violation:
                try:
                    callback(metrics, violations)
                except Exception as e:
                    logger.debug(f"SLA violation callback Fehler: {e}")

    def get_current_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle Statistiken zurück"""
        recent = [r for r in self.requests[-1000:]]

        if not recent:
            return {"status": "no_data"}

        latencies = [r.latency_ms for r in recent]
        success_count = len([r for r in recent if r.success])

        return {
            "request_count": len(recent),
            "success_rate": round(success_count / len(recent), 4),
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "p50_latency_ms": round(sorted(latencies)[len(latencies)//2], 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies)*0.95)], 2),
            "tokens_per_second": round(
                sum(r.tokens_output for r in recent) /
                sum((r.end_time - r.start_time) for r in recent if r.end_time > r.start_time),
                1
            ) if recent else 0,
            "active_requests": len(self.active_requests)
        }

    def get_model_stats(self, model: str = None) -> Dict[str, Any]:
        """Gibt Statistiken pro Model zurück"""
        if model:
            if model not in self.model_metrics:
                return {"error": "Model not found"}
            mm = self.model_metrics[model]
            latencies = list(mm["latencies"])
            return {
                "model": model,
                "requests": mm["requests"],
                "tokens": mm["tokens"],
                "cost_eur": round(mm["cost_cents"] / 100, 2),
                "error_rate": round(mm["errors"] / mm["requests"], 4) if mm["requests"] > 0 else 0,
                "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0
            }

        # Alle Models
        return {
            model: {
                "requests": mm["requests"],
                "tokens": mm["tokens"],
                "cost_eur": round(mm["cost_cents"] / 100, 2)
            }
            for model, mm in self.model_metrics.items()
        }

    def get_hourly_report(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Gibt stündliche Metriken zurück"""
        report = []
        now = datetime.now()

        for i in range(hours):
            hour = now - timedelta(hours=i)
            key = hour.strftime("%Y-%m-%d-%H")

            if key in self.hourly_metrics:
                m = self.hourly_metrics[key]
                report.append({
                    "hour": hour.strftime("%Y-%m-%d %H:00"),
                    "requests": m.request_count,
                    "success_rate": round(m.success_rate, 4),
                    "avg_latency_ms": round(m.avg_latency_ms, 2),
                    "p95_latency_ms": round(m.p95_latency_ms, 2),
                    "tokens": m.total_tokens_input + m.total_tokens_output,
                    "cost_eur": round(m.total_cost_cents / 100, 2)
                })
            else:
                report.append({
                    "hour": hour.strftime("%Y-%m-%d %H:00"),
                    "requests": 0
                })

        return report

    def get_daily_report(self, days: int = 30) -> List[Dict[str, Any]]:
        """Gibt tägliche Metriken zurück"""
        report = []
        now = datetime.now()

        for i in range(days):
            day = now - timedelta(days=i)
            key = day.strftime("%Y-%m-%d")

            if key in self.daily_metrics:
                m = self.daily_metrics[key]
                report.append({
                    "date": key,
                    "requests": m.request_count,
                    "success_rate": round(m.success_rate, 4),
                    "avg_latency_ms": round(m.avg_latency_ms, 2),
                    "p95_latency_ms": round(m.p95_latency_ms, 2),
                    "tokens": m.total_tokens_input + m.total_tokens_output,
                    "cost_eur": round(m.total_cost_cents / 100, 2),
                    "gpu_hours": round(m.total_gpu_seconds / 3600, 2)
                })
            else:
                report.append({
                    "date": key,
                    "requests": 0
                })

        return report

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Gesamtstatistiken zurück"""
        return {
            "total_requests_tracked": len(self.requests),
            "active_requests": len(self.active_requests),
            "models_tracked": len(self.model_metrics),
            "hourly_periods": len(self.hourly_metrics),
            "daily_periods": len(self.daily_metrics),
            "current": self.get_current_stats()
        }


# Singleton
_performance_tracker: Optional[PerformanceTracker] = None

def get_performance_tracker() -> PerformanceTracker:
    """Gibt Singleton-Instanz zurück"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker
