#!/usr/bin/env python3
"""
SCIO - Prometheus Metrics Exporter
Exportiert Metriken im Prometheus-Format für Monitoring
"""

import time
import threading
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class Counter:
    """Prometheus Counter Metrik"""
    name: str
    help: str
    labels: List[str] = field(default_factory=list)
    values: Dict[tuple, float] = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0, **labels):
        """Inkrementiert Counter"""
        label_values = tuple(labels.get(l, "") for l in self.labels)
        with self._lock:
            self.values[label_values] += amount

    def get(self, **labels) -> float:
        """Holt aktuellen Wert"""
        label_values = tuple(labels.get(l, "") for l in self.labels)
        return self.values.get(label_values, 0)


@dataclass
class Gauge:
    """Prometheus Gauge Metrik"""
    name: str
    help: str
    labels: List[str] = field(default_factory=list)
    values: Dict[tuple, float] = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, **labels):
        """Setzt Gauge-Wert"""
        label_values = tuple(labels.get(l, "") for l in self.labels)
        with self._lock:
            self.values[label_values] = value

    def inc(self, amount: float = 1.0, **labels):
        """Inkrementiert Gauge"""
        label_values = tuple(labels.get(l, "") for l in self.labels)
        with self._lock:
            self.values[label_values] += amount

    def dec(self, amount: float = 1.0, **labels):
        """Dekrementiert Gauge"""
        label_values = tuple(labels.get(l, "") for l in self.labels)
        with self._lock:
            self.values[label_values] -= amount

    def get(self, **labels) -> float:
        """Holt aktuellen Wert"""
        label_values = tuple(labels.get(l, "") for l in self.labels)
        return self.values.get(label_values, 0)


@dataclass
class Histogram:
    """Prometheus Histogram Metrik"""
    name: str
    help: str
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
    bucket_counts: Dict[tuple, Dict[float, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    sums: Dict[tuple, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[tuple, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, **labels):
        """Beobachtet einen Wert"""
        label_values = tuple(labels.get(l, "") for l in self.labels)
        with self._lock:
            self.sums[label_values] += value
            self.counts[label_values] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[label_values][bucket] += 1


class MetricsRegistry:
    """
    Zentrale Registry für alle Prometheus-Metriken

    Usage:
        registry = MetricsRegistry()

        # Counter erstellen
        requests_total = registry.counter(
            "scio_requests_total",
            "Total number of requests",
            ["method", "endpoint"]
        )
        requests_total.inc(method="POST", endpoint="/api/jobs")

        # Gauge erstellen
        active_jobs = registry.gauge("scio_active_jobs", "Number of active jobs")
        active_jobs.set(5)

        # Histogram erstellen
        request_duration = registry.histogram(
            "scio_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint"]
        )
        request_duration.observe(0.25, endpoint="/api/health")

        # Metriken exportieren
        output = registry.export()
    """

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str, labels: List[str] = None) -> Counter:
        """Erstellt oder holt Counter"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, help, labels or [])
            return self._counters[name]

    def gauge(self, name: str, help: str, labels: List[str] = None) -> Gauge:
        """Erstellt oder holt Gauge"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, help, labels or [])
            return self._gauges[name]

    def histogram(self, name: str, help: str, labels: List[str] = None,
                  buckets: List[float] = None) -> Histogram:
        """Erstellt oder holt Histogram"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name, help, labels or [],
                    buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
                )
            return self._histograms[name]

    def _format_labels(self, label_names: List[str], label_values: tuple) -> str:
        """Formatiert Labels für Prometheus-Output"""
        if not label_names:
            return ""
        pairs = [f'{name}="{value}"' for name, value in zip(label_names, label_values)]
        return "{" + ",".join(pairs) + "}"

    def export(self) -> str:
        """
        Exportiert alle Metriken im Prometheus-Text-Format

        Returns:
            String im Prometheus-Format
        """
        lines = []

        # Counters
        for counter in self._counters.values():
            lines.append(f"# HELP {counter.name} {counter.help}")
            lines.append(f"# TYPE {counter.name} counter")
            for label_values, value in counter.values.items():
                labels = self._format_labels(counter.labels, label_values)
                lines.append(f"{counter.name}{labels} {value}")

        # Gauges
        for gauge in self._gauges.values():
            lines.append(f"# HELP {gauge.name} {gauge.help}")
            lines.append(f"# TYPE {gauge.name} gauge")
            for label_values, value in gauge.values.items():
                labels = self._format_labels(gauge.labels, label_values)
                lines.append(f"{gauge.name}{labels} {value}")

        # Histograms
        for hist in self._histograms.values():
            lines.append(f"# HELP {hist.name} {hist.help}")
            lines.append(f"# TYPE {hist.name} histogram")
            for label_values in hist.counts.keys():
                labels = self._format_labels(hist.labels, label_values)
                base_labels = labels.rstrip("}") if labels else "{"

                # Buckets
                cumulative = 0
                for bucket in sorted(hist.buckets):
                    cumulative += hist.bucket_counts[label_values].get(bucket, 0)
                    if base_labels == "{":
                        bucket_labels = f'{{le="{bucket}"}}'
                    else:
                        bucket_labels = f'{base_labels},le="{bucket}"}}'
                    lines.append(f"{hist.name}_bucket{bucket_labels} {cumulative}")

                # +Inf bucket
                if base_labels == "{":
                    inf_labels = '{le="+Inf"}'
                else:
                    inf_labels = f'{base_labels},le="+Inf"}}'
                lines.append(f"{hist.name}_bucket{inf_labels} {hist.counts[label_values]}")

                # Sum and count
                lines.append(f"{hist.name}_sum{labels} {hist.sums[label_values]}")
                lines.append(f"{hist.name}_count{labels} {hist.counts[label_values]}")

        return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# SCIO METRICS
# ═══════════════════════════════════════════════════════════════════════════

class SCIOMetrics:
    """
    SCIO-spezifische Metriken für Prometheus

    Enthält alle relevanten Metriken für:
    - Job Queue
    - Hardware (GPU, CPU, RAM)
    - API Requests
    - Decision Engine
    - Learning Module
    - Orchestrator
    """

    def __init__(self):
        self.registry = MetricsRegistry()
        self._setup_metrics()
        self._start_time = time.time()

    def _setup_metrics(self):
        """Initialisiert alle SCIO-Metriken"""

        # ─── System Metrics ───────────────────────────────────────────
        self.uptime = self.registry.gauge(
            "scio_uptime_seconds",
            "Time since SCIO started"
        )

        # ─── Job Queue Metrics ────────────────────────────────────────
        self.jobs_queued = self.registry.gauge(
            "scio_jobs_queued",
            "Number of jobs in queue"
        )
        self.jobs_active = self.registry.gauge(
            "scio_jobs_active",
            "Number of currently active jobs"
        )
        self.jobs_total = self.registry.counter(
            "scio_jobs_total",
            "Total number of jobs processed",
            ["status", "type"]
        )
        self.job_duration = self.registry.histogram(
            "scio_job_duration_seconds",
            "Job processing duration",
            ["type"],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600]
        )

        # ─── Hardware Metrics ─────────────────────────────────────────
        self.gpu_vram_used = self.registry.gauge(
            "scio_gpu_vram_used_bytes",
            "GPU VRAM used in bytes",
            ["gpu_index", "gpu_name"]
        )
        self.gpu_vram_total = self.registry.gauge(
            "scio_gpu_vram_total_bytes",
            "GPU VRAM total in bytes",
            ["gpu_index", "gpu_name"]
        )
        self.gpu_utilization = self.registry.gauge(
            "scio_gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_index", "gpu_name"]
        )
        self.gpu_temperature = self.registry.gauge(
            "scio_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["gpu_index", "gpu_name"]
        )
        self.cpu_usage = self.registry.gauge(
            "scio_cpu_usage_percent",
            "CPU usage percentage"
        )
        self.ram_used = self.registry.gauge(
            "scio_ram_used_bytes",
            "RAM used in bytes"
        )
        self.ram_total = self.registry.gauge(
            "scio_ram_total_bytes",
            "RAM total in bytes"
        )

        # ─── API Metrics ──────────────────────────────────────────────
        self.requests_total = self.registry.counter(
            "scio_requests_total",
            "Total number of API requests",
            ["method", "endpoint", "status"]
        )
        self.request_duration = self.registry.histogram(
            "scio_request_duration_seconds",
            "API request duration",
            ["method", "endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
        )
        self.request_errors = self.registry.counter(
            "scio_request_errors_total",
            "Total number of request errors",
            ["method", "endpoint", "error_type"]
        )

        # ─── Decision Engine Metrics ──────────────────────────────────
        self.decisions_total = self.registry.counter(
            "scio_decisions_total",
            "Total number of decisions made",
            ["tree", "action"]
        )
        self.decision_confidence = self.registry.histogram(
            "scio_decision_confidence",
            "Decision confidence distribution",
            ["tree"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self.decision_duration = self.registry.histogram(
            "scio_decision_duration_seconds",
            "Decision processing time",
            ["tree"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        )

        # ─── Learning Metrics ─────────────────────────────────────────
        self.learning_observations = self.registry.counter(
            "scio_learning_observations_total",
            "Total learning observations recorded"
        )
        self.learning_patterns = self.registry.gauge(
            "scio_learning_patterns",
            "Number of learned patterns"
        )
        self.rl_rewards = self.registry.histogram(
            "scio_rl_rewards",
            "Reinforcement learning rewards",
            ["action_space"],
            buckets=[-1, -0.5, 0, 0.5, 1]
        )

        # ─── Orchestrator Metrics ─────────────────────────────────────
        self.modules_registered = self.registry.gauge(
            "scio_modules_registered",
            "Number of registered modules"
        )
        self.modules_healthy = self.registry.gauge(
            "scio_modules_healthy",
            "Number of healthy modules"
        )
        self.events_processed = self.registry.counter(
            "scio_events_processed_total",
            "Total events processed by event bus",
            ["event_type"]
        )
        self.workflows_total = self.registry.counter(
            "scio_workflows_total",
            "Total workflows executed",
            ["status"]
        )

        # ─── Rate Limiting Metrics ────────────────────────────────────
        self.rate_limit_hits = self.registry.counter(
            "scio_rate_limit_hits_total",
            "Number of rate limit hits"
        )

    def update_system_metrics(self):
        """Aktualisiert System-Metriken"""
        self.uptime.set(time.time() - self._start_time)

    def update_from_hardware_monitor(self):
        """Aktualisiert Hardware-Metriken vom Hardware Monitor"""
        try:
            from backend.services.hardware_monitor import get_hardware_monitor
            monitor = get_hardware_monitor()
            status = monitor.get_status()

            if status:
                self.cpu_usage.set(status.cpu_percent)
                self.ram_used.set(status.ram_used * 1024 * 1024 * 1024)  # GB to bytes
                self.ram_total.set(status.ram_total * 1024 * 1024 * 1024)

                for gpu in status.gpus:
                    labels = {"gpu_index": str(gpu.index), "gpu_name": gpu.name}
                    self.gpu_vram_used.set(gpu.memory_used * 1024 * 1024, **labels)
                    self.gpu_vram_total.set(gpu.memory_total * 1024 * 1024, **labels)
                    self.gpu_utilization.set(gpu.utilization, **labels)
                    self.gpu_temperature.set(gpu.temperature, **labels)
        except Exception:
            pass

    def update_from_job_queue(self):
        """Aktualisiert Job Queue Metriken"""
        try:
            from backend.services.job_queue import get_job_queue
            queue = get_job_queue()
            self.jobs_queued.set(queue.queue_size)
            self.jobs_active.set(queue.active_job_count)
        except Exception:
            pass

    def update_from_orchestrator(self):
        """Aktualisiert Orchestrator Metriken"""
        try:
            from backend.orchestration import get_orchestrator
            orch = get_orchestrator()
            health = orch.get_health()

            self.modules_registered.set(health.get("total_modules", 0))
            self.modules_healthy.set(health.get("healthy_modules", 0))
        except Exception:
            pass

    def update_all(self):
        """Aktualisiert alle Metriken"""
        self.update_system_metrics()
        self.update_from_hardware_monitor()
        self.update_from_job_queue()
        self.update_from_orchestrator()

    def export(self) -> str:
        """Exportiert alle Metriken im Prometheus-Format"""
        self.update_all()
        return self.registry.export()


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

_metrics: SCIOMetrics = None


def get_metrics() -> SCIOMetrics:
    """Gibt globale Metrics-Instanz zurück"""
    global _metrics
    if _metrics is None:
        _metrics = SCIOMetrics()
    return _metrics
