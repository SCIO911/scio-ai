#!/usr/bin/env python3
"""
SCIO - Hardware History Service
Speichert und aggregiert Hardware-Metriken über Zeit
"""

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Deque

from backend.services.hardware_monitor import get_hardware_monitor, SystemStatus

logger = logging.getLogger(__name__)


@dataclass
class HistoryDataPoint:
    """Ein Datenpunkt in der Hardware-Historie"""
    timestamp: float
    gpu_utilization: float  # Durchschnitt über alle GPUs
    gpu_memory_percent: float
    gpu_temperature: float
    gpu_power_watts: float
    cpu_percent: float
    ram_percent: float
    active_jobs: int

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'gpu': {
                'utilization': round(self.gpu_utilization, 1),
                'memory_percent': round(self.gpu_memory_percent, 1),
                'temperature': round(self.gpu_temperature, 1),
                'power_watts': round(self.gpu_power_watts, 1),
            },
            'cpu_percent': round(self.cpu_percent, 1),
            'ram_percent': round(self.ram_percent, 1),
            'active_jobs': self.active_jobs,
        }


class HardwareHistory:
    """
    Hardware History Service

    Sammelt periodisch Hardware-Metriken und speichert sie
    für historische Auswertungen.

    Speichert:
    - Letzte 24h in Minuten-Auflösung (1440 Punkte)
    - Letzte 7 Tage in Stunden-Auflösung (168 Punkte)
    - Letzte 30 Tage in Tages-Auflösung (30 Punkte)
    """

    def __init__(self, sample_interval: float = 60.0):
        self.sample_interval = sample_interval  # Default: 1 Minute

        # History-Buffer (Thread-safe via deque mit maxlen)
        self._minute_history: Deque[HistoryDataPoint] = deque(maxlen=1440)  # 24h
        self._hour_history: Deque[HistoryDataPoint] = deque(maxlen=168)     # 7 Tage
        self._day_history: Deque[HistoryDataPoint] = deque(maxlen=30)       # 30 Tage

        # Aggregation buffer für stündliche/tägliche Werte
        self._current_hour_samples: List[HistoryDataPoint] = []
        self._current_day_samples: List[HistoryDataPoint] = []
        self._last_hour = -1
        self._last_day = -1

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Startet die History-Aufzeichnung"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        logger.info("Hardware History Service gestartet")

    def stop(self):
        """Stoppt die History-Aufzeichnung"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Hardware History Service gestoppt")

    def _collection_loop(self):
        """Haupt-Sammel-Loop"""
        while self._running:
            try:
                self._collect_sample()
            except Exception as e:
                logger.debug(f"Hardware History Sampling Fehler: {e}")

            time.sleep(self.sample_interval)

    def _collect_sample(self):
        """Sammelt einen Sample von aktuellen Hardware-Werten"""
        monitor = get_hardware_monitor()
        status = monitor.get_status()

        # Berechne Durchschnittswerte für GPUs
        gpu_util = 0.0
        gpu_mem = 0.0
        gpu_temp = 0.0
        gpu_power = 0.0

        if status.gpus:
            gpu_util = sum(g.gpu_utilization for g in status.gpus) / len(status.gpus)
            gpu_mem = sum(g.vram_usage_percent for g in status.gpus) / len(status.gpus)
            gpu_temp = sum(g.temperature for g in status.gpus) / len(status.gpus)
            gpu_power = sum(g.power_usage_watts for g in status.gpus)

        data_point = HistoryDataPoint(
            timestamp=status.timestamp,
            gpu_utilization=gpu_util,
            gpu_memory_percent=gpu_mem,
            gpu_temperature=gpu_temp,
            gpu_power_watts=gpu_power,
            cpu_percent=status.cpu.usage_percent if status.cpu else 0.0,
            ram_percent=status.ram.usage_percent if status.ram else 0.0,
            active_jobs=status.active_jobs,
        )

        with self._lock:
            # Immer in Minuten-History speichern
            self._minute_history.append(data_point)

            # Stündliche Aggregation
            current_hour = datetime.fromtimestamp(status.timestamp).hour
            if current_hour != self._last_hour and self._last_hour != -1:
                self._aggregate_to_hour()
            self._current_hour_samples.append(data_point)
            self._last_hour = current_hour

            # Tägliche Aggregation
            current_day = datetime.fromtimestamp(status.timestamp).day
            if current_day != self._last_day and self._last_day != -1:
                self._aggregate_to_day()
            self._current_day_samples.append(data_point)
            self._last_day = current_day

    def _aggregate_to_hour(self):
        """Aggregiert Samples zu einem Stunden-Durchschnitt"""
        if not self._current_hour_samples:
            return

        samples = self._current_hour_samples
        avg_point = HistoryDataPoint(
            timestamp=samples[-1].timestamp,
            gpu_utilization=sum(s.gpu_utilization for s in samples) / len(samples),
            gpu_memory_percent=sum(s.gpu_memory_percent for s in samples) / len(samples),
            gpu_temperature=sum(s.gpu_temperature for s in samples) / len(samples),
            gpu_power_watts=sum(s.gpu_power_watts for s in samples) / len(samples),
            cpu_percent=sum(s.cpu_percent for s in samples) / len(samples),
            ram_percent=sum(s.ram_percent for s in samples) / len(samples),
            active_jobs=max(s.active_jobs for s in samples),
        )
        self._hour_history.append(avg_point)
        self._current_hour_samples = []

    def _aggregate_to_day(self):
        """Aggregiert Samples zu einem Tages-Durchschnitt"""
        if not self._current_day_samples:
            return

        samples = self._current_day_samples
        avg_point = HistoryDataPoint(
            timestamp=samples[-1].timestamp,
            gpu_utilization=sum(s.gpu_utilization for s in samples) / len(samples),
            gpu_memory_percent=sum(s.gpu_memory_percent for s in samples) / len(samples),
            gpu_temperature=sum(s.gpu_temperature for s in samples) / len(samples),
            gpu_power_watts=sum(s.gpu_power_watts for s in samples) / len(samples),
            cpu_percent=sum(s.cpu_percent for s in samples) / len(samples),
            ram_percent=sum(s.ram_percent for s in samples) / len(samples),
            active_jobs=max(s.active_jobs for s in samples),
        )
        self._day_history.append(avg_point)
        self._current_day_samples = []

    def get_history(self, hours: int = 24, resolution: str = 'minute') -> List[dict]:
        """
        Gibt Hardware-Historie zurück

        Args:
            hours: Anzahl Stunden in die Vergangenheit
            resolution: 'minute', 'hour', oder 'day'

        Returns:
            Liste von Datenpunkten
        """
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            if resolution == 'minute':
                history = list(self._minute_history)
            elif resolution == 'hour':
                history = list(self._hour_history)
            elif resolution == 'day':
                history = list(self._day_history)
            else:
                history = list(self._minute_history)

        # Filtere nach Zeit
        filtered = [
            dp.to_dict() for dp in history
            if dp.timestamp >= cutoff_time
        ]

        return filtered

    def get_current_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle Statistiken zurück"""
        with self._lock:
            return {
                'minute_samples': len(self._minute_history),
                'hour_samples': len(self._hour_history),
                'day_samples': len(self._day_history),
                'collecting': self._running,
            }

    def get_peak_values(self, hours: int = 24) -> Dict[str, Any]:
        """Gibt Spitzenwerte der letzten Stunden zurück"""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            samples = [
                dp for dp in self._minute_history
                if dp.timestamp >= cutoff_time
            ]

        if not samples:
            return {'error': 'No data available'}

        return {
            'peak_gpu_utilization': max(s.gpu_utilization for s in samples),
            'peak_gpu_memory': max(s.gpu_memory_percent for s in samples),
            'peak_gpu_temperature': max(s.gpu_temperature for s in samples),
            'peak_gpu_power': max(s.gpu_power_watts for s in samples),
            'peak_cpu': max(s.cpu_percent for s in samples),
            'peak_ram': max(s.ram_percent for s in samples),
            'avg_gpu_utilization': sum(s.gpu_utilization for s in samples) / len(samples),
            'avg_cpu': sum(s.cpu_percent for s in samples) / len(samples),
            'total_samples': len(samples),
            'hours_covered': hours,
        }


# Singleton
_hardware_history: Optional[HardwareHistory] = None


def get_hardware_history() -> HardwareHistory:
    """Gibt Singleton-Instanz zurück"""
    global _hardware_history
    if _hardware_history is None:
        _hardware_history = HardwareHistory()
        _hardware_history.start()  # Automatisch starten
    return _hardware_history
