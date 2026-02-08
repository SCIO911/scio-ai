#!/usr/bin/env python3
"""
SCIO - Drift Detector
Erkennt Model Drift und Performance-Degradation
"""

import math
import logging
import statistics
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Typen von Drift"""
    DATA_DRIFT = 'data_drift'           # Veränderung in Input-Daten
    CONCEPT_DRIFT = 'concept_drift'     # Veränderung der Beziehung Input->Output
    PERFORMANCE_DRIFT = 'performance_drift'  # Verschlechterung der Metriken


class DriftSeverity(str, Enum):
    """Schweregrad des Drifts"""
    NONE = 'none'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


@dataclass
class DriftAlert:
    """Ein Drift-Alarm"""
    id: str
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    current_value: float
    baseline_value: float
    deviation: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class MetricWindow:
    """Sliding Window für Metriken"""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Baseline-Statistiken
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    baseline_min: Optional[float] = None
    baseline_max: Optional[float] = None

    def add(self, value: float):
        self.values.append(value)
        self.timestamps.append(datetime.now())

    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    def std(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def recent_mean(self, n: int = 100) -> float:
        recent = list(self.values)[-n:]
        return statistics.mean(recent) if recent else 0.0

    def set_baseline(self):
        """Setzt aktuelle Statistiken als Baseline"""
        if len(self.values) >= 30:
            self.baseline_mean = self.mean()
            self.baseline_std = self.std()
            self.baseline_min = min(self.values)
            self.baseline_max = max(self.values)


class DriftDetector:
    """
    SCIO Drift Detector

    Erkennt verschiedene Arten von Drift:
    - Data Drift: Statistische Veränderungen in den Eingabedaten
    - Concept Drift: Veränderung der Beziehung zwischen Input und Output
    - Performance Drift: Verschlechterung von Modell-Metriken

    Methoden:
    - Z-Score basierte Anomalie-Erkennung
    - Sliding Window Statistiken
    - Kolmogorov-Smirnov ähnlicher Test (vereinfacht)
    """

    def __init__(self):
        self.metrics: Dict[str, MetricWindow] = {}
        self.alerts: List[DriftAlert] = []
        self.alert_callbacks: List[Callable] = []

        # Thresholds
        self.z_score_threshold = 3.0  # Standard-Abweichungen
        self.performance_drop_threshold = 0.15  # 15% Verschlechterung
        self.window_comparison_threshold = 0.2  # 20% Unterschied

        # Baseline-Periode
        self.baseline_samples = 500

        self._initialized = False
        self._alert_counter = 0

    def initialize(self) -> bool:
        """Initialisiert den Drift Detector"""
        try:
            self._setup_default_metrics()
            self._initialized = True
            logger.info("Drift Detector initialisiert")
            return True
        except Exception as e:
            logger.error(f"Drift Detector Fehler: {e}")
            return False

    def _setup_default_metrics(self):
        """Erstellt Standard-Metriken"""
        default_metrics = [
            "response_latency_ms",
            "tokens_per_second",
            "success_rate",
            "error_rate",
            "input_length",
            "output_length",
            "confidence_score",
            "memory_usage_mb",
            "gpu_utilization"
        ]

        for name in default_metrics:
            self.metrics[name] = MetricWindow(name=name)

    def record(self, metric_name: str, value: float):
        """
        Zeichnet einen Metrik-Wert auf und prüft auf Drift

        Args:
            metric_name: Name der Metrik
            value: Wert
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MetricWindow(name=metric_name)

        window = self.metrics[metric_name]
        window.add(value)

        # Baseline setzen wenn genug Daten
        if len(window.values) == self.baseline_samples and window.baseline_mean is None:
            window.set_baseline()
            return

        # Drift-Check nur wenn Baseline existiert
        if window.baseline_mean is not None:
            self._check_drift(metric_name, value)

    def _check_drift(self, metric_name: str, value: float):
        """Prüft auf verschiedene Drift-Arten"""
        window = self.metrics[metric_name]

        # 1. Z-Score Check (Anomalie-Erkennung)
        if window.baseline_std > 0:
            z_score = abs(value - window.baseline_mean) / window.baseline_std

            if z_score > self.z_score_threshold * 2:
                self._create_alert(
                    DriftType.DATA_DRIFT,
                    DriftSeverity.HIGH,
                    metric_name,
                    value,
                    window.baseline_mean,
                    z_score,
                    f"Extreme anomaly detected: Z-score {z_score:.2f}"
                )
            elif z_score > self.z_score_threshold:
                self._create_alert(
                    DriftType.DATA_DRIFT,
                    DriftSeverity.MEDIUM,
                    metric_name,
                    value,
                    window.baseline_mean,
                    z_score,
                    f"Anomaly detected: Z-score {z_score:.2f}"
                )

        # 2. Window Comparison (Trend-Erkennung)
        if len(window.values) >= 200:
            recent_mean = window.recent_mean(100)
            baseline_mean = window.baseline_mean

            if baseline_mean > 0:
                deviation = (recent_mean - baseline_mean) / baseline_mean

                # Für positive Metriken (success_rate) ist negative Abweichung schlecht
                # Für negative Metriken (error_rate, latency) ist positive Abweichung schlecht
                is_degrading = False
                if metric_name in ["success_rate", "confidence_score", "tokens_per_second"]:
                    is_degrading = deviation < -self.performance_drop_threshold
                elif metric_name in ["error_rate", "response_latency_ms", "memory_usage_mb"]:
                    is_degrading = deviation > self.performance_drop_threshold

                if is_degrading:
                    severity = DriftSeverity.HIGH if abs(deviation) > 0.3 else DriftSeverity.MEDIUM
                    self._create_alert(
                        DriftType.PERFORMANCE_DRIFT,
                        severity,
                        metric_name,
                        recent_mean,
                        baseline_mean,
                        deviation,
                        f"Performance degradation: {abs(deviation)*100:.1f}% change"
                    )

    def _create_alert(self,
                      drift_type: DriftType,
                      severity: DriftSeverity,
                      metric_name: str,
                      current: float,
                      baseline: float,
                      deviation: float,
                      message: str):
        """Erstellt einen Drift-Alarm"""
        # Prüfe auf Duplikate (gleicher Typ + Metrik in letzter Minute)
        recent_cutoff = datetime.now() - timedelta(minutes=1)
        for alert in self.alerts[-10:]:
            if (alert.drift_type == drift_type and
                alert.metric_name == metric_name and
                alert.timestamp > recent_cutoff):
                return  # Duplikat

        self._alert_counter += 1
        alert = DriftAlert(
            id=f"drift_{self._alert_counter}",
            drift_type=drift_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current,
            baseline_value=baseline,
            deviation=deviation,
            message=message
        )

        self.alerts.append(alert)

        # Nur letzte 100 Alerts behalten
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        # Callbacks aufrufen
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.debug(f"Drift alert callback Fehler: {e}")

        logger.warning(f"DRIFT {severity.value.upper()}: {metric_name} - {message}")

    def add_callback(self, callback: Callable):
        """Registriert Callback für Alerts"""
        self.alert_callbacks.append(callback)

    def reset_baseline(self, metric_name: str = None):
        """Setzt Baseline zurück"""
        if metric_name:
            if metric_name in self.metrics:
                window = self.metrics[metric_name]
                window.set_baseline()
        else:
            for window in self.metrics.values():
                if len(window.values) >= 30:
                    window.set_baseline()

    def get_alerts(self,
                   severity: DriftSeverity = None,
                   drift_type: DriftType = None,
                   acknowledged: bool = None) -> List[DriftAlert]:
        """Gibt gefilterte Alerts zurück"""
        filtered = self.alerts

        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        if drift_type:
            filtered = [a for a in filtered if a.drift_type == drift_type]

        if acknowledged is not None:
            filtered = [a for a in filtered if a.acknowledged == acknowledged]

        return filtered

    def acknowledge_alert(self, alert_id: str):
        """Markiert Alert als bestätigt"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break

    def get_metric_health(self, metric_name: str) -> Dict[str, Any]:
        """Gibt Gesundheitsstatus einer Metrik zurück"""
        if metric_name not in self.metrics:
            return {"error": "Metric not found"}

        window = self.metrics[metric_name]

        if not window.values:
            return {"status": "no_data"}

        current_mean = window.recent_mean(50)
        current_std = statistics.stdev(list(window.values)[-50:]) if len(window.values) >= 50 else 0

        health = "healthy"
        deviation = 0

        if window.baseline_mean is not None and window.baseline_mean > 0:
            deviation = (current_mean - window.baseline_mean) / window.baseline_mean

            if abs(deviation) > 0.3:
                health = "critical"
            elif abs(deviation) > 0.15:
                health = "degraded"
            elif abs(deviation) > 0.05:
                health = "warning"

        return {
            "metric": metric_name,
            "health": health,
            "current_mean": round(current_mean, 4),
            "current_std": round(current_std, 4),
            "baseline_mean": round(window.baseline_mean, 4) if window.baseline_mean else None,
            "baseline_std": round(window.baseline_std, 4) if window.baseline_std else None,
            "deviation_percent": round(deviation * 100, 2),
            "samples": len(window.values)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Gesamtstatistiken zurück"""
        severity_counts = {}
        for alert in self.alerts:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1

        metrics_health = {}
        for name in list(self.metrics.keys())[:10]:
            health_info = self.get_metric_health(name)
            metrics_health[name] = health_info.get("health", health_info.get("status", "unknown"))

        return {
            "metrics_tracked": len(self.metrics),
            "total_alerts": len(self.alerts),
            "unacknowledged_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "alerts_by_severity": severity_counts,
            "metrics_health": metrics_health
        }


# Singleton
_drift_detector: Optional[DriftDetector] = None

def get_drift_detector() -> DriftDetector:
    """Gibt Singleton-Instanz zurück"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector
