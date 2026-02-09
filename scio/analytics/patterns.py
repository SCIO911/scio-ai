"""
SCIO Pattern Detection

Mustererkennung und Anomalie-Detektion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
import numpy as np

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class AnomalyType(str, Enum):
    """Typen von Anomalien."""
    POINT = "point"           # Einzelner Ausreißer
    CONTEXTUAL = "contextual" # Kontextabhängige Anomalie
    COLLECTIVE = "collective" # Gruppe von Anomalien
    TREND = "trend"          # Trendänderung
    SEASONAL = "seasonal"    # Saisonale Anomalie


class TrendDirection(str, Enum):
    """Trendrichtungen."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class Anomaly:
    """Eine erkannte Anomalie."""

    id: str
    anomaly_type: AnomalyType
    index: int
    value: float
    score: float  # Anomalie-Score (höher = anomaler)
    expected_value: Optional[float] = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)

    @property
    def deviation(self) -> Optional[float]:
        """Abweichung vom erwarteten Wert."""
        if self.expected_value is not None:
            return self.value - self.expected_value
        return None


@dataclass
class Pattern:
    """Ein erkanntes Muster."""

    id: str
    pattern_type: str
    start_index: int
    end_index: int
    confidence: float
    description: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trend:
    """Ein erkannter Trend."""

    direction: TrendDirection
    slope: float
    intercept: float
    r_squared: float
    start_index: int
    end_index: int
    confidence: float


@dataclass
class Cluster:
    """Ein Datencluster."""

    id: int
    centroid: np.ndarray
    members: list[int]  # Indizes
    size: int
    variance: float


class PatternDetector:
    """
    Mustererkennung in Datenreihen.

    Features:
    - Wiederkehrende Muster
    - Zykluserkennung
    - Motif-Erkennung
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def find_repeating_patterns(
        self,
        data: Union[list, np.ndarray],
        min_pattern_length: int = 3,
        min_occurrences: int = 2,
        similarity_threshold: float = 0.9,
    ) -> list[Pattern]:
        """Findet wiederkehrende Muster in einer Zeitreihe."""
        arr = np.array(data, dtype=float)
        n = len(arr)
        patterns = []

        # Normalisiere Daten
        if np.std(arr) > 0:
            arr_norm = (arr - np.mean(arr)) / np.std(arr)
        else:
            return patterns

        # Sliding Window Ansatz
        for length in range(min_pattern_length, min(n // 2, 50)):
            motifs: list[tuple[int, np.ndarray]] = []

            for i in range(n - length + 1):
                window = arr_norm[i:i + length]

                # Vergleiche mit existierenden Motifs
                matched = False
                for j, (_, motif) in enumerate(motifs):
                    sim = self._correlation(window, motif)
                    if sim >= similarity_threshold:
                        matched = True
                        break

                if not matched:
                    # Zähle Vorkommen dieses Fensters
                    occurrences = []
                    for k in range(i, n - length + 1):
                        other = arr_norm[k:k + length]
                        if self._correlation(window, other) >= similarity_threshold:
                            occurrences.append(k)

                    if len(occurrences) >= min_occurrences:
                        motifs.append((i, window.copy()))
                        patterns.append(Pattern(
                            id=generate_id("pat"),
                            pattern_type="repeating",
                            start_index=i,
                            end_index=i + length,
                            confidence=len(occurrences) / (n - length + 1),
                            description=f"Repeating pattern of length {length}",
                            features={
                                "length": length,
                                "occurrences": occurrences,
                                "count": len(occurrences),
                            },
                        ))

        return patterns

    def _correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """Berechnet die Pearson-Korrelation."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(r) else float(r)

    def find_cycles(
        self,
        data: Union[list, np.ndarray],
        min_period: int = 2,
        max_period: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Erkennt zyklische Muster mittels Autokorrelation."""
        arr = np.array(data, dtype=float)
        n = len(arr)
        max_period = max_period or n // 2

        cycles = []

        # Berechne Autokorrelation für verschiedene Lags
        for lag in range(min_period, min(max_period, n // 2)):
            ac = self._autocorrelation(arr, lag)
            if ac > 0.5:  # Signifikante Korrelation
                cycles.append({
                    "period": lag,
                    "strength": ac,
                    "type": "autocorrelation",
                })

        # Sortiere nach Stärke
        cycles.sort(key=lambda x: x["strength"], reverse=True)
        return cycles[:5]  # Top 5

    def _autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """Berechnet die Autokorrelation für einen bestimmten Lag."""
        n = len(data)
        if lag >= n:
            return 0.0

        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return 0.0

        autocov = np.mean((data[:-lag] - mean) * (data[lag:] - mean))
        return float(autocov / var)


class AnomalyDetector:
    """
    Anomalie-Erkennung in Daten.

    Features:
    - Z-Score basiert
    - IQR basiert
    - Isolation Forest Prinzip
    - Lokale Ausreißer-Faktor
    """

    def __init__(
        self,
        method: str = "zscore",
        threshold: float = 3.0,
        window_size: int = 20,
    ):
        self.method = method
        self.threshold = threshold
        self.window_size = window_size

    def detect(
        self,
        data: Union[list, np.ndarray],
        return_scores: bool = False,
    ) -> Union[list[Anomaly], tuple[list[Anomaly], np.ndarray]]:
        """
        Erkennt Anomalien in einer Datenreihe.

        Args:
            data: Die zu analysierende Datenreihe
            return_scores: Ob die Anomalie-Scores zurückgegeben werden sollen
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        if self.method == "zscore":
            scores = self._zscore_scores(arr)
        elif self.method == "iqr":
            scores = self._iqr_scores(arr)
        elif self.method == "mad":
            scores = self._mad_scores(arr)
        elif self.method == "rolling":
            scores = self._rolling_scores(arr)
        else:
            scores = self._zscore_scores(arr)

        anomalies = []
        for i, (value, score) in enumerate(zip(arr, scores)):
            if score >= self.threshold:
                # Berechne erwarteten Wert (ohne Anomalie)
                if i > 0 and i < n - 1:
                    expected = (arr[i-1] + arr[i+1]) / 2
                elif i == 0:
                    expected = arr[1] if n > 1 else value
                else:
                    expected = arr[i-1]

                anomalies.append(Anomaly(
                    id=generate_id("anom"),
                    anomaly_type=AnomalyType.POINT,
                    index=i,
                    value=float(value),
                    score=float(score),
                    expected_value=float(expected),
                ))

        if return_scores:
            return anomalies, scores
        return anomalies

    def _zscore_scores(self, data: np.ndarray) -> np.ndarray:
        """Berechnet Z-Scores."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return np.abs((data - mean) / std)

    def _iqr_scores(self, data: np.ndarray) -> np.ndarray:
        """Berechnet IQR-basierte Scores."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return np.zeros_like(data)

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        scores = np.zeros_like(data)
        below = data < lower
        above = data > upper
        scores[below] = (lower - data[below]) / iqr
        scores[above] = (data[above] - upper) / iqr
        return scores

    def _mad_scores(self, data: np.ndarray) -> np.ndarray:
        """Berechnet MAD-basierte Scores (robust)."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return np.zeros_like(data)
        return np.abs(data - median) / (1.4826 * mad)

    def _rolling_scores(self, data: np.ndarray) -> np.ndarray:
        """Berechnet Scores basierend auf rollenden Statistiken."""
        n = len(data)
        scores = np.zeros(n)
        half_window = self.window_size // 2

        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            window = np.concatenate([data[start:i], data[i+1:end]])

            if len(window) > 0:
                mean = np.mean(window)
                std = np.std(window)
                if std > 0:
                    scores[i] = abs(data[i] - mean) / std

        return scores

    def detect_contextual(
        self,
        data: Union[list, np.ndarray],
        context: Union[list, np.ndarray],
    ) -> list[Anomaly]:
        """
        Erkennt kontextabhängige Anomalien.

        Args:
            data: Die Werte
            context: Kontextvariable (z.B. Zeit, Kategorie)
        """
        arr = np.array(data, dtype=float)
        ctx = np.array(context)
        n = len(arr)
        anomalies = []

        # Gruppiere nach Kontext
        unique_ctx = np.unique(ctx)
        for c in unique_ctx:
            mask = ctx == c
            group_data = arr[mask]
            group_indices = np.where(mask)[0]

            mean = np.mean(group_data)
            std = np.std(group_data)
            if std == 0:
                continue

            for idx, value in zip(group_indices, group_data):
                score = abs(value - mean) / std
                if score >= self.threshold:
                    anomalies.append(Anomaly(
                        id=generate_id("anom"),
                        anomaly_type=AnomalyType.CONTEXTUAL,
                        index=int(idx),
                        value=float(value),
                        score=float(score),
                        expected_value=float(mean),
                        context={"group": c, "group_mean": mean, "group_std": std},
                    ))

        return anomalies


class TrendAnalyzer:
    """
    Trend-Analyse für Zeitreihen.

    Features:
    - Lineare Trendschätzung
    - Trendwechselerkennung
    - Saisonale Bereinigung
    """

    def __init__(self, min_segment_length: int = 5):
        self.min_segment_length = min_segment_length

    def analyze(self, data: Union[list, np.ndarray]) -> Trend:
        """Analysiert den Gesamttrend einer Zeitreihe."""
        arr = np.array(data, dtype=float)
        n = len(arr)
        x = np.arange(n)

        # Lineare Regression
        slope, intercept = self._linear_regression(x, arr)

        # R-Quadrat
        y_pred = slope * x + intercept
        ss_res = np.sum((arr - y_pred) ** 2)
        ss_tot = np.sum((arr - np.mean(arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Bestimme Richtung
        if abs(slope) < 0.001 * np.std(arr):
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Prüfe auf Volatilität
        residuals = arr - y_pred
        if np.std(residuals) > 0.5 * np.std(arr):
            direction = TrendDirection.VOLATILE

        confidence = max(0, min(1, r_squared))

        return Trend(
            direction=direction,
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            start_index=0,
            end_index=n - 1,
            confidence=confidence,
        )

    def _linear_regression(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Einfache lineare Regression."""
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0, float(np.mean(y))

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        return float(slope), float(intercept)

    def find_changepoints(
        self,
        data: Union[list, np.ndarray],
        penalty: float = 1.0,
    ) -> list[int]:
        """
        Findet Trendwechsel-Punkte mittels PELT-ähnlichem Algorithmus.
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        if n < 2 * self.min_segment_length:
            return []

        changepoints = []

        # Sliding Window Varianzvergleich
        for i in range(self.min_segment_length, n - self.min_segment_length):
            before = arr[max(0, i - self.min_segment_length):i]
            after = arr[i:min(n, i + self.min_segment_length)]

            # Vergleiche Mittelwerte
            mean_before = np.mean(before)
            mean_after = np.mean(after)
            pooled_std = np.std(np.concatenate([before, after]))

            if pooled_std > 0:
                change_score = abs(mean_after - mean_before) / pooled_std
                if change_score > penalty:
                    # Prüfe ob nicht zu nah an letztem Changepoint
                    if not changepoints or i - changepoints[-1] >= self.min_segment_length:
                        changepoints.append(i)

        return changepoints

    def segment_trends(
        self,
        data: Union[list, np.ndarray],
    ) -> list[Trend]:
        """Segmentiert Daten und analysiert Trends pro Segment."""
        arr = np.array(data, dtype=float)
        changepoints = self.find_changepoints(arr)

        segments = []
        boundaries = [0] + changepoints + [len(arr)]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment_data = arr[start:end]

            if len(segment_data) >= self.min_segment_length:
                trend = self.analyze(segment_data)
                trend.start_index = start
                trend.end_index = end - 1
                segments.append(trend)

        return segments


class ClusterAnalyzer:
    """
    Cluster-Analyse für mehrdimensionale Daten.

    Features:
    - K-Means Clustering
    - Automatische K-Bestimmung
    - Cluster-Qualitätsmetriken
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-4):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def kmeans(
        self,
        data: Union[list, np.ndarray],
        k: int,
        random_state: Optional[int] = None,
    ) -> list[Cluster]:
        """
        K-Means Clustering.

        Args:
            data: Datenmatrix (n_samples x n_features)
            k: Anzahl der Cluster
            random_state: Seed für Reproduzierbarkeit
        """
        arr = np.array(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_samples, n_features = arr.shape

        if random_state is not None:
            np.random.seed(random_state)

        # Initialisiere Zentroide (k-means++)
        centroids = self._kmeans_plusplus_init(arr, k)

        for iteration in range(self.max_iterations):
            # Weise Punkte zu nächstem Zentroid zu
            labels = np.argmin(
                np.linalg.norm(arr[:, np.newaxis] - centroids, axis=2),
                axis=1
            )

            # Berechne neue Zentroide
            new_centroids = np.array([
                arr[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i]
                for i in range(k)
            ])

            # Prüfe Konvergenz
            if np.max(np.abs(new_centroids - centroids)) < self.tolerance:
                break

            centroids = new_centroids

        # Erstelle Cluster-Objekte
        clusters = []
        for i in range(k):
            mask = labels == i
            members = list(np.where(mask)[0])

            if members:
                cluster_data = arr[mask]
                variance = float(np.mean(np.var(cluster_data, axis=0)))
            else:
                variance = 0.0

            clusters.append(Cluster(
                id=i,
                centroid=centroids[i],
                members=members,
                size=len(members),
                variance=variance,
            ))

        return clusters

    def _kmeans_plusplus_init(self, data: np.ndarray, k: int) -> np.ndarray:
        """K-Means++ Initialisierung."""
        n_samples = len(data)
        centroids = []

        # Erster Zentroid zufällig
        centroids.append(data[np.random.randint(n_samples)])

        for _ in range(1, k):
            # Berechne Distanzen zum nächsten Zentroid
            distances = np.min([
                np.linalg.norm(data - c, axis=1) ** 2
                for c in centroids
            ], axis=0)

            # Wähle nächsten Zentroid proportional zur Distanz
            probs = distances / np.sum(distances)
            idx = np.random.choice(n_samples, p=probs)
            centroids.append(data[idx])

        return np.array(centroids)

    def find_optimal_k(
        self,
        data: Union[list, np.ndarray],
        max_k: int = 10,
        method: str = "elbow",
    ) -> int:
        """
        Findet die optimale Anzahl von Clustern.

        Args:
            data: Datenmatrix
            max_k: Maximale Anzahl zu testender Cluster
            method: "elbow" oder "silhouette"
        """
        arr = np.array(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n = len(arr)
        max_k = min(max_k, n - 1)

        if max_k < 2:
            return 1

        inertias = []
        for k in range(1, max_k + 1):
            clusters = self.kmeans(arr, k, random_state=42)
            inertia = sum(c.variance * c.size for c in clusters)
            inertias.append(inertia)

        if method == "elbow":
            # Elbow-Methode
            diffs = np.diff(inertias)
            diff2 = np.diff(diffs)

            if len(diff2) > 0:
                optimal_k = np.argmax(diff2) + 2
            else:
                optimal_k = 2
        else:
            optimal_k = 2

        return max(2, min(optimal_k, max_k))

    def silhouette_score(
        self,
        data: Union[list, np.ndarray],
        labels: Union[list, np.ndarray],
    ) -> float:
        """Berechnet den Silhouette-Score."""
        arr = np.array(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        labels = np.array(labels)
        n = len(arr)
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            return 0.0

        silhouettes = []
        for i in range(n):
            # a(i): Durchschnittliche Distanz zu Punkten im selben Cluster
            same_cluster = labels == labels[i]
            same_cluster[i] = False
            if np.sum(same_cluster) > 0:
                a = np.mean(np.linalg.norm(arr[same_cluster] - arr[i], axis=1))
            else:
                a = 0

            # b(i): Minimale durchschnittliche Distanz zu anderem Cluster
            b = float('inf')
            for label in unique_labels:
                if label != labels[i]:
                    other_cluster = labels == label
                    if np.sum(other_cluster) > 0:
                        avg_dist = np.mean(np.linalg.norm(arr[other_cluster] - arr[i], axis=1))
                        b = min(b, avg_dist)

            if b == float('inf'):
                b = 0

            # Silhouette für Punkt i
            if max(a, b) > 0:
                silhouettes.append((b - a) / max(a, b))
            else:
                silhouettes.append(0)

        return float(np.mean(silhouettes))
