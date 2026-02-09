"""
SCIO Time Series Analysis

Fortgeschrittene Zeitreihenanalyse und Prognose.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
import numpy as np

from scio.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SeasonalComponents:
    """Komponenten einer saisonalen Zerlegung."""

    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    period: int


@dataclass
class Forecast:
    """Prognoseergebnis."""

    values: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    horizon: int
    method: str


class TimeSeriesAnalyzer:
    """
    Analyse von Zeitreihen.

    Features:
    - Stationaritätstests
    - Autokorrelationsanalyse
    - Spektralanalyse
    """

    def __init__(self):
        logger.info("TimeSeriesAnalyzer initialized")

    def is_stationary(
        self,
        data: Union[list, np.ndarray],
        window: int = 10,
    ) -> dict[str, Any]:
        """
        Prüft auf Stationarität mittels rollender Statistiken.
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        if n < window * 2:
            return {"is_stationary": None, "reason": "Insufficient data"}

        # Berechne rollende Mittelwerte und Varianzen
        rolling_mean = np.array([
            np.mean(arr[i:i+window]) for i in range(n - window + 1)
        ])
        rolling_var = np.array([
            np.var(arr[i:i+window]) for i in range(n - window + 1)
        ])

        # Prüfe auf Trends in Mittelwert und Varianz
        mean_trend = np.std(rolling_mean) / np.mean(np.abs(rolling_mean) + 1e-10)
        var_trend = np.std(rolling_var) / np.mean(np.abs(rolling_var) + 1e-10)

        is_stationary = mean_trend < 0.3 and var_trend < 0.5

        return {
            "is_stationary": is_stationary,
            "mean_variation": float(mean_trend),
            "variance_variation": float(var_trend),
            "rolling_mean_range": (float(rolling_mean.min()), float(rolling_mean.max())),
            "rolling_var_range": (float(rolling_var.min()), float(rolling_var.max())),
        }

    def autocorrelation(
        self,
        data: Union[list, np.ndarray],
        max_lag: Optional[int] = None,
    ) -> np.ndarray:
        """
        Berechnet die Autokorrelationsfunktion (ACF).
        """
        arr = np.array(data, dtype=float)
        n = len(arr)
        max_lag = max_lag or n // 2

        arr = arr - np.mean(arr)
        acf = np.zeros(max_lag + 1)
        var = np.var(arr)

        if var == 0:
            return acf

        for lag in range(max_lag + 1):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean(arr[:-lag] * arr[lag:]) / var

        return acf

    def partial_autocorrelation(
        self,
        data: Union[list, np.ndarray],
        max_lag: Optional[int] = None,
    ) -> np.ndarray:
        """
        Berechnet die partielle Autokorrelationsfunktion (PACF).
        """
        arr = np.array(data, dtype=float)
        n = len(arr)
        max_lag = max_lag or min(n // 4, 40)

        acf = self.autocorrelation(arr, max_lag)
        pacf = np.zeros(max_lag + 1)
        pacf[0] = 1.0

        # Durbin-Levinson Algorithmus
        phi = np.zeros((max_lag + 1, max_lag + 1))

        for k in range(1, max_lag + 1):
            # Berechne phi_kk
            if k == 1:
                phi[k, k] = acf[1]
            else:
                num = acf[k] - sum(phi[k-1, j] * acf[k-j] for j in range(1, k))
                denom = 1 - sum(phi[k-1, j] * acf[j] for j in range(1, k))
                phi[k, k] = num / denom if denom != 0 else 0

                # Update phi_kj für j < k
                for j in range(1, k):
                    phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]

            pacf[k] = phi[k, k]

        return pacf

    def difference(
        self,
        data: Union[list, np.ndarray],
        order: int = 1,
        seasonal: Optional[int] = None,
    ) -> np.ndarray:
        """
        Differenziert die Zeitreihe.

        Args:
            data: Eingangsdaten
            order: Differenzierungsordnung
            seasonal: Saisonale Periode (falls vorhanden)
        """
        arr = np.array(data, dtype=float)

        # Reguläre Differenzierung
        for _ in range(order):
            arr = np.diff(arr)

        # Saisonale Differenzierung
        if seasonal and seasonal < len(arr):
            arr = arr[seasonal:] - arr[:-seasonal]

        return arr

    def spectral_analysis(
        self,
        data: Union[list, np.ndarray],
    ) -> dict[str, Any]:
        """
        Spektralanalyse mittels FFT.
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        # Entferne Trend
        arr = arr - np.mean(arr)

        # FFT
        fft = np.fft.fft(arr)
        freqs = np.fft.fftfreq(n)
        power = np.abs(fft) ** 2

        # Nur positive Frequenzen
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_power = power[positive_mask]

        # Finde dominante Frequenzen
        sorted_indices = np.argsort(positive_power)[::-1]
        top_indices = sorted_indices[:5]

        dominant_frequencies = []
        for idx in top_indices:
            freq = positive_freqs[idx]
            period = 1 / freq if freq > 0 else float('inf')
            dominant_frequencies.append({
                "frequency": float(freq),
                "period": float(period),
                "power": float(positive_power[idx]),
            })

        return {
            "frequencies": positive_freqs.tolist(),
            "power_spectrum": positive_power.tolist(),
            "dominant_frequencies": dominant_frequencies,
        }


class SeasonalDecomposer:
    """
    Saisonale Zerlegung von Zeitreihen.

    Zerlegt eine Zeitreihe in:
    - Trend-Komponente
    - Saisonale Komponente
    - Residuum
    """

    def __init__(self, method: str = "additive"):
        self.method = method  # "additive" oder "multiplicative"

    def decompose(
        self,
        data: Union[list, np.ndarray],
        period: int,
    ) -> SeasonalComponents:
        """
        Führt die saisonale Zerlegung durch.

        Args:
            data: Die Zeitreihe
            period: Die Saisonperiode
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        # 1. Trend extrahieren (zentrierter gleitender Durchschnitt)
        trend = self._moving_average(arr, period)

        # 2. Detrend
        if self.method == "multiplicative":
            detrended = arr / np.where(trend != 0, trend, 1)
        else:
            detrended = arr - trend

        # 3. Saisonale Komponente (Durchschnitt pro Position in Periode)
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal_value = np.nanmean(detrended[indices])
            seasonal[indices] = seasonal_value

        # Normalisiere saisonale Komponente
        if self.method == "multiplicative":
            seasonal = seasonal / np.mean(seasonal)
        else:
            seasonal = seasonal - np.mean(seasonal)

        # 4. Residuum
        if self.method == "multiplicative":
            residual = arr / (trend * seasonal)
            residual = np.where(np.isfinite(residual), residual, 1)
        else:
            residual = arr - trend - seasonal

        return SeasonalComponents(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            period=period,
        )

    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Berechnet zentrierten gleitenden Durchschnitt."""
        n = len(data)
        result = np.full(n, np.nan)

        half = window // 2
        for i in range(half, n - half):
            result[i] = np.mean(data[i - half:i + half + 1])

        # Fülle Ränder mit linearer Interpolation
        for i in range(half):
            if not np.isnan(result[half]):
                result[i] = result[half]
        for i in range(n - half, n):
            if not np.isnan(result[n - half - 1]):
                result[i] = result[n - half - 1]

        return result


class Forecaster:
    """
    Zeitreihenprognose.

    Features:
    - Exponentielles Glätten
    - Holt-Winters
    - ARIMA-ähnliche Modelle
    """

    def __init__(self):
        logger.info("Forecaster initialized")

    def simple_exponential_smoothing(
        self,
        data: Union[list, np.ndarray],
        alpha: float = 0.3,
        horizon: int = 10,
        confidence: float = 0.95,
    ) -> Forecast:
        """
        Simple Exponential Smoothing (SES).

        Gut für Zeitreihen ohne Trend oder Saisonalität.
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        # Initialisiere mit erstem Wert
        level = arr[0]
        smoothed = [level]

        # Glätte historische Daten
        for i in range(1, n):
            level = alpha * arr[i] + (1 - alpha) * level
            smoothed.append(level)

        # Residuen für Konfidenzintervall
        residuals = arr - np.array(smoothed)
        std_error = np.std(residuals)

        # Prognose
        forecast = np.full(horizon, level)

        # Konfidenzintervall
        z = 1.96 if confidence == 0.95 else 2.576  # 95% oder 99%
        margin = z * std_error * np.sqrt(1 + np.arange(1, horizon + 1) / n)
        lower = forecast - margin
        upper = forecast + margin

        return Forecast(
            values=forecast,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            horizon=horizon,
            method="SES",
        )

    def holt_linear(
        self,
        data: Union[list, np.ndarray],
        alpha: float = 0.3,
        beta: float = 0.1,
        horizon: int = 10,
        confidence: float = 0.95,
    ) -> Forecast:
        """
        Holt's Linear Exponential Smoothing.

        Gut für Zeitreihen mit Trend, ohne Saisonalität.
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        # Initialisierung
        level = arr[0]
        trend = arr[1] - arr[0] if n > 1 else 0
        smoothed = [level]

        # Glätte historische Daten
        for i in range(1, n):
            new_level = alpha * arr[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level, trend = new_level, new_trend
            smoothed.append(level)

        # Residuen
        residuals = arr - np.array(smoothed)
        std_error = np.std(residuals)

        # Prognose
        forecast = np.array([level + (i + 1) * trend for i in range(horizon)])

        # Konfidenzintervall
        z = 1.96 if confidence == 0.95 else 2.576
        margin = z * std_error * np.sqrt(1 + np.arange(1, horizon + 1) / n)
        lower = forecast - margin
        upper = forecast + margin

        return Forecast(
            values=forecast,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            horizon=horizon,
            method="Holt",
        )

    def holt_winters(
        self,
        data: Union[list, np.ndarray],
        period: int,
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.1,
        horizon: int = 10,
        confidence: float = 0.95,
        multiplicative: bool = True,
    ) -> Forecast:
        """
        Holt-Winters Exponential Smoothing.

        Gut für Zeitreihen mit Trend und Saisonalität.
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        if n < 2 * period:
            # Fallback auf Holt bei zu wenig Daten
            return self.holt_linear(arr, alpha, beta, horizon, confidence)

        # Initialisierung
        level = np.mean(arr[:period])
        trend = (np.mean(arr[period:2*period]) - np.mean(arr[:period])) / period

        if multiplicative:
            seasonal = arr[:period] / level
        else:
            seasonal = arr[:period] - level

        # Glätte
        smoothed = []
        for i in range(n):
            season_idx = i % period
            if multiplicative:
                new_level = alpha * (arr[i] / seasonal[season_idx]) + (1 - alpha) * (level + trend)
                new_trend = beta * (new_level - level) + (1 - beta) * trend
                new_seasonal = gamma * (arr[i] / new_level) + (1 - gamma) * seasonal[season_idx]
                smoothed.append(new_level * seasonal[season_idx])
            else:
                new_level = alpha * (arr[i] - seasonal[season_idx]) + (1 - alpha) * (level + trend)
                new_trend = beta * (new_level - level) + (1 - beta) * trend
                new_seasonal = gamma * (arr[i] - new_level) + (1 - gamma) * seasonal[season_idx]
                smoothed.append(new_level + seasonal[season_idx])

            level, trend = new_level, new_trend
            seasonal[season_idx] = new_seasonal

        # Residuen
        residuals = arr - np.array(smoothed)
        std_error = np.std(residuals)

        # Prognose
        forecast = []
        for h in range(1, horizon + 1):
            season_idx = (n + h - 1) % period
            if multiplicative:
                forecast.append((level + h * trend) * seasonal[season_idx])
            else:
                forecast.append(level + h * trend + seasonal[season_idx])
        forecast = np.array(forecast)

        # Konfidenzintervall
        z = 1.96 if confidence == 0.95 else 2.576
        margin = z * std_error * np.sqrt(1 + np.arange(1, horizon + 1) / n)
        lower = forecast - margin
        upper = forecast + margin

        return Forecast(
            values=forecast,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            horizon=horizon,
            method="Holt-Winters",
        )

    def auto_forecast(
        self,
        data: Union[list, np.ndarray],
        horizon: int = 10,
        confidence: float = 0.95,
    ) -> Forecast:
        """
        Automatische Methodenauswahl und Prognose.
        """
        arr = np.array(data, dtype=float)
        n = len(arr)

        # Analysiere Zeitreihe
        analyzer = TimeSeriesAnalyzer()
        stationarity = analyzer.is_stationary(arr)

        # Prüfe auf Saisonalität
        acf = analyzer.autocorrelation(arr, min(n // 2, 50))
        peaks = self._find_acf_peaks(acf)

        if peaks:
            # Saisonalität erkannt
            period = peaks[0]
            if period >= 2 and n >= 2 * period:
                return self.holt_winters(arr, period, horizon=horizon, confidence=confidence)

        # Prüfe auf Trend
        x = np.arange(n)
        corr = np.corrcoef(x, arr)[0, 1]

        if abs(corr) > 0.5:
            # Trend erkannt
            return self.holt_linear(arr, horizon=horizon, confidence=confidence)
        else:
            # Kein Trend
            return self.simple_exponential_smoothing(arr, horizon=horizon, confidence=confidence)

    def _find_acf_peaks(self, acf: np.ndarray) -> list[int]:
        """Findet Peaks in der ACF für Periodendetektion."""
        peaks = []
        n = len(acf)

        for i in range(2, n - 1):
            if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.2:
                peaks.append(i)

        return peaks
