"""
SCIO Market Analyzer

Knallharte Marktanalyse mit technischen und fundamentalen Indikatoren.
Liefert Daten fuer sichere, gewinnorientierte Entscheidungen.
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import math

from scio.core.logging import get_logger

logger = get_logger(__name__)


class TrendDirection(str, Enum):
    """Trendrichtung."""
    STRONG_UP = "strong_up"
    UP = "up"
    SIDEWAYS = "sideways"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


class SignalStrength(str, Enum):
    """Signalstaerke."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MarketCondition(str, Enum):
    """Marktbedingung."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"


@dataclass
class PriceData:
    """Preisdaten fuer Analyse."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def typical_price(self) -> float:
        """Typischer Preis (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        """Tages-Range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Kerzenkoerper."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Bullische Kerze?"""
        return self.close > self.open


@dataclass
class TechnicalIndicators:
    """Sammlung technischer Indikatoren."""
    # Trend
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0

    # Momentum
    rsi_14: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Volatilitaet
    atr_14: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_middle: float = 0.0

    # Volumen
    volume_sma_20: float = 0.0
    volume_ratio: float = 1.0

    # Berechnet
    trend: TrendDirection = TrendDirection.SIDEWAYS
    signal: SignalStrength = SignalStrength.NEUTRAL
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "ema_12": self.ema_12,
            "ema_26": self.ema_26,
            "rsi_14": self.rsi_14,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "atr_14": self.atr_14,
            "bollinger_upper": self.bollinger_upper,
            "bollinger_lower": self.bollinger_lower,
            "bollinger_middle": self.bollinger_middle,
            "volume_ratio": self.volume_ratio,
            "trend": self.trend.value,
            "signal": self.signal.value,
            "confidence": self.confidence,
        }


@dataclass
class MarketAnalysis:
    """Vollstaendige Marktanalyse."""
    symbol: str
    timestamp: datetime
    current_price: float
    indicators: TechnicalIndicators
    market_condition: MarketCondition
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    risk_score: float = 0.5  # 0 = niedrig, 1 = hoch
    opportunity_score: float = 0.5  # 0 = niedrig, 1 = hoch
    recommendation: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "indicators": self.indicators.to_dict(),
            "market_condition": self.market_condition.value,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "risk_score": self.risk_score,
            "opportunity_score": self.opportunity_score,
            "recommendation": self.recommendation,
            "warnings": self.warnings,
        }


class MarketAnalyzer:
    """
    Umfassender Marktanalysator.

    Berechnet technische Indikatoren und generiert
    Handlungsempfehlungen basierend auf knallharten Daten.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".scio" / "market_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[PriceData]] = {}
        logger.info("MarketAnalyzer initialized")

    # === INDIKATOREN-BERECHNUNG ===

    def calculate_sma(self, prices: list[float], period: int) -> float:
        """Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return statistics.mean(prices[-period:])

    def calculate_ema(self, prices: list[float], period: int) -> float:
        """Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def calculate_rsi(self, prices: list[float], period: int = 14) -> float:
        """Relative Strength Index (0-100)."""
        if len(prices) < period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = statistics.mean(gains[-period:])
        avg_loss = statistics.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        prices: list[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, float, float]:
        """MACD (Line, Signal, Histogram)."""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow

        # Signal berechnen (vereinfacht)
        if len(prices) > slow + signal:
            macd_values = []
            for i in range(signal):
                end = len(prices) - i
                ema_f = self.calculate_ema(prices[:end], fast)
                ema_s = self.calculate_ema(prices[:end], slow)
                macd_values.append(ema_f - ema_s)
            macd_signal = statistics.mean(macd_values)
        else:
            macd_signal = macd_line

        histogram = macd_line - macd_signal

        return macd_line, macd_signal, histogram

    def calculate_atr(self, candles: list[PriceData], period: int = 14) -> float:
        """Average True Range."""
        if len(candles) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            high_low = candles[i].high - candles[i].low
            high_prev = abs(candles[i].high - candles[i - 1].close)
            low_prev = abs(candles[i].low - candles[i - 1].close)
            true_ranges.append(max(high_low, high_prev, low_prev))

        if len(true_ranges) < period:
            return statistics.mean(true_ranges)

        return statistics.mean(true_ranges[-period:])

    def calculate_bollinger_bands(
        self,
        prices: list[float],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[float, float, float]:
        """Bollinger Bands (Upper, Middle, Lower)."""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]

        middle = statistics.mean(prices[-period:])
        std = statistics.stdev(prices[-period:])

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def find_support_resistance(
        self,
        candles: list[PriceData],
        lookback: int = 50,
    ) -> tuple[list[float], list[float]]:
        """Findet Support- und Resistance-Levels."""
        if len(candles) < lookback:
            lookback = len(candles)

        highs = [c.high for c in candles[-lookback:]]
        lows = [c.low for c in candles[-lookback:]]

        # Finde lokale Hochs/Tiefs
        support_levels = []
        resistance_levels = []

        for i in range(2, len(lows) - 2):
            # Lokales Tief (Support)
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2]:
                if lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                    support_levels.append(lows[i])

            # Lokales Hoch (Resistance)
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2]:
                if highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                    resistance_levels.append(highs[i])

        # Dedupliziere (Levels die nahe beieinander liegen zusammenfassen)
        support_levels = self._cluster_levels(support_levels)
        resistance_levels = self._cluster_levels(resistance_levels)

        return support_levels[:5], resistance_levels[:5]

    def _cluster_levels(self, levels: list[float], threshold: float = 0.02) -> list[float]:
        """Gruppiert nahe Levels."""
        if not levels:
            return []

        levels = sorted(levels)
        clustered = [levels[0]]

        for level in levels[1:]:
            if abs(level - clustered[-1]) / clustered[-1] > threshold:
                clustered.append(level)
            else:
                # Durchschnitt der Gruppe
                clustered[-1] = (clustered[-1] + level) / 2

        return clustered

    # === VOLLSTAENDIGE ANALYSE ===

    def analyze(self, symbol: str, candles: list[PriceData]) -> MarketAnalysis:
        """
        Fuehrt vollstaendige Marktanalyse durch.

        Args:
            symbol: Trading-Symbol (z.B. BTC, ETH, AAPL)
            candles: Liste von Preisdaten

        Returns:
            MarketAnalysis mit allen Indikatoren und Empfehlungen
        """
        if len(candles) < 30:
            raise ValueError("Mindestens 30 Kerzen fuer Analyse benoetigt")

        closes = [c.close for c in candles]
        current_price = closes[-1]

        # Berechne alle Indikatoren
        indicators = TechnicalIndicators(
            sma_20=self.calculate_sma(closes, 20),
            sma_50=self.calculate_sma(closes, 50),
            sma_200=self.calculate_sma(closes, 200),
            ema_12=self.calculate_ema(closes, 12),
            ema_26=self.calculate_ema(closes, 26),
            rsi_14=self.calculate_rsi(closes, 14),
            atr_14=self.calculate_atr(candles, 14),
        )

        # MACD
        macd = self.calculate_macd(closes)
        indicators.macd_line = macd[0]
        indicators.macd_signal = macd[1]
        indicators.macd_histogram = macd[2]

        # Bollinger Bands
        bb = self.calculate_bollinger_bands(closes)
        indicators.bollinger_upper = bb[0]
        indicators.bollinger_middle = bb[1]
        indicators.bollinger_lower = bb[2]

        # Volumen-Analyse
        volumes = [c.volume for c in candles]
        indicators.volume_sma_20 = self.calculate_sma(volumes, 20)
        if indicators.volume_sma_20 > 0:
            indicators.volume_ratio = volumes[-1] / indicators.volume_sma_20

        # Trend bestimmen
        indicators.trend = self._determine_trend(current_price, indicators)

        # Signal generieren
        indicators.signal = self._generate_signal(current_price, indicators)

        # Konfidenz berechnen
        indicators.confidence = self._calculate_confidence(indicators)

        # Support/Resistance
        support, resistance = self.find_support_resistance(candles)

        # Marktbedingung
        market_condition = self._assess_market_condition(candles, indicators)

        # Risiko und Opportunity Scores
        risk_score = self._calculate_risk_score(indicators, market_condition)
        opportunity_score = self._calculate_opportunity_score(
            current_price, indicators, support, resistance
        )

        # Empfehlung generieren
        recommendation = self._generate_recommendation(
            indicators, market_condition, risk_score, opportunity_score
        )

        # Warnungen
        warnings = self._generate_warnings(indicators, market_condition)

        analysis = MarketAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
            indicators=indicators,
            market_condition=market_condition,
            support_levels=support,
            resistance_levels=resistance,
            risk_score=risk_score,
            opportunity_score=opportunity_score,
            recommendation=recommendation,
            warnings=warnings,
        )

        # Speichern
        self._save_analysis(analysis)

        return analysis

    def _determine_trend(
        self,
        price: float,
        ind: TechnicalIndicators,
    ) -> TrendDirection:
        """Bestimmt die Trendrichtung."""
        score = 0

        # SMA-Hierarchie
        if price > ind.sma_20 > ind.sma_50 > ind.sma_200:
            score += 2
        elif price > ind.sma_50 > ind.sma_200:
            score += 1
        elif price < ind.sma_20 < ind.sma_50 < ind.sma_200:
            score -= 2
        elif price < ind.sma_50 < ind.sma_200:
            score -= 1

        # MACD
        if ind.macd_histogram > 0:
            score += 1
        elif ind.macd_histogram < 0:
            score -= 1

        # RSI
        if ind.rsi_14 > 60:
            score += 0.5
        elif ind.rsi_14 < 40:
            score -= 0.5

        if score >= 2:
            return TrendDirection.STRONG_UP
        elif score >= 1:
            return TrendDirection.UP
        elif score <= -2:
            return TrendDirection.STRONG_DOWN
        elif score <= -1:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS

    def _generate_signal(
        self,
        price: float,
        ind: TechnicalIndicators,
    ) -> SignalStrength:
        """Generiert Trading-Signal."""
        buy_signals = 0
        sell_signals = 0

        # RSI
        if ind.rsi_14 < 30:
            buy_signals += 2  # Ueberverkauft
        elif ind.rsi_14 < 40:
            buy_signals += 1
        elif ind.rsi_14 > 70:
            sell_signals += 2  # Ueberkauft
        elif ind.rsi_14 > 60:
            sell_signals += 1

        # MACD Crossover
        if ind.macd_histogram > 0 and ind.macd_line > ind.macd_signal:
            buy_signals += 1
        elif ind.macd_histogram < 0 and ind.macd_line < ind.macd_signal:
            sell_signals += 1

        # Bollinger Bands
        if price < ind.bollinger_lower:
            buy_signals += 1  # Nahe unterer Band
        elif price > ind.bollinger_upper:
            sell_signals += 1  # Nahe oberer Band

        # SMA Crossover
        if ind.sma_20 > ind.sma_50:
            buy_signals += 1
        elif ind.sma_20 < ind.sma_50:
            sell_signals += 1

        # Volumen-Bestaetigung
        if ind.volume_ratio > 1.5:
            if buy_signals > sell_signals:
                buy_signals += 1
            else:
                sell_signals += 1

        # Signal bestimmen
        diff = buy_signals - sell_signals

        if diff >= 4:
            return SignalStrength.STRONG_BUY
        elif diff >= 2:
            return SignalStrength.BUY
        elif diff <= -4:
            return SignalStrength.STRONG_SELL
        elif diff <= -2:
            return SignalStrength.SELL
        else:
            return SignalStrength.NEUTRAL

    def _calculate_confidence(self, ind: TechnicalIndicators) -> float:
        """Berechnet Konfidenz der Analyse."""
        # Je mehr Indikatoren uebereinstimmen, desto hoeher die Konfidenz
        agreement = 0
        total = 0

        # Trend-Indikatoren
        trend_bullish = ind.trend in [TrendDirection.STRONG_UP, TrendDirection.UP]
        trend_bearish = ind.trend in [TrendDirection.STRONG_DOWN, TrendDirection.DOWN]

        # Signal
        signal_bullish = ind.signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]
        signal_bearish = ind.signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]

        # RSI
        rsi_bullish = ind.rsi_14 < 50
        rsi_bearish = ind.rsi_14 > 50

        # MACD
        macd_bullish = ind.macd_histogram > 0
        macd_bearish = ind.macd_histogram < 0

        # Zaehle Uebereinstimmungen
        bullish_count = sum([trend_bullish, signal_bullish, rsi_bullish, macd_bullish])
        bearish_count = sum([trend_bearish, signal_bearish, rsi_bearish, macd_bearish])

        max_agreement = max(bullish_count, bearish_count)
        confidence = max_agreement / 4.0

        # Reduziere bei neutralen Signalen
        if ind.signal == SignalStrength.NEUTRAL:
            confidence *= 0.7

        return min(0.95, confidence)

    def _assess_market_condition(
        self,
        candles: list[PriceData],
        ind: TechnicalIndicators,
    ) -> MarketCondition:
        """Bestimmt die Marktbedingung."""
        # Volatilitaet pruefen
        closes = [c.close for c in candles[-20:]]
        if len(closes) >= 2:
            volatility = statistics.stdev(closes) / statistics.mean(closes)
        else:
            volatility = 0

        # ATR-basierte Volatilitaet
        atr_percent = ind.atr_14 / candles[-1].close if candles[-1].close > 0 else 0

        # Krisen-Check (hohe Volatilitaet + starker Downtrend)
        if atr_percent > 0.05 and ind.trend == TrendDirection.STRONG_DOWN:
            return MarketCondition.CRISIS

        # Volatile Maerkte
        if volatility > 0.03 or atr_percent > 0.04:
            return MarketCondition.VOLATILE

        # Bull/Bear Market
        if ind.trend in [TrendDirection.STRONG_UP, TrendDirection.UP]:
            return MarketCondition.BULL_MARKET
        elif ind.trend in [TrendDirection.STRONG_DOWN, TrendDirection.DOWN]:
            return MarketCondition.BEAR_MARKET
        else:
            return MarketCondition.RANGING

    def _calculate_risk_score(
        self,
        ind: TechnicalIndicators,
        condition: MarketCondition,
    ) -> float:
        """Berechnet Risiko-Score (0-1)."""
        risk = 0.5

        # Marktbedingung
        condition_risk = {
            MarketCondition.BULL_MARKET: 0.3,
            MarketCondition.BEAR_MARKET: 0.7,
            MarketCondition.RANGING: 0.5,
            MarketCondition.VOLATILE: 0.8,
            MarketCondition.CRISIS: 0.95,
        }
        risk = condition_risk.get(condition, 0.5)

        # RSI Extreme
        if ind.rsi_14 > 80 or ind.rsi_14 < 20:
            risk += 0.1

        # Niedrige Konfidenz = hoeheres Risiko
        risk += (1 - ind.confidence) * 0.2

        return min(1.0, risk)

    def _calculate_opportunity_score(
        self,
        price: float,
        ind: TechnicalIndicators,
        support: list[float],
        resistance: list[float],
    ) -> float:
        """Berechnet Opportunity-Score (0-1)."""
        opp = 0.5

        # Naehe zu Support (Kaufgelegenheit)
        if support:
            nearest_support = min(support, key=lambda x: abs(x - price))
            support_distance = (price - nearest_support) / price
            if 0 < support_distance < 0.05:  # 5% ueber Support
                opp += 0.2

        # Ueberverkauft (RSI)
        if ind.rsi_14 < 30:
            opp += 0.2
        elif ind.rsi_14 < 40:
            opp += 0.1

        # Bullisches Signal
        if ind.signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            opp += 0.2

        # Volumen-Bestaetigung
        if ind.volume_ratio > 1.5:
            opp += 0.1

        return min(1.0, opp)

    def _generate_recommendation(
        self,
        ind: TechnicalIndicators,
        condition: MarketCondition,
        risk: float,
        opportunity: float,
    ) -> str:
        """Generiert Handlungsempfehlung."""
        recommendations = []

        # Signal-basiert
        if ind.signal == SignalStrength.STRONG_BUY:
            recommendations.append(
                "STARKES KAUFSIGNAL: Mehrere Indikatoren bullisch."
            )
        elif ind.signal == SignalStrength.BUY:
            recommendations.append(
                "KAUFSIGNAL: Tendenz positiv, auf Bestaetigung warten."
            )
        elif ind.signal == SignalStrength.STRONG_SELL:
            recommendations.append(
                "STARKES VERKAUFSSIGNAL: Mehrere Indikatoren bearisch."
            )
        elif ind.signal == SignalStrength.SELL:
            recommendations.append(
                "VERKAUFSSIGNAL: Tendenz negativ, Gewinne sichern."
            )
        else:
            recommendations.append(
                "NEUTRAL: Abwarten bis klares Signal entsteht."
            )

        # Risiko-Warnung
        if risk > 0.7:
            recommendations.append(
                "HOHES RISIKO: Positionsgroesse reduzieren oder absichern."
            )
        elif risk > 0.5:
            recommendations.append(
                "MODERATES RISIKO: Stop-Loss unbedingt setzen."
            )

        # Opportunity
        if opportunity > 0.7 and risk < 0.6:
            recommendations.append(
                "GUTE CHANCE: Risk/Reward-Verhaeltnis attraktiv."
            )

        # Marktbedingung
        if condition == MarketCondition.CRISIS:
            recommendations.append(
                "VORSICHT: Krisenmarkt - nur mit minimaler Position agieren."
            )
        elif condition == MarketCondition.VOLATILE:
            recommendations.append(
                "VOLATIL: Weiterer Stop-Loss empfohlen, schnelle Bewegungen moeglich."
            )

        return " | ".join(recommendations)

    def _generate_warnings(
        self,
        ind: TechnicalIndicators,
        condition: MarketCondition,
    ) -> list[str]:
        """Generiert Warnungen."""
        warnings = []

        if ind.rsi_14 > 70:
            warnings.append("RSI ueberkauft (>70) - Ruecksetzer wahrscheinlich")
        if ind.rsi_14 < 30:
            warnings.append("RSI ueberverkauft (<30) - Erholung moeglich")

        if condition == MarketCondition.CRISIS:
            warnings.append("KRISENMARKT: Extreme Vorsicht geboten!")
        if condition == MarketCondition.VOLATILE:
            warnings.append("Hohe Volatilitaet: Groessere Preisschwankungen erwartet")

        if ind.confidence < 0.5:
            warnings.append("Niedrige Analyse-Konfidenz: Widersprüchliche Signale")

        # MACD-Divergenz warnen
        if ind.macd_histogram < 0 and ind.trend == TrendDirection.UP:
            warnings.append("Bearische MACD-Divergenz: Trendumkehr moeglich")
        elif ind.macd_histogram > 0 and ind.trend == TrendDirection.DOWN:
            warnings.append("Bullische MACD-Divergenz: Trendumkehr moeglich")

        return warnings

    def _save_analysis(self, analysis: MarketAnalysis) -> None:
        """Speichert Analyse fuer Backtesting."""
        file_path = self.data_dir / f"analysis_{analysis.symbol}_{analysis.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        file_path.write_text(json.dumps(analysis.to_dict(), indent=2))


# === BENCHMARK SYSTEM ===

@dataclass
class BenchmarkResult:
    """Ergebnis eines Strategie-Benchmarks."""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    start_date: datetime
    end_date: datetime

    @property
    def is_profitable(self) -> bool:
        return self.total_return > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "is_profitable": self.is_profitable,
        }


class StrategyBenchmark:
    """
    Backtesting und Benchmarking von Trading-Strategien.
    """

    def __init__(self, analyzer: MarketAnalyzer):
        self.analyzer = analyzer
        self._results: list[BenchmarkResult] = []

    def backtest_rsi_strategy(
        self,
        candles: list[PriceData],
        oversold: float = 30,
        overbought: float = 70,
        initial_capital: float = 10000,
    ) -> BenchmarkResult:
        """
        Backtestet RSI-Mean-Reversion-Strategie.

        Kaufe bei RSI < oversold, verkaufe bei RSI > overbought.
        """
        if len(candles) < 50:
            raise ValueError("Mindestens 50 Kerzen fuer Backtest benoetigt")

        trades = []
        position = None
        capital = initial_capital
        peak_capital = initial_capital

        for i in range(30, len(candles)):
            closes = [c.close for c in candles[:i + 1]]
            rsi = self.analyzer.calculate_rsi(closes)
            price = candles[i].close

            if position is None and rsi < oversold:
                # Kaufsignal
                position = {
                    "entry_price": price,
                    "entry_date": candles[i].timestamp,
                    "shares": capital / price,
                }
            elif position is not None and rsi > overbought:
                # Verkaufssignal
                exit_value = position["shares"] * price
                trade_return = (exit_value - capital) / capital
                trades.append(trade_return)
                capital = exit_value
                peak_capital = max(peak_capital, capital)
                position = None

        # Position am Ende schliessen
        if position is not None:
            exit_value = position["shares"] * candles[-1].close
            trade_return = (exit_value - capital) / capital
            trades.append(trade_return)
            capital = exit_value

        return self._calculate_benchmark_result(
            "RSI_MeanReversion",
            trades,
            initial_capital,
            capital,
            candles[0].timestamp,
            candles[-1].timestamp,
        )

    def backtest_ma_crossover(
        self,
        candles: list[PriceData],
        fast_period: int = 20,
        slow_period: int = 50,
        initial_capital: float = 10000,
    ) -> BenchmarkResult:
        """
        Backtestet Moving-Average-Crossover-Strategie.

        Kaufe bei Golden Cross, verkaufe bei Death Cross.
        """
        if len(candles) < slow_period + 10:
            raise ValueError(f"Mindestens {slow_period + 10} Kerzen benoetigt")

        trades = []
        position = None
        capital = initial_capital
        prev_fast = 0
        prev_slow = 0

        for i in range(slow_period, len(candles)):
            closes = [c.close for c in candles[:i + 1]]
            fast_ma = self.analyzer.calculate_sma(closes, fast_period)
            slow_ma = self.analyzer.calculate_sma(closes, slow_period)
            price = candles[i].close

            # Golden Cross
            if position is None and prev_fast <= prev_slow and fast_ma > slow_ma:
                position = {
                    "entry_price": price,
                    "shares": capital / price,
                }

            # Death Cross
            elif position is not None and prev_fast >= prev_slow and fast_ma < slow_ma:
                exit_value = position["shares"] * price
                trade_return = (exit_value - capital) / capital
                trades.append(trade_return)
                capital = exit_value
                position = None

            prev_fast = fast_ma
            prev_slow = slow_ma

        # Position am Ende schliessen
        if position is not None:
            exit_value = position["shares"] * candles[-1].close
            trade_return = (exit_value - capital) / capital
            trades.append(trade_return)
            capital = exit_value

        return self._calculate_benchmark_result(
            "MA_Crossover",
            trades,
            initial_capital,
            capital,
            candles[0].timestamp,
            candles[-1].timestamp,
        )

    def _calculate_benchmark_result(
        self,
        strategy_name: str,
        trades: list[float],
        initial_capital: float,
        final_capital: float,
        start_date: datetime,
        end_date: datetime,
    ) -> BenchmarkResult:
        """Berechnet Benchmark-Metriken."""
        if not trades:
            return BenchmarkResult(
                strategy_name=strategy_name,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                avg_trade_return=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                start_date=start_date,
                end_date=end_date,
            )

        winning = [t for t in trades if t > 0]
        losing = [t for t in trades if t <= 0]

        win_rate = len(winning) / len(trades) if trades else 0
        total_return = (final_capital - initial_capital) / initial_capital
        avg_return = statistics.mean(trades)

        # Sharpe Ratio (vereinfacht, annualisiert)
        if len(trades) > 1:
            std_return = statistics.stdev(trades)
            sharpe = (avg_return / std_return) * math.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0

        # Profit Factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0.0001
        profit_factor = gross_profit / gross_loss

        # Max Drawdown
        cumulative = [initial_capital]
        for t in trades:
            cumulative.append(cumulative[-1] * (1 + t))
        peak = cumulative[0]
        max_dd = 0
        for val in cumulative:
            peak = max(peak, val)
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)

        result = BenchmarkResult(
            strategy_name=strategy_name,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_return=avg_return,
            best_trade=max(trades),
            worst_trade=min(trades),
            start_date=start_date,
            end_date=end_date,
        )

        self._results.append(result)
        return result

    def compare_strategies(self) -> list[dict[str, Any]]:
        """Vergleicht alle getesteten Strategien."""
        comparison = []
        for result in self._results:
            comparison.append({
                "strategy": result.strategy_name,
                "return": f"{result.total_return * 100:.2f}%",
                "win_rate": f"{result.win_rate * 100:.1f}%",
                "sharpe": f"{result.sharpe_ratio:.2f}",
                "max_dd": f"{result.max_drawdown * 100:.1f}%",
                "trades": result.total_trades,
                "profitable": result.is_profitable,
            })

        # Sortiere nach Sharpe Ratio
        comparison.sort(key=lambda x: float(x["sharpe"].replace(",", ".")), reverse=True)
        return comparison


if __name__ == "__main__":
    # Test-Beispiel
    import random

    # Generiere Testdaten
    candles = []
    price = 100.0
    for i in range(200):
        change = random.uniform(-0.03, 0.03)
        open_p = price
        close_p = price * (1 + change)
        high_p = max(open_p, close_p) * (1 + random.uniform(0, 0.02))
        low_p = min(open_p, close_p) * (1 - random.uniform(0, 0.02))
        volume = random.uniform(1000000, 5000000)

        candles.append(PriceData(
            timestamp=datetime.now() - timedelta(days=200 - i),
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=volume,
        ))
        price = close_p

    # Analyse
    analyzer = MarketAnalyzer()
    analysis = analyzer.analyze("TEST", candles)

    print("=" * 60)
    print("MARKTANALYSE")
    print("=" * 60)
    print(f"Symbol: {analysis.symbol}")
    print(f"Preis: {analysis.current_price:.2f}")
    print(f"Trend: {analysis.indicators.trend.value}")
    print(f"Signal: {analysis.indicators.signal.value}")
    print(f"Konfidenz: {analysis.indicators.confidence:.0%}")
    print(f"RSI: {analysis.indicators.rsi_14:.1f}")
    print(f"Markt: {analysis.market_condition.value}")
    print(f"Risiko: {analysis.risk_score:.0%}")
    print(f"Opportunity: {analysis.opportunity_score:.0%}")
    print(f"\nEmpfehlung: {analysis.recommendation}")

    # Backtest
    print("\n" + "=" * 60)
    print("STRATEGIE-BENCHMARK")
    print("=" * 60)

    benchmark = StrategyBenchmark(analyzer)
    rsi_result = benchmark.backtest_rsi_strategy(candles)
    ma_result = benchmark.backtest_ma_crossover(candles)

    print(f"\nRSI Strategy: {rsi_result.total_return * 100:.2f}% Return, {rsi_result.win_rate * 100:.0f}% Win Rate")
    print(f"MA Crossover: {ma_result.total_return * 100:.2f}% Return, {ma_result.win_rate * 100:.0f}% Win Rate")

    print("\nStrategie-Vergleich:")
    for s in benchmark.compare_strategies():
        print(f"  {s['strategy']}: {s['return']} | Sharpe: {s['sharpe']}")
