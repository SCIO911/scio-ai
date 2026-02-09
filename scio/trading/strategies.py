"""
SCIO Trading Strategies

Vollautomatische Trading-Strategien mit:
- Backtesting
- Signal-Generierung
- Risk Management
- Position Sizing
"""

import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from scio.core.logging import get_logger
from scio.trading.market_analyzer import PriceData, TechnicalIndicators

logger = get_logger(__name__)


class SignalType(str, Enum):
    """Signal-Typen."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class PositionType(str, Enum):
    """Position-Typen."""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class TradingSignal:
    """Ein Trading-Signal."""
    timestamp: datetime
    signal_type: SignalType
    symbol: str
    price: Decimal
    confidence: float  # 0-1
    strategy_name: str
    reason: str
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: float = 1.0  # 0-1 des Kapitals

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "price": str(self.price),
            "confidence": self.confidence,
            "strategy_name": self.strategy_name,
            "reason": self.reason,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "position_size": self.position_size,
        }


@dataclass
class Trade:
    """Ein ausgefuehrter Trade."""
    id: str
    symbol: str
    position_type: PositionType
    entry_price: Decimal
    entry_time: datetime
    quantity: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    pnl: Decimal = Decimal("0")
    pnl_percent: float = 0.0
    status: str = "open"

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def close(self, exit_price: Decimal, exit_time: datetime) -> None:
        """Schliesst den Trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time

        if self.position_type == PositionType.LONG:
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity

        self.pnl_percent = float(self.pnl / (self.entry_price * self.quantity) * 100)
        self.status = "closed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "position_type": self.position_type.value,
            "entry_price": str(self.entry_price),
            "entry_time": self.entry_time.isoformat(),
            "quantity": str(self.quantity),
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": str(self.pnl),
            "pnl_percent": self.pnl_percent,
            "status": self.status,
        }


class TradingStrategy(ABC):
    """
    Basis-Klasse fuer Trading-Strategien.
    """

    def __init__(self, name: str):
        self.name = name
        self.signals: list[TradingSignal] = []
        self.trades: list[Trade] = []

    @abstractmethod
    def generate_signal(
        self,
        candles: list[PriceData],
        current_position: PositionType = PositionType.NONE,
    ) -> Optional[TradingSignal]:
        """Generiert ein Trading-Signal."""
        pass

    def calculate_position_size(
        self,
        capital: Decimal,
        risk_percent: float,
        entry_price: Decimal,
        stop_loss: Decimal,
    ) -> Decimal:
        """
        Berechnet Positionsgroesse basierend auf Risiko.

        Args:
            capital: Verfuegbares Kapital
            risk_percent: Max. Risiko pro Trade (z.B. 0.02 = 2%)
            entry_price: Einstiegspreis
            stop_loss: Stop-Loss Preis

        Returns:
            Anzahl der Einheiten
        """
        risk_amount = capital * Decimal(str(risk_percent))
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return Decimal("0")

        position_size = risk_amount / price_risk
        return position_size


class RSIMeanReversionStrategy(TradingStrategy):
    """
    RSI Mean Reversion Strategie.

    Kauft bei ueberverkauft (RSI < 30), verkauft bei ueberkauft (RSI > 70).
    """

    def __init__(
        self,
        oversold: float = 30,
        overbought: float = 70,
        rsi_period: int = 14,
    ):
        super().__init__("RSI_MeanReversion")
        self.oversold = oversold
        self.overbought = overbought
        self.rsi_period = rsi_period

    def _calculate_rsi(self, prices: list[float]) -> float:
        """Berechnet RSI."""
        if len(prices) < self.rsi_period + 1:
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

        avg_gain = statistics.mean(gains[-self.rsi_period:])
        avg_loss = statistics.mean(losses[-self.rsi_period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signal(
        self,
        candles: list[PriceData],
        current_position: PositionType = PositionType.NONE,
    ) -> Optional[TradingSignal]:
        if len(candles) < self.rsi_period + 5:
            return None

        closes = [float(c.close) for c in candles]
        rsi = self._calculate_rsi(closes)
        current_price = Decimal(str(candles[-1].close))

        # ATR fuer Stop-Loss
        atr = self._calculate_atr(candles)

        if rsi < self.oversold and current_position != PositionType.LONG:
            # Kaufsignal
            stop_loss = current_price - Decimal(str(atr * 2))
            take_profit = current_price + Decimal(str(atr * 3))

            confidence = (self.oversold - rsi) / self.oversold

            return TradingSignal(
                timestamp=candles[-1].timestamp,
                signal_type=SignalType.BUY,
                symbol="",  # Wird extern gesetzt
                price=current_price,
                confidence=min(0.9, confidence),
                strategy_name=self.name,
                reason=f"RSI ueberverkauft ({rsi:.1f})",
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        elif rsi > self.overbought and current_position == PositionType.LONG:
            # Verkaufssignal
            confidence = (rsi - self.overbought) / (100 - self.overbought)

            return TradingSignal(
                timestamp=candles[-1].timestamp,
                signal_type=SignalType.CLOSE,
                symbol="",
                price=current_price,
                confidence=min(0.9, confidence),
                strategy_name=self.name,
                reason=f"RSI ueberkauft ({rsi:.1f})",
            )

        return None

    def _calculate_atr(self, candles: list[PriceData], period: int = 14) -> float:
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


class MACrossoverStrategy(TradingStrategy):
    """
    Moving Average Crossover Strategie.

    Golden Cross (Fast MA > Slow MA) = Buy
    Death Cross (Fast MA < Slow MA) = Sell
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("MA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prev_fast = 0.0
        self._prev_slow = 0.0

    def _calculate_sma(self, prices: list[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return statistics.mean(prices[-period:])

    def generate_signal(
        self,
        candles: list[PriceData],
        current_position: PositionType = PositionType.NONE,
    ) -> Optional[TradingSignal]:
        if len(candles) < self.slow_period + 5:
            return None

        closes = [float(c.close) for c in candles]
        fast_ma = self._calculate_sma(closes, self.fast_period)
        slow_ma = self._calculate_sma(closes, self.slow_period)
        current_price = Decimal(str(candles[-1].close))

        signal = None

        # Golden Cross
        if (self._prev_fast <= self._prev_slow and
            fast_ma > slow_ma and
            current_position != PositionType.LONG):

            atr = self._calculate_atr(candles)
            stop_loss = current_price - Decimal(str(atr * 2))

            signal = TradingSignal(
                timestamp=candles[-1].timestamp,
                signal_type=SignalType.BUY,
                symbol="",
                price=current_price,
                confidence=0.7,
                strategy_name=self.name,
                reason=f"Golden Cross (MA{self.fast_period} > MA{self.slow_period})",
                stop_loss=stop_loss,
            )

        # Death Cross
        elif (self._prev_fast >= self._prev_slow and
              fast_ma < slow_ma and
              current_position == PositionType.LONG):

            signal = TradingSignal(
                timestamp=candles[-1].timestamp,
                signal_type=SignalType.CLOSE,
                symbol="",
                price=current_price,
                confidence=0.7,
                strategy_name=self.name,
                reason=f"Death Cross (MA{self.fast_period} < MA{self.slow_period})",
            )

        self._prev_fast = fast_ma
        self._prev_slow = slow_ma

        return signal

    def _calculate_atr(self, candles: list[PriceData], period: int = 14) -> float:
        if len(candles) < 2:
            return float(candles[-1].close) * 0.02

        true_ranges = []
        for i in range(1, min(len(candles), period + 1)):
            idx = -i
            prev_idx = -i - 1
            high_low = candles[idx].high - candles[idx].low
            high_prev = abs(candles[idx].high - candles[prev_idx].close)
            low_prev = abs(candles[idx].low - candles[prev_idx].close)
            true_ranges.append(max(high_low, high_prev, low_prev))

        return statistics.mean(true_ranges)


class BreakoutStrategy(TradingStrategy):
    """
    Breakout Strategie.

    Kauft bei Ausbruch ueber Resistance, verkauft bei Bruch unter Support.
    """

    def __init__(self, lookback: int = 20, breakout_threshold: float = 0.02):
        super().__init__("Breakout")
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold

    def generate_signal(
        self,
        candles: list[PriceData],
        current_position: PositionType = PositionType.NONE,
    ) -> Optional[TradingSignal]:
        if len(candles) < self.lookback + 5:
            return None

        lookback_candles = candles[-self.lookback - 1:-1]
        current_candle = candles[-1]
        current_price = Decimal(str(current_candle.close))

        # Resistance/Support
        resistance = max(c.high for c in lookback_candles)
        support = min(c.low for c in lookback_candles)

        # Breakout nach oben
        if (current_candle.close > resistance * (1 + self.breakout_threshold) and
            current_position != PositionType.LONG):

            stop_loss = Decimal(str(support))

            return TradingSignal(
                timestamp=current_candle.timestamp,
                signal_type=SignalType.BUY,
                symbol="",
                price=current_price,
                confidence=0.65,
                strategy_name=self.name,
                reason=f"Breakout ueber Resistance ({resistance:.2f})",
                stop_loss=stop_loss,
            )

        # Breakdown nach unten
        elif (current_candle.close < support * (1 - self.breakout_threshold) and
              current_position == PositionType.LONG):

            return TradingSignal(
                timestamp=current_candle.timestamp,
                signal_type=SignalType.CLOSE,
                symbol="",
                price=current_price,
                confidence=0.65,
                strategy_name=self.name,
                reason=f"Breakdown unter Support ({support:.2f})",
            )

        return None


class BollingerBandStrategy(TradingStrategy):
    """
    Bollinger Band Mean Reversion.

    Kauft bei Touch des unteren Bands, verkauft bei Touch des oberen.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("BollingerBands")
        self.period = period
        self.std_dev = std_dev

    def _calculate_bands(self, prices: list[float]) -> tuple[float, float, float]:
        """Berechnet Bollinger Bands."""
        if len(prices) < self.period:
            return prices[-1], prices[-1], prices[-1]

        middle = statistics.mean(prices[-self.period:])
        std = statistics.stdev(prices[-self.period:])

        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        return upper, middle, lower

    def generate_signal(
        self,
        candles: list[PriceData],
        current_position: PositionType = PositionType.NONE,
    ) -> Optional[TradingSignal]:
        if len(candles) < self.period + 5:
            return None

        closes = [float(c.close) for c in candles]
        upper, middle, lower = self._calculate_bands(closes)
        current_price = Decimal(str(candles[-1].close))
        prev_price = Decimal(str(candles[-2].close))

        # Touch unteres Band - Kaufsignal
        if (float(current_price) <= lower and
            float(prev_price) > lower and
            current_position != PositionType.LONG):

            stop_loss = current_price * Decimal("0.95")
            take_profit = Decimal(str(middle))

            return TradingSignal(
                timestamp=candles[-1].timestamp,
                signal_type=SignalType.BUY,
                symbol="",
                price=current_price,
                confidence=0.6,
                strategy_name=self.name,
                reason=f"Touch unteres Bollinger Band ({lower:.2f})",
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # Touch oberes Band - Verkaufssignal
        elif (float(current_price) >= upper and
              current_position == PositionType.LONG):

            return TradingSignal(
                timestamp=candles[-1].timestamp,
                signal_type=SignalType.CLOSE,
                symbol="",
                price=current_price,
                confidence=0.6,
                strategy_name=self.name,
                reason=f"Touch oberes Bollinger Band ({upper:.2f})",
            )

        return None


class StrategyManager:
    """
    Verwaltet mehrere Strategien und kombiniert Signale.
    """

    def __init__(self):
        self.strategies: list[TradingStrategy] = []
        self._trade_counter = 0

    def add_strategy(self, strategy: TradingStrategy) -> None:
        """Fuegt eine Strategie hinzu."""
        self.strategies.append(strategy)
        logger.info("Strategy added", name=strategy.name)

    def generate_signals(
        self,
        candles: list[PriceData],
        current_position: PositionType = PositionType.NONE,
    ) -> list[TradingSignal]:
        """Generiert Signale von allen Strategien."""
        signals = []

        for strategy in self.strategies:
            signal = strategy.generate_signal(candles, current_position)
            if signal:
                signals.append(signal)

        return signals

    def get_consensus_signal(
        self,
        candles: list[PriceData],
        current_position: PositionType = PositionType.NONE,
        min_agreement: float = 0.5,
    ) -> Optional[TradingSignal]:
        """
        Generiert Konsens-Signal wenn mehrere Strategien uebereinstimmen.
        """
        signals = self.generate_signals(candles, current_position)

        if not signals:
            return None

        # Zaehle Signale pro Typ
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.CLOSE]]

        total = len(self.strategies)

        # Buy Konsens?
        if len(buy_signals) >= total * min_agreement:
            # Kombiniere Signale
            avg_confidence = statistics.mean(s.confidence for s in buy_signals)
            reasons = [s.reason for s in buy_signals]

            # Nimm konservativsten Stop-Loss
            stop_losses = [s.stop_loss for s in buy_signals if s.stop_loss]
            stop_loss = max(stop_losses) if stop_losses else None

            return TradingSignal(
                timestamp=candles[-1].timestamp,
                signal_type=SignalType.BUY,
                symbol="",
                price=Decimal(str(candles[-1].close)),
                confidence=avg_confidence,
                strategy_name="Consensus",
                reason=f"Konsens ({len(buy_signals)}/{total}): " + " | ".join(reasons[:2]),
                stop_loss=stop_loss,
            )

        # Sell Konsens?
        elif len(sell_signals) >= total * min_agreement:
            avg_confidence = statistics.mean(s.confidence for s in sell_signals)
            reasons = [s.reason for s in sell_signals]

            return TradingSignal(
                timestamp=candles[-1].timestamp,
                signal_type=SignalType.CLOSE,
                symbol="",
                price=Decimal(str(candles[-1].close)),
                confidence=avg_confidence,
                strategy_name="Consensus",
                reason=f"Konsens ({len(sell_signals)}/{total}): " + " | ".join(reasons[:2]),
            )

        return None


def create_default_strategy_manager() -> StrategyManager:
    """Erstellt Manager mit Standard-Strategien."""
    manager = StrategyManager()
    manager.add_strategy(RSIMeanReversionStrategy())
    manager.add_strategy(MACrossoverStrategy())
    manager.add_strategy(BollingerBandStrategy())
    return manager


if __name__ == "__main__":
    from datetime import timedelta
    import random

    print("=== TRADING STRATEGIES TEST ===\n")

    # Generiere Test-Daten
    candles = []
    price = 100.0
    for i in range(100):
        change = random.uniform(-0.02, 0.02)
        o = price
        c = price * (1 + change)
        h = max(o, c) * 1.01
        l = min(o, c) * 0.99

        candles.append(PriceData(
            timestamp=datetime.now() - timedelta(days=100 - i),
            open=o, high=h, low=l, close=c,
            volume=random.uniform(1e6, 5e6),
        ))
        price = c

    # Test einzelne Strategien
    rsi_strategy = RSIMeanReversionStrategy()
    signal = rsi_strategy.generate_signal(candles)
    print(f"RSI Strategy: {signal.signal_type.value if signal else 'HOLD'}")

    ma_strategy = MACrossoverStrategy()
    signal = ma_strategy.generate_signal(candles)
    print(f"MA Strategy: {signal.signal_type.value if signal else 'HOLD'}")

    # Test Konsens
    manager = create_default_strategy_manager()
    consensus = manager.get_consensus_signal(candles)
    print(f"\nKonsens: {consensus.signal_type.value if consensus else 'HOLD'}")
    if consensus:
        print(f"  Grund: {consensus.reason}")
        print(f"  Konfidenz: {consensus.confidence:.0%}")
