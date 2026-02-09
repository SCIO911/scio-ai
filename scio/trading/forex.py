"""
SCIO Forex Module

Devisenhandel und Waehrungsanalyse:
- Echtzeit-Kurse
- Technische Analyse
- Carry Trade Berechnung
- Waehrungspaar-Korrelationen
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import urllib.request
import hashlib

from scio.core.logging import get_logger

logger = get_logger(__name__)


class CurrencyPair(str, Enum):
    """Wichtigste Waehrungspaare."""
    EURUSD = "EUR/USD"
    GBPUSD = "GBP/USD"
    USDJPY = "USD/JPY"
    USDCHF = "USD/CHF"
    AUDUSD = "AUD/USD"
    USDCAD = "USD/CAD"
    NZDUSD = "NZD/USD"
    EURGBP = "EUR/GBP"
    EURJPY = "EUR/JPY"
    GBPJPY = "GBP/JPY"


@dataclass
class ForexQuote:
    """Ein Forex-Kurs."""
    pair: str
    bid: Decimal
    ask: Decimal
    timestamp: datetime
    change_24h: float = 0.0
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None

    @property
    def mid(self) -> Decimal:
        """Mittelkurs."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Spread in Pips."""
        return (self.ask - self.bid) * 10000

    @property
    def spread_percent(self) -> float:
        """Spread in Prozent."""
        return float((self.ask - self.bid) / self.mid * 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "bid": str(self.bid),
            "ask": str(self.ask),
            "mid": str(self.mid),
            "spread_pips": float(self.spread),
            "spread_percent": self.spread_percent,
            "change_24h": self.change_24h,
            "high_24h": str(self.high_24h) if self.high_24h else None,
            "low_24h": str(self.low_24h) if self.low_24h else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CarryTradeAnalysis:
    """Carry Trade Analyse."""
    pair: str
    base_currency: str
    quote_currency: str
    base_interest_rate: float
    quote_interest_rate: float
    interest_differential: float
    annual_carry_percent: float
    daily_carry_pips: float
    risk_assessment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "base_interest_rate": self.base_interest_rate,
            "quote_interest_rate": self.quote_interest_rate,
            "interest_differential": self.interest_differential,
            "annual_carry_percent": self.annual_carry_percent,
            "daily_carry_pips": self.daily_carry_pips,
            "risk_assessment": self.risk_assessment,
        }


# Aktuelle Zentralbank-Zinsen (Stand: 2024, sollte regelmaessig aktualisiert werden)
CENTRAL_BANK_RATES = {
    "USD": 5.25,   # Federal Reserve
    "EUR": 4.50,   # ECB
    "GBP": 5.25,   # Bank of England
    "JPY": 0.10,   # Bank of Japan
    "CHF": 1.75,   # SNB
    "AUD": 4.35,   # RBA
    "CAD": 5.00,   # Bank of Canada
    "NZD": 5.50,   # RBNZ
    "SEK": 4.00,   # Riksbank
    "NOK": 4.50,   # Norges Bank
}


class ForexRateFetcher:
    """
    Holt Forex-Kurse.
    Nutzt freie API (mit Limitierungen).
    """

    # Freie API ohne Key
    FRANKFURTER_API = "https://api.frankfurter.app"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".scio" / "forex_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)

    def _fetch_json(self, url: str) -> Optional[dict]:
        """Holt JSON mit Caching."""
        cache_key = hashlib.md5(url.encode()).hexdigest()

        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return data

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SCIO/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                self._cache[cache_key] = (data, datetime.now())
                return data
        except Exception as e:
            logger.warning("Failed to fetch forex data", url=url, error=str(e))
            return None

    def get_rate(self, base: str, quote: str) -> Optional[ForexQuote]:
        """
        Holt aktuellen Wechselkurs.

        Args:
            base: Basiswaehrung (z.B. "EUR")
            quote: Quotierungswaehrung (z.B. "USD")
        """
        url = f"{self.FRANKFURTER_API}/latest?from={base}&to={quote}"
        data = self._fetch_json(url)

        if not data or "rates" not in data:
            return None

        rate = data["rates"].get(quote)
        if not rate:
            return None

        # Simuliere Bid/Ask (echte Spreads waeren von Broker)
        mid = Decimal(str(rate))
        spread = mid * Decimal("0.0002")  # Typischer Spread
        bid = mid - spread / 2
        ask = mid + spread / 2

        return ForexQuote(
            pair=f"{base}/{quote}",
            bid=bid,
            ask=ask,
            timestamp=datetime.now(),
        )

    def get_multiple_rates(self, base: str, quotes: list[str]) -> dict[str, ForexQuote]:
        """Holt mehrere Kurse auf einmal."""
        quotes_str = ",".join(quotes)
        url = f"{self.FRANKFURTER_API}/latest?from={base}&to={quotes_str}"
        data = self._fetch_json(url)

        if not data or "rates" not in data:
            return {}

        results = {}
        for quote, rate in data["rates"].items():
            mid = Decimal(str(rate))
            spread = mid * Decimal("0.0002")

            results[quote] = ForexQuote(
                pair=f"{base}/{quote}",
                bid=mid - spread / 2,
                ask=mid + spread / 2,
                timestamp=datetime.now(),
            )

        return results

    def get_historical_rate(
        self,
        base: str,
        quote: str,
        date: datetime,
    ) -> Optional[Decimal]:
        """Holt historischen Kurs."""
        date_str = date.strftime("%Y-%m-%d")
        url = f"{self.FRANKFURTER_API}/{date_str}?from={base}&to={quote}"
        data = self._fetch_json(url)

        if not data or "rates" not in data:
            return None

        rate = data["rates"].get(quote)
        return Decimal(str(rate)) if rate else None

    def get_rate_series(
        self,
        base: str,
        quote: str,
        days: int = 30,
    ) -> list[tuple[datetime, Decimal]]:
        """Holt Kursverlauf."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        url = (
            f"{self.FRANKFURTER_API}/{start_date.strftime('%Y-%m-%d')}.."
            f"{end_date.strftime('%Y-%m-%d')}?from={base}&to={quote}"
        )
        data = self._fetch_json(url)

        if not data or "rates" not in data:
            return []

        series = []
        for date_str, rates in sorted(data["rates"].items()):
            if quote in rates:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                series.append((dt, Decimal(str(rates[quote]))))

        return series


class ForexAnalyzer:
    """
    Forex-Analyse.
    """

    def __init__(self):
        self.fetcher = ForexRateFetcher()

    def analyze_pair(self, base: str, quote: str) -> dict[str, Any]:
        """Vollstaendige Analyse eines Waehrungspaars."""
        # Aktueller Kurs
        current = self.fetcher.get_rate(base, quote)
        if not current:
            return {"error": f"Could not fetch {base}/{quote}"}

        # Historische Daten
        series = self.fetcher.get_rate_series(base, quote, days=30)

        analysis = {
            "pair": f"{base}/{quote}",
            "current_rate": current.to_dict(),
            "base_interest_rate": CENTRAL_BANK_RATES.get(base, 0),
            "quote_interest_rate": CENTRAL_BANK_RATES.get(quote, 0),
        }

        # Trend berechnen
        if len(series) >= 7:
            week_ago = series[-7][1]
            current_rate = series[-1][1]
            week_change = float((current_rate - week_ago) / week_ago * 100)

            if week_change > 1:
                trend = f"bullish ({base} staerkt sich)"
            elif week_change < -1:
                trend = f"bearish ({base} schwaecht sich)"
            else:
                trend = "neutral"

            analysis["trend"] = trend
            analysis["week_change_percent"] = week_change

        # Volatilitaet
        if len(series) >= 14:
            import statistics
            returns = []
            for i in range(1, len(series)):
                ret = float((series[i][1] - series[i - 1][1]) / series[i - 1][1])
                returns.append(ret)

            volatility = statistics.stdev(returns) * (252 ** 0.5) * 100  # Annualisiert
            analysis["volatility_annual_percent"] = volatility

        # Carry Trade Analyse
        carry = self.analyze_carry_trade(base, quote)
        analysis["carry_trade"] = carry.to_dict()

        return analysis

    def analyze_carry_trade(self, base: str, quote: str) -> CarryTradeAnalysis:
        """
        Analysiert Carry Trade Potenzial.

        Carry Trade: Leihe in Niedrigzins-Waehrung, investiere in Hochzins-Waehrung.
        """
        base_rate = CENTRAL_BANK_RATES.get(base, 0)
        quote_rate = CENTRAL_BANK_RATES.get(quote, 0)
        differential = base_rate - quote_rate

        # Taeglicher Carry in Pips (vereinfacht)
        daily_carry = differential / 365 * 100  # In Pips

        # Risikobewertung
        if abs(differential) < 1:
            risk = "Niedriges Carry-Potenzial, geringes Risiko"
        elif differential > 3:
            risk = "Hohes Carry Long {}, aber Waehrungsrisiko beachten".format(base)
        elif differential < -3:
            risk = "Hohes Carry Short {}, aber Waehrungsrisiko beachten".format(base)
        else:
            risk = "Moderates Carry-Potenzial"

        return CarryTradeAnalysis(
            pair=f"{base}/{quote}",
            base_currency=base,
            quote_currency=quote,
            base_interest_rate=base_rate,
            quote_interest_rate=quote_rate,
            interest_differential=differential,
            annual_carry_percent=differential,
            daily_carry_pips=daily_carry,
            risk_assessment=risk,
        )

    def find_best_carry_trades(self) -> list[CarryTradeAnalysis]:
        """Findet die besten Carry Trade Moeglichkeiten."""
        currencies = list(CENTRAL_BANK_RATES.keys())
        trades = []

        for base in currencies:
            for quote in currencies:
                if base != quote:
                    carry = self.analyze_carry_trade(base, quote)
                    if carry.interest_differential > 2:
                        trades.append(carry)

        # Sortiere nach Differential
        trades.sort(key=lambda x: x.interest_differential, reverse=True)
        return trades[:10]

    def calculate_pip_value(
        self,
        pair: str,
        lot_size: Decimal = Decimal("100000"),  # Standard Lot
        account_currency: str = "EUR",
    ) -> dict[str, Any]:
        """
        Berechnet Pip-Wert.

        Args:
            pair: Waehrungspaar (z.B. "EUR/USD")
            lot_size: Positionsgroesse
            account_currency: Kontowährung
        """
        base, quote = pair.split("/")

        # Ein Pip ist 0.0001 (ausser JPY-Paare: 0.01)
        if "JPY" in pair:
            pip_size = Decimal("0.01")
        else:
            pip_size = Decimal("0.0001")

        # Pip-Wert in Quotierungswaehrung
        pip_value_quote = lot_size * pip_size

        # Konvertiere zu Kontowährung falls noetig
        if quote == account_currency:
            pip_value_account = pip_value_quote
        else:
            conversion = self.fetcher.get_rate(quote, account_currency)
            if conversion:
                pip_value_account = pip_value_quote * conversion.mid
            else:
                pip_value_account = pip_value_quote  # Fallback

        return {
            "pair": pair,
            "lot_size": str(lot_size),
            "pip_size": str(pip_size),
            "pip_value_quote": str(pip_value_quote),
            "pip_value_account": str(pip_value_account),
            "account_currency": account_currency,
        }


class ForexConverter:
    """
    Waehrungsrechner.
    """

    def __init__(self):
        self.fetcher = ForexRateFetcher()

    def convert(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
    ) -> dict[str, Any]:
        """Konvertiert Betrag."""
        if from_currency == to_currency:
            return {
                "amount": str(amount),
                "from": from_currency,
                "to": to_currency,
                "rate": "1",
                "result": str(amount),
            }

        quote = self.fetcher.get_rate(from_currency, to_currency)
        if not quote:
            return {"error": f"Could not get rate for {from_currency}/{to_currency}"}

        result = amount * quote.mid

        return {
            "amount": str(amount),
            "from": from_currency,
            "to": to_currency,
            "rate": str(quote.mid),
            "bid_rate": str(quote.bid),
            "ask_rate": str(quote.ask),
            "result": str(result),
            "result_bid": str(amount * quote.bid),
            "result_ask": str(amount * quote.ask),
        }

    def convert_multiple(
        self,
        amount: Decimal,
        from_currency: str,
        to_currencies: list[str],
    ) -> dict[str, Any]:
        """Konvertiert zu mehreren Waehrungen."""
        quotes = self.fetcher.get_multiple_rates(from_currency, to_currencies)

        results = {
            "amount": str(amount),
            "from": from_currency,
            "conversions": {},
        }

        for currency, quote in quotes.items():
            results["conversions"][currency] = {
                "rate": str(quote.mid),
                "result": str(amount * quote.mid),
            }

        return results


def get_forex_dashboard() -> dict[str, Any]:
    """
    Gibt Forex-Dashboard mit wichtigsten Kursen zurueck.
    """
    fetcher = ForexRateFetcher()
    analyzer = ForexAnalyzer()

    # Hole wichtigste Kurse
    eur_rates = fetcher.get_multiple_rates("EUR", ["USD", "GBP", "JPY", "CHF"])
    usd_rates = fetcher.get_multiple_rates("USD", ["JPY", "CAD", "AUD"])

    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "eur_rates": {k: v.to_dict() for k, v in eur_rates.items()},
        "usd_rates": {k: v.to_dict() for k, v in usd_rates.items()},
        "central_bank_rates": CENTRAL_BANK_RATES,
        "top_carry_trades": [t.to_dict() for t in analyzer.find_best_carry_trades()[:5]],
    }

    return dashboard


if __name__ == "__main__":
    print("=== FOREX MODULE TEST ===\n")

    fetcher = ForexRateFetcher()
    analyzer = ForexAnalyzer()
    converter = ForexConverter()

    # Test Kursabfrage
    print("--- EUR/USD Kurs ---")
    eurusd = fetcher.get_rate("EUR", "USD")
    if eurusd:
        print(f"Bid: {eurusd.bid}")
        print(f"Ask: {eurusd.ask}")
        print(f"Spread: {eurusd.spread:.1f} Pips")

    # Test Analyse
    print("\n--- EUR/USD Analyse ---")
    analysis = analyzer.analyze_pair("EUR", "USD")
    print(f"Trend: {analysis.get('trend', 'N/A')}")
    print(f"Carry: {analysis['carry_trade']['interest_differential']:.2f}%")

    # Test Konvertierung
    print("\n--- Konvertierung ---")
    result = converter.convert(Decimal("1000"), "EUR", "USD")
    print(f"1000 EUR = {result.get('result', 'N/A')} USD")

    # Test Carry Trades
    print("\n--- Top Carry Trades ---")
    for carry in analyzer.find_best_carry_trades()[:3]:
        print(f"  {carry.pair}: {carry.interest_differential:.2f}% Differential")
