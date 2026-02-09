"""
SCIO Crypto Trading Module

Vollstaendiges Krypto-Handelssystem mit:
- Preisabfrage (CoinGecko, Binance)
- Technische Analyse
- DeFi Integration
- Staking Tracking
- Portfolio Management
"""

import json
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import urllib.request
import urllib.error

from scio.core.logging import get_logger

logger = get_logger(__name__)


class CryptoExchange(str, Enum):
    """Unterstuetzte Boersen."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    COINGECKO = "coingecko"  # Aggregator


class NetworkType(str, Enum):
    """Blockchain Netzwerke."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    SOLANA = "solana"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    AVALANCHE = "avalanche"
    BSC = "bsc"


@dataclass
class CryptoAsset:
    """Ein Krypto-Asset."""
    symbol: str
    name: str
    price_usd: Decimal
    price_eur: Decimal
    market_cap: Decimal
    volume_24h: Decimal
    change_24h: float
    change_7d: float
    circulating_supply: Decimal
    max_supply: Optional[Decimal]
    rank: int
    last_updated: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price_usd": str(self.price_usd),
            "price_eur": str(self.price_eur),
            "market_cap": str(self.market_cap),
            "volume_24h": str(self.volume_24h),
            "change_24h": self.change_24h,
            "change_7d": self.change_7d,
            "rank": self.rank,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class StakingPosition:
    """Eine Staking-Position."""
    asset: str
    amount: Decimal
    apy: float  # Annual Percentage Yield
    network: NetworkType
    validator: Optional[str]
    start_date: datetime
    rewards_earned: Decimal = Decimal("0")
    locked_until: Optional[datetime] = None

    @property
    def daily_reward(self) -> Decimal:
        """Taegliche Belohnung."""
        daily_rate = Decimal(str(self.apy / 365 / 100))
        return self.amount * daily_rate

    @property
    def monthly_reward(self) -> Decimal:
        """Monatliche Belohnung."""
        return self.daily_reward * 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "amount": str(self.amount),
            "apy": self.apy,
            "network": self.network.value,
            "validator": self.validator,
            "start_date": self.start_date.isoformat(),
            "rewards_earned": str(self.rewards_earned),
            "daily_reward": str(self.daily_reward),
            "monthly_reward": str(self.monthly_reward),
        }


@dataclass
class DeFiPosition:
    """Eine DeFi-Position (Liquidity, Lending, etc.)."""
    protocol: str  # z.B. "Aave", "Uniswap", "Compound"
    position_type: str  # "lending", "liquidity", "farming"
    assets: list[str]
    amounts: list[Decimal]
    apy: float
    network: NetworkType
    tvl: Decimal  # Total Value Locked
    impermanent_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "position_type": self.position_type,
            "assets": self.assets,
            "amounts": [str(a) for a in self.amounts],
            "apy": self.apy,
            "network": self.network.value,
            "tvl": str(self.tvl),
            "impermanent_loss": self.impermanent_loss,
        }


class CryptoPriceFetcher:
    """
    Holt Krypto-Preise von verschiedenen APIs.
    """

    COINGECKO_API = "https://api.coingecko.com/api/v3"

    # Populaere Coins mit CoinGecko IDs
    COIN_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "MATIC": "matic-network",
        "DOT": "polkadot",
        "AVAX": "avalanche-2",
        "LINK": "chainlink",
        "UNI": "uniswap",
        "ATOM": "cosmos",
        "LTC": "litecoin",
        "NEAR": "near",
        "ARB": "arbitrum",
        "OP": "optimism",
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".scio" / "crypto_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def _fetch_json(self, url: str) -> Optional[dict]:
        """Holt JSON von URL mit Caching."""
        cache_key = hashlib.md5(url.encode()).hexdigest()

        # Cache pruefen
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return data

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SCIO/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                self._cache[cache_key] = (data, datetime.now())
                return data
        except Exception as e:
            logger.warning("Failed to fetch crypto data", url=url, error=str(e))
            return None

    def get_price(self, symbol: str) -> Optional[CryptoAsset]:
        """Holt aktuellen Preis fuer ein Symbol."""
        coin_id = self.COIN_IDS.get(symbol.upper())
        if not coin_id:
            logger.warning("Unknown crypto symbol", symbol=symbol)
            return None

        url = f"{self.COINGECKO_API}/coins/{coin_id}?localization=false&sparkline=false"
        data = self._fetch_json(url)

        if not data:
            return None

        try:
            market = data.get("market_data", {})
            return CryptoAsset(
                symbol=symbol.upper(),
                name=data.get("name", ""),
                price_usd=Decimal(str(market.get("current_price", {}).get("usd", 0))),
                price_eur=Decimal(str(market.get("current_price", {}).get("eur", 0))),
                market_cap=Decimal(str(market.get("market_cap", {}).get("usd", 0))),
                volume_24h=Decimal(str(market.get("total_volume", {}).get("usd", 0))),
                change_24h=market.get("price_change_percentage_24h", 0) or 0,
                change_7d=market.get("price_change_percentage_7d", 0) or 0,
                circulating_supply=Decimal(str(market.get("circulating_supply", 0) or 0)),
                max_supply=Decimal(str(market.get("max_supply"))) if market.get("max_supply") else None,
                rank=data.get("market_cap_rank", 0) or 0,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error("Failed to parse crypto data", symbol=symbol, error=str(e))
            return None

    def get_top_coins(self, limit: int = 20) -> list[CryptoAsset]:
        """Holt Top-Coins nach Marktkapitalisierung."""
        url = f"{self.COINGECKO_API}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={limit}&page=1"
        data = self._fetch_json(url)

        if not data:
            return []

        coins = []
        for item in data:
            try:
                coins.append(CryptoAsset(
                    symbol=item.get("symbol", "").upper(),
                    name=item.get("name", ""),
                    price_usd=Decimal(str(item.get("current_price", 0) or 0)),
                    price_eur=Decimal(str(item.get("current_price", 0) or 0)) * Decimal("0.92"),
                    market_cap=Decimal(str(item.get("market_cap", 0) or 0)),
                    volume_24h=Decimal(str(item.get("total_volume", 0) or 0)),
                    change_24h=item.get("price_change_percentage_24h", 0) or 0,
                    change_7d=item.get("price_change_percentage_7d_in_currency", 0) or 0,
                    circulating_supply=Decimal(str(item.get("circulating_supply", 0) or 0)),
                    max_supply=Decimal(str(item.get("max_supply"))) if item.get("max_supply") else None,
                    rank=item.get("market_cap_rank", 0) or 0,
                    last_updated=datetime.now(),
                ))
            except Exception:
                continue

        return coins

    def get_historical_prices(
        self,
        symbol: str,
        days: int = 30,
    ) -> list[tuple[datetime, Decimal]]:
        """Holt historische Preise."""
        coin_id = self.COIN_IDS.get(symbol.upper())
        if not coin_id:
            return []

        url = f"{self.COINGECKO_API}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
        data = self._fetch_json(url)

        if not data:
            return []

        prices = []
        for timestamp, price in data.get("prices", []):
            dt = datetime.fromtimestamp(timestamp / 1000)
            prices.append((dt, Decimal(str(price))))

        return prices


class CryptoPortfolio:
    """
    Verwaltet ein Krypto-Portfolio.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".scio" / "crypto_portfolio.json"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fetcher = CryptoPriceFetcher()

        self._holdings: dict[str, Decimal] = {}
        self._cost_basis: dict[str, Decimal] = {}
        self._staking: list[StakingPosition] = []
        self._defi: list[DeFiPosition] = []

        self._load()

    def _load(self) -> None:
        """Laedt Portfolio von Disk."""
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self._holdings = {k: Decimal(v) for k, v in data.get("holdings", {}).items()}
                self._cost_basis = {k: Decimal(v) for k, v in data.get("cost_basis", {}).items()}
            except Exception as e:
                logger.error("Failed to load portfolio", error=str(e))

    def _save(self) -> None:
        """Speichert Portfolio."""
        data = {
            "holdings": {k: str(v) for k, v in self._holdings.items()},
            "cost_basis": {k: str(v) for k, v in self._cost_basis.items()},
            "staking": [s.to_dict() for s in self._staking],
            "defi": [d.to_dict() for d in self._defi],
            "last_updated": datetime.now().isoformat(),
        }
        self.db_path.write_text(json.dumps(data, indent=2))

    def add_holding(
        self,
        symbol: str,
        amount: Decimal,
        cost_usd: Decimal,
    ) -> None:
        """Fuegt eine Position hinzu."""
        symbol = symbol.upper()
        self._holdings[symbol] = self._holdings.get(symbol, Decimal("0")) + amount
        self._cost_basis[symbol] = self._cost_basis.get(symbol, Decimal("0")) + cost_usd
        self._save()
        logger.info("Holding added", symbol=symbol, amount=str(amount))

    def remove_holding(
        self,
        symbol: str,
        amount: Decimal,
    ) -> bool:
        """Entfernt eine Position."""
        symbol = symbol.upper()
        if symbol not in self._holdings:
            return False

        if self._holdings[symbol] < amount:
            return False

        self._holdings[symbol] -= amount
        if self._holdings[symbol] == 0:
            del self._holdings[symbol]
            del self._cost_basis[symbol]

        self._save()
        return True

    def add_staking(
        self,
        asset: str,
        amount: Decimal,
        apy: float,
        network: NetworkType,
        validator: Optional[str] = None,
    ) -> StakingPosition:
        """Fuegt Staking-Position hinzu."""
        position = StakingPosition(
            asset=asset.upper(),
            amount=amount,
            apy=apy,
            network=network,
            validator=validator,
            start_date=datetime.now(),
        )
        self._staking.append(position)
        self._save()
        return position

    def get_value(self) -> dict[str, Any]:
        """Berechnet aktuellen Portfolio-Wert."""
        total_value_usd = Decimal("0")
        total_cost = Decimal("0")
        positions = []

        for symbol, amount in self._holdings.items():
            asset = self.fetcher.get_price(symbol)
            if asset:
                value = amount * asset.price_usd
                cost = self._cost_basis.get(symbol, Decimal("0"))
                pnl = value - cost
                pnl_percent = (pnl / cost * 100) if cost > 0 else Decimal("0")

                total_value_usd += value
                total_cost += cost

                positions.append({
                    "symbol": symbol,
                    "amount": str(amount),
                    "price_usd": str(asset.price_usd),
                    "value_usd": str(value),
                    "cost_basis": str(cost),
                    "pnl_usd": str(pnl),
                    "pnl_percent": float(pnl_percent),
                    "change_24h": asset.change_24h,
                })

        total_pnl = total_value_usd - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else Decimal("0")

        # Staking-Wert
        staking_value = Decimal("0")
        staking_daily_rewards = Decimal("0")
        for stake in self._staking:
            asset = self.fetcher.get_price(stake.asset)
            if asset:
                staking_value += stake.amount * asset.price_usd
                staking_daily_rewards += stake.daily_reward * asset.price_usd

        return {
            "total_value_usd": str(total_value_usd),
            "total_cost_usd": str(total_cost),
            "total_pnl_usd": str(total_pnl),
            "total_pnl_percent": float(total_pnl_percent),
            "positions": positions,
            "staking_value_usd": str(staking_value),
            "staking_daily_rewards_usd": str(staking_daily_rewards),
            "staking_positions": len(self._staking),
            "defi_positions": len(self._defi),
            "last_updated": datetime.now().isoformat(),
        }

    def get_allocation(self) -> dict[str, float]:
        """Berechnet Portfolio-Allokation."""
        total = Decimal("0")
        values = {}

        for symbol, amount in self._holdings.items():
            asset = self.fetcher.get_price(symbol)
            if asset:
                value = amount * asset.price_usd
                values[symbol] = value
                total += value

        if total == 0:
            return {}

        return {symbol: float(value / total * 100) for symbol, value in values.items()}


class CryptoAnalyzer:
    """
    Analysiert Krypto-Maerkte.
    """

    def __init__(self):
        self.fetcher = CryptoPriceFetcher()

    def analyze_coin(self, symbol: str) -> dict[str, Any]:
        """Vollstaendige Analyse eines Coins."""
        asset = self.fetcher.get_price(symbol)
        if not asset:
            return {"error": f"Could not fetch data for {symbol}"}

        # Historische Daten fuer Analyse
        prices = self.fetcher.get_historical_prices(symbol, days=30)

        analysis = {
            "symbol": symbol,
            "current_price_usd": str(asset.price_usd),
            "market_cap_rank": asset.rank,
            "market_cap_usd": str(asset.market_cap),
        }

        # Trend-Analyse
        if len(prices) > 7:
            week_ago_price = prices[-7][1]
            current_price = prices[-1][1]
            week_change = float((current_price - week_ago_price) / week_ago_price * 100)

            if week_change > 10:
                trend = "strong_bullish"
            elif week_change > 3:
                trend = "bullish"
            elif week_change < -10:
                trend = "strong_bearish"
            elif week_change < -3:
                trend = "bearish"
            else:
                trend = "neutral"

            analysis["trend"] = trend
            analysis["week_change_percent"] = week_change

        # Volatilitaet
        if len(prices) > 14:
            import statistics
            recent_prices = [float(p[1]) for p in prices[-14:]]
            volatility = statistics.stdev(recent_prices) / statistics.mean(recent_prices) * 100
            analysis["volatility_14d"] = volatility

            if volatility > 10:
                analysis["volatility_level"] = "high"
            elif volatility > 5:
                analysis["volatility_level"] = "medium"
            else:
                analysis["volatility_level"] = "low"

        # Support/Resistance (vereinfacht)
        if len(prices) > 20:
            price_values = [float(p[1]) for p in prices[-20:]]
            analysis["support_estimate"] = min(price_values)
            analysis["resistance_estimate"] = max(price_values)

        # Empfehlung
        analysis["recommendation"] = self._generate_recommendation(analysis)

        return analysis

    def _generate_recommendation(self, analysis: dict) -> str:
        """Generiert Empfehlung basierend auf Analyse."""
        parts = []

        trend = analysis.get("trend", "neutral")
        if trend in ["strong_bullish", "bullish"]:
            parts.append("Trend positiv")
        elif trend in ["strong_bearish", "bearish"]:
            parts.append("Trend negativ - Vorsicht")

        vol = analysis.get("volatility_level", "medium")
        if vol == "high":
            parts.append("Hohe Volatilitaet - nur mit Stop-Loss")

        if not parts:
            return "Neutral - keine klare Richtung"

        return " | ".join(parts)

    def compare_coins(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Vergleicht mehrere Coins."""
        comparisons = []

        for symbol in symbols:
            analysis = self.analyze_coin(symbol)
            if "error" not in analysis:
                comparisons.append(analysis)

        # Sortiere nach Performance
        comparisons.sort(
            key=lambda x: x.get("week_change_percent", 0),
            reverse=True,
        )

        return comparisons

    def find_opportunities(self) -> list[dict[str, Any]]:
        """Findet potenzielle Trading-Gelegenheiten."""
        opportunities = []

        for symbol in list(CryptoPriceFetcher.COIN_IDS.keys())[:10]:
            analysis = self.analyze_coin(symbol)
            if "error" in analysis:
                continue

            # Ueberverkauft?
            if analysis.get("week_change_percent", 0) < -15:
                opportunities.append({
                    "symbol": symbol,
                    "type": "oversold_bounce",
                    "reason": f"Starker Rueckgang ({analysis['week_change_percent']:.1f}%)",
                    "risk": "high",
                })

            # Starkes Momentum?
            elif analysis.get("trend") == "strong_bullish" and analysis.get("volatility_level") != "high":
                opportunities.append({
                    "symbol": symbol,
                    "type": "momentum",
                    "reason": "Starker Aufwaertstrend mit moderater Volatilitaet",
                    "risk": "medium",
                })

        return opportunities


# Staking APY Daten (typische Werte)
STAKING_APYS = {
    "ETH": 4.0,
    "SOL": 7.0,
    "ADA": 5.0,
    "DOT": 12.0,
    "ATOM": 15.0,
    "AVAX": 8.0,
    "NEAR": 10.0,
    "MATIC": 5.0,
}


def calculate_staking_rewards(
    symbol: str,
    amount: Decimal,
    days: int = 365,
) -> dict[str, Any]:
    """Berechnet erwartete Staking-Rewards."""
    apy = STAKING_APYS.get(symbol.upper(), 0)
    if apy == 0:
        return {"error": f"{symbol} does not support staking or APY unknown"}

    daily_rate = Decimal(str(apy / 365 / 100))
    daily_reward = amount * daily_rate
    total_reward = daily_reward * days

    fetcher = CryptoPriceFetcher()
    asset = fetcher.get_price(symbol)
    price_usd = asset.price_usd if asset else Decimal("0")

    return {
        "symbol": symbol.upper(),
        "staked_amount": str(amount),
        "apy_percent": apy,
        "days": days,
        "daily_reward": str(daily_reward),
        "total_reward": str(total_reward),
        "daily_reward_usd": str(daily_reward * price_usd),
        "total_reward_usd": str(total_reward * price_usd),
        "current_price_usd": str(price_usd),
    }


if __name__ == "__main__":
    # Demo
    print("=== CRYPTO MODULE TEST ===\n")

    fetcher = CryptoPriceFetcher()

    print("Top 5 Coins:")
    for coin in fetcher.get_top_coins(5):
        print(f"  {coin.rank}. {coin.symbol}: ${coin.price_usd:.2f} ({coin.change_24h:+.1f}%)")

    print("\n--- Bitcoin Analyse ---")
    analyzer = CryptoAnalyzer()
    btc = analyzer.analyze_coin("BTC")
    print(f"Preis: ${btc.get('current_price_usd')}")
    print(f"Trend: {btc.get('trend')}")
    print(f"Empfehlung: {btc.get('recommendation')}")

    print("\n--- Staking Rewards (10 ETH) ---")
    rewards = calculate_staking_rewards("ETH", Decimal("10"), 365)
    print(f"APY: {rewards['apy_percent']}%")
    print(f"Jaehrlich: {rewards['total_reward']} ETH (${rewards['total_reward_usd']})")
