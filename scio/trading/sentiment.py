"""
SCIO Sentiment Analyzer

Market Sentiment Analyse:
- Fear & Greed Index
- Social Media Sentiment
- News Sentiment
- On-Chain Sentiment (Crypto)
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import urllib.request

from scio.core.logging import get_logger

logger = get_logger(__name__)


class SentimentLevel(str, Enum):
    """Sentiment-Stufen."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"

    @classmethod
    def from_score(cls, score: float) -> "SentimentLevel":
        """Konvertiert Score (0-100) zu Level."""
        if score <= 20:
            return cls.EXTREME_FEAR
        elif score <= 40:
            return cls.FEAR
        elif score <= 60:
            return cls.NEUTRAL
        elif score <= 80:
            return cls.GREED
        else:
            return cls.EXTREME_GREED


@dataclass
class SentimentData:
    """Sentiment-Daten."""
    source: str
    score: float  # 0-100 (0=extreme fear, 100=extreme greed)
    level: SentimentLevel
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "score": self.score,
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class NewsItem:
    """Eine Nachricht."""
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float  # -1 (negativ) bis +1 (positiv)
    relevance: float  # 0-1
    keywords: list[str] = field(default_factory=list)

    @property
    def sentiment_label(self) -> str:
        if self.sentiment_score > 0.3:
            return "positive"
        elif self.sentiment_score < -0.3:
            return "negative"
        else:
            return "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "relevance": self.relevance,
            "keywords": self.keywords,
        }


@dataclass
class MarketSentiment:
    """Gesamtes Market Sentiment."""
    overall_score: float
    overall_level: SentimentLevel
    fear_greed_index: Optional[SentimentData]
    social_sentiment: Optional[SentimentData]
    news_sentiment: Optional[SentimentData]
    technical_sentiment: Optional[SentimentData]
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.value,
            "fear_greed_index": self.fear_greed_index.to_dict() if self.fear_greed_index else None,
            "social_sentiment": self.social_sentiment.to_dict() if self.social_sentiment else None,
            "news_sentiment": self.news_sentiment.to_dict() if self.news_sentiment else None,
            "technical_sentiment": self.technical_sentiment.to_dict() if self.technical_sentiment else None,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


class FearGreedIndexFetcher:
    """
    Holt Fear & Greed Index Daten.
    """

    CRYPTO_FGI_API = "https://api.alternative.me/fng/"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".scio" / "sentiment_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)

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
            logger.warning("Failed to fetch sentiment data", url=url, error=str(e))
            return None

    def get_crypto_fear_greed(self) -> Optional[SentimentData]:
        """Holt Crypto Fear & Greed Index."""
        data = self._fetch_json(self.CRYPTO_FGI_API)

        if not data or "data" not in data:
            return None

        try:
            fgi_data = data["data"][0]
            score = float(fgi_data["value"])
            classification = fgi_data["value_classification"]

            return SentimentData(
                source="crypto_fear_greed_index",
                score=score,
                level=SentimentLevel.from_score(score),
                timestamp=datetime.now(),
                details={
                    "classification": classification,
                    "time_until_update": fgi_data.get("time_until_update"),
                },
            )
        except Exception as e:
            logger.error("Failed to parse FGI data", error=str(e))
            return None

    def get_historical_fgi(self, days: int = 30) -> list[SentimentData]:
        """Holt historische FGI Daten."""
        url = f"{self.CRYPTO_FGI_API}?limit={days}"
        data = self._fetch_json(url)

        if not data or "data" not in data:
            return []

        results = []
        for item in data["data"]:
            try:
                score = float(item["value"])
                timestamp = datetime.fromtimestamp(int(item["timestamp"]))

                results.append(SentimentData(
                    source="crypto_fear_greed_index",
                    score=score,
                    level=SentimentLevel.from_score(score),
                    timestamp=timestamp,
                    details={"classification": item["value_classification"]},
                ))
            except Exception:
                continue

        return results


class TextSentimentAnalyzer:
    """
    Einfacher regelbasierter Text-Sentiment-Analyzer.
    """

    # Positive Woerter
    POSITIVE_WORDS = {
        "bullish", "bull", "moon", "pump", "surge", "rally", "gain", "profit",
        "growth", "rise", "rising", "up", "high", "record", "ath", "breakout",
        "strong", "strength", "buy", "long", "accumulate", "hodl", "hold",
        "green", "positive", "optimistic", "confidence", "success", "win",
        "upgrade", "approved", "adoption", "institutional", "partnership",
        "innovation", "revolutionary", "breakthrough", "milestone",
    }

    # Negative Woerter
    NEGATIVE_WORDS = {
        "bearish", "bear", "crash", "dump", "plunge", "drop", "fall", "loss",
        "decline", "down", "low", "sell", "short", "liquidation", "liquidated",
        "red", "negative", "pessimistic", "fear", "panic", "risk", "warning",
        "hack", "hacked", "scam", "fraud", "ban", "banned", "regulation",
        "investigation", "lawsuit", "sec", "fine", "penalty", "bankruptcy",
        "collapse", "failed", "failure", "weak", "weakness", "trouble",
    }

    # Verstaerker
    INTENSIFIERS = {
        "very", "extremely", "highly", "strongly", "significantly",
        "massive", "huge", "major", "big", "substantial",
    }

    def analyze_text(self, text: str) -> float:
        """
        Analysiert Text und gibt Sentiment-Score zurueck.

        Returns:
            Score von -1 (sehr negativ) bis +1 (sehr positiv)
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        positive_count = 0
        negative_count = 0
        intensifier_next = False

        for word in words:
            multiplier = 1.5 if intensifier_next else 1.0
            intensifier_next = word in self.INTENSIFIERS

            if word in self.POSITIVE_WORDS:
                positive_count += multiplier
            elif word in self.NEGATIVE_WORDS:
                negative_count += multiplier

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        score = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, score))

    def analyze_headlines(self, headlines: list[str]) -> SentimentData:
        """Analysiert mehrere Headlines."""
        if not headlines:
            return SentimentData(
                source="news_headlines",
                score=50,
                level=SentimentLevel.NEUTRAL,
                timestamp=datetime.now(),
            )

        scores = [self.analyze_text(h) for h in headlines]
        avg_score = sum(scores) / len(scores)

        # Konvertiere -1..+1 zu 0..100
        normalized_score = (avg_score + 1) * 50

        return SentimentData(
            source="news_headlines",
            score=normalized_score,
            level=SentimentLevel.from_score(normalized_score),
            timestamp=datetime.now(),
            details={
                "headlines_analyzed": len(headlines),
                "positive_count": len([s for s in scores if s > 0.1]),
                "negative_count": len([s for s in scores if s < -0.1]),
                "neutral_count": len([s for s in scores if -0.1 <= s <= 0.1]),
            },
        )


class SocialSentimentAnalyzer:
    """
    Social Media Sentiment Analyzer mit echten API-Integrationen.
    """

    def __init__(self):
        self._reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self._reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")

    async def get_sentiment(self, symbol: str) -> SentimentData:
        """
        Analysiert Social Media Sentiment für ein Symbol.
        Nutzt Reddit API wenn verfügbar.
        """
        sentiment_scores = []
        platforms_used = []

        # Reddit Sentiment
        if self._reddit_client_id and self._reddit_client_secret:
            reddit_score = await self._analyze_reddit(symbol)
            if reddit_score is not None:
                sentiment_scores.append(reddit_score)
                platforms_used.append("reddit")

        # Wenn keine APIs verfügbar, klare Fehlermeldung
        if not sentiment_scores:
            return SentimentData(
                source="social_media",
                score=50,  # Neutral
                level=SentimentLevel.NEUTRAL,
                timestamp=datetime.now(),
                details={
                    "symbol": symbol,
                    "error": "Keine Social Media APIs konfiguriert",
                    "required_env": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
                },
            )

        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        return SentimentData(
            source="social_media",
            score=avg_score,
            level=SentimentLevel.from_score(avg_score),
            timestamp=datetime.now(),
            details={
                "symbol": symbol,
                "platforms": platforms_used,
                "individual_scores": dict(zip(platforms_used, sentiment_scores)),
            },
        )

    async def _analyze_reddit(self, symbol: str) -> Optional[float]:
        """Analysiert Reddit-Posts für Sentiment."""
        import aiohttp
        try:
            # Reddit OAuth Token
            auth = aiohttp.BasicAuth(self._reddit_client_id, self._reddit_client_secret)
            async with aiohttp.ClientSession() as session:
                # Token holen
                token_resp = await session.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=auth,
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": "SCIO/1.0"}
                )
                if token_resp.status != 200:
                    return None
                token_data = await token_resp.json()
                access_token = token_data.get("access_token")

                # Suche nach Posts
                headers = {"Authorization": f"Bearer {access_token}", "User-Agent": "SCIO/1.0"}
                search_resp = await session.get(
                    f"https://oauth.reddit.com/search?q={symbol}&sort=hot&limit=25",
                    headers=headers
                )
                if search_resp.status != 200:
                    return None

                data = await search_resp.json()
                posts = data.get("data", {}).get("children", [])

                if not posts:
                    return 50  # Neutral wenn keine Posts

                # Einfache Sentiment-Analyse basierend auf Upvote-Ratio
                total_ratio = sum(p["data"].get("upvote_ratio", 0.5) for p in posts)
                avg_ratio = total_ratio / len(posts)
                return avg_ratio * 100  # 0-100 Score

        except Exception as e:
            logger.error(f"Reddit sentiment analysis failed: {e}")
            return None

    def get_sentiment_sync(self, symbol: str) -> SentimentData:
        """Synchrone Version von get_sentiment für nicht-async Kontexte."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Wenn bereits ein Loop läuft, erstelle neuen Thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.get_sentiment(symbol))
                    return future.result(timeout=10)
            else:
                return loop.run_until_complete(self.get_sentiment(symbol))
        except Exception as e:
            logger.error(f"Sync sentiment fetch failed: {e}")
            return SentimentData(
                source="social_media",
                score=50,
                level=SentimentLevel.NEUTRAL,
                timestamp=datetime.now(),
                details={"symbol": symbol, "error": str(e)},
            )


class TechnicalSentimentAnalyzer:
    """
    Technisches Sentiment basierend auf Indikatoren.
    """

    def analyze(
        self,
        rsi: float,
        macd_histogram: float,
        price_vs_sma200: float,  # % Abweichung
    ) -> SentimentData:
        """
        Berechnet technisches Sentiment.

        Args:
            rsi: RSI Wert (0-100)
            macd_histogram: MACD Histogram Wert
            price_vs_sma200: Preis vs 200-Tage-SMA in %
        """
        score = 50.0

        # RSI Beitrag (0-30 Punkte)
        if rsi < 30:
            score += (30 - rsi) / 30 * 15  # Ueberverkauft = bullisch
        elif rsi > 70:
            score -= (rsi - 70) / 30 * 15  # Ueberkauft = bearisch

        # MACD Beitrag (0-20 Punkte)
        if macd_histogram > 0:
            score += min(15, macd_histogram * 100)
        else:
            score -= min(15, abs(macd_histogram) * 100)

        # SMA200 Beitrag (0-20 Punkte)
        if price_vs_sma200 > 0:
            score += min(15, price_vs_sma200)
        else:
            score -= min(15, abs(price_vs_sma200))

        score = max(0, min(100, score))

        return SentimentData(
            source="technical_indicators",
            score=score,
            level=SentimentLevel.from_score(score),
            timestamp=datetime.now(),
            details={
                "rsi": rsi,
                "macd_histogram": macd_histogram,
                "price_vs_sma200_percent": price_vs_sma200,
            },
        )


class SentimentAggregator:
    """
    Aggregiert alle Sentiment-Quellen.
    """

    def __init__(self):
        self.fgi_fetcher = FearGreedIndexFetcher()
        self.text_analyzer = TextSentimentAnalyzer()
        self.social_analyzer = SocialSentimentAnalyzer()
        self.technical_analyzer = TechnicalSentimentAnalyzer()

    def get_market_sentiment(
        self,
        symbol: str = "BTC",
        headlines: Optional[list[str]] = None,
        rsi: float = 50,
        macd_histogram: float = 0,
        price_vs_sma200: float = 0,
    ) -> MarketSentiment:
        """
        Aggregiert alle Sentiment-Quellen zu einem Gesamtbild.
        """
        # Fear & Greed Index (Crypto)
        fgi = self.fgi_fetcher.get_crypto_fear_greed()

        # News Sentiment
        news_sentiment = None
        if headlines:
            news_sentiment = self.text_analyzer.analyze_headlines(headlines)

        # Social Sentiment
        social_sentiment = self.social_analyzer.get_sentiment_sync(symbol)

        # Technical Sentiment
        technical_sentiment = self.technical_analyzer.analyze(
            rsi=rsi,
            macd_histogram=macd_histogram,
            price_vs_sma200=price_vs_sma200,
        )

        # Gewichteter Durchschnitt
        weights = {
            "fgi": 0.35,
            "news": 0.20,
            "social": 0.15,
            "technical": 0.30,
        }

        scores = []
        total_weight = 0

        if fgi:
            scores.append(fgi.score * weights["fgi"])
            total_weight += weights["fgi"]

        if news_sentiment:
            scores.append(news_sentiment.score * weights["news"])
            total_weight += weights["news"]

        if social_sentiment:
            scores.append(social_sentiment.score * weights["social"])
            total_weight += weights["social"]

        if technical_sentiment:
            scores.append(technical_sentiment.score * weights["technical"])
            total_weight += weights["technical"]

        overall_score = sum(scores) / total_weight if total_weight > 0 else 50
        overall_level = SentimentLevel.from_score(overall_score)

        # Empfehlung generieren
        recommendation = self._generate_recommendation(
            overall_score, overall_level, fgi, technical_sentiment
        )

        return MarketSentiment(
            overall_score=overall_score,
            overall_level=overall_level,
            fear_greed_index=fgi,
            social_sentiment=social_sentiment,
            news_sentiment=news_sentiment,
            technical_sentiment=technical_sentiment,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        score: float,
        level: SentimentLevel,
        fgi: Optional[SentimentData],
        technical: Optional[SentimentData],
    ) -> str:
        """Generiert Handlungsempfehlung basierend auf Sentiment."""
        parts = []

        if level == SentimentLevel.EXTREME_FEAR:
            parts.append(
                "EXTREME FEAR: Historisch guter Kaufzeitpunkt. "
                "'Be greedy when others are fearful.'"
            )
        elif level == SentimentLevel.FEAR:
            parts.append(
                "FEAR: Markt pessimistisch. Potenzielle Kaufgelegenheit fuer Langfristige."
            )
        elif level == SentimentLevel.NEUTRAL:
            parts.append(
                "NEUTRAL: Abwarten bis klare Richtung. "
                "Keine extreme Positionierung empfohlen."
            )
        elif level == SentimentLevel.GREED:
            parts.append(
                "GREED: Markt optimistisch. Vorsicht bei neuen Kaeufen. "
                "Stop-Loss setzen."
            )
        elif level == SentimentLevel.EXTREME_GREED:
            parts.append(
                "EXTREME GREED: Historisch schlechter Kaufzeitpunkt. "
                "Gewinne sichern erwägen. 'Be fearful when others are greedy.'"
            )

        # Divergenz-Check
        if fgi and technical:
            diff = abs(fgi.score - technical.score)
            if diff > 30:
                parts.append(
                    f"DIVERGENZ: FGI ({fgi.score:.0f}) vs Technical ({technical.score:.0f}) "
                    f"- moeglicher Wendepunkt!"
                )

        return " | ".join(parts)


def analyze_sentiment_for_trading(
    symbol: str,
    headlines: list[str] = None,
    rsi: float = 50,
    macd_histogram: float = 0,
    price_vs_sma200: float = 0,
) -> dict[str, Any]:
    """
    Einfache Funktion fuer Sentiment-Analyse im Trading.
    """
    aggregator = SentimentAggregator()
    sentiment = aggregator.get_market_sentiment(
        symbol=symbol,
        headlines=headlines,
        rsi=rsi,
        macd_histogram=macd_histogram,
        price_vs_sma200=price_vs_sma200,
    )

    return sentiment.to_dict()


if __name__ == "__main__":
    print("=== SENTIMENT ANALYZER TEST ===\n")

    aggregator = SentimentAggregator()

    # Test Fear & Greed Index
    print("--- Crypto Fear & Greed Index ---")
    fgi = aggregator.fgi_fetcher.get_crypto_fear_greed()
    if fgi:
        print(f"Score: {fgi.score:.0f}")
        print(f"Level: {fgi.level.value}")
        print(f"Details: {fgi.details}")

    # Test Text Sentiment
    print("\n--- News Sentiment ---")
    headlines = [
        "Bitcoin surges to new all-time high amid institutional adoption",
        "SEC investigation causes crypto market panic",
        "Ethereum upgrade successful, network stronger than ever",
        "Major exchange hacked, millions lost",
        "Bullish momentum continues as BTC breaks resistance",
    ]

    news_sentiment = aggregator.text_analyzer.analyze_headlines(headlines)
    print(f"Score: {news_sentiment.score:.1f}")
    print(f"Level: {news_sentiment.level.value}")
    print(f"Details: {news_sentiment.details}")

    # Test Gesamtsentiment
    print("\n--- Gesamt-Sentiment (BTC) ---")
    sentiment = aggregator.get_market_sentiment(
        symbol="BTC",
        headlines=headlines,
        rsi=35,
        macd_histogram=0.002,
        price_vs_sma200=5,
    )

    print(f"Overall Score: {sentiment.overall_score:.1f}")
    print(f"Overall Level: {sentiment.overall_level.value}")
    print(f"\nEmpfehlung: {sentiment.recommendation}")
