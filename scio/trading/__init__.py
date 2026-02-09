"""
SCIO Trading Module

Vollstaendiges Trading-System mit:
- Marktanalyse
- Technische Indikatoren
- Krypto-Trading
- Portfolio-Optimierung
- Trading-Strategien
- Sentiment-Analyse
- Backtesting
"""

from scio.trading.market_analyzer import (
    MarketAnalyzer,
    MarketAnalysis,
    TechnicalIndicators,
    PriceData,
    TrendDirection,
    SignalStrength,
    MarketCondition,
    StrategyBenchmark,
    BenchmarkResult,
)

from scio.trading.crypto import (
    CryptoPriceFetcher,
    CryptoPortfolio,
    CryptoAnalyzer,
    CryptoAsset,
    StakingPosition,
    DeFiPosition,
    NetworkType,
    calculate_staking_rewards,
)

from scio.trading.portfolio_optimizer import (
    PortfolioOptimizer,
    Portfolio,
    Asset,
    OptimizationResult,
    RiskAnalyzer,
    suggest_portfolio_for_goal,
)

from scio.trading.strategies import (
    TradingStrategy,
    TradingSignal,
    Trade,
    SignalType,
    PositionType,
    RSIMeanReversionStrategy,
    MACrossoverStrategy,
    BreakoutStrategy,
    BollingerBandStrategy,
    StrategyManager,
    create_default_strategy_manager,
)

from scio.trading.sentiment import (
    SentimentAggregator,
    SentimentData,
    SentimentLevel,
    MarketSentiment,
    FearGreedIndexFetcher,
    TextSentimentAnalyzer,
    analyze_sentiment_for_trading,
)

__all__ = [
    # Market Analyzer
    "MarketAnalyzer",
    "MarketAnalysis",
    "TechnicalIndicators",
    "PriceData",
    "TrendDirection",
    "SignalStrength",
    "MarketCondition",
    "StrategyBenchmark",
    "BenchmarkResult",
    # Crypto
    "CryptoPriceFetcher",
    "CryptoPortfolio",
    "CryptoAnalyzer",
    "CryptoAsset",
    "StakingPosition",
    "DeFiPosition",
    "NetworkType",
    "calculate_staking_rewards",
    # Portfolio
    "PortfolioOptimizer",
    "Portfolio",
    "Asset",
    "OptimizationResult",
    "RiskAnalyzer",
    "suggest_portfolio_for_goal",
    # Strategies
    "TradingStrategy",
    "TradingSignal",
    "Trade",
    "SignalType",
    "PositionType",
    "RSIMeanReversionStrategy",
    "MACrossoverStrategy",
    "BreakoutStrategy",
    "BollingerBandStrategy",
    "StrategyManager",
    "create_default_strategy_manager",
    # Sentiment
    "SentimentAggregator",
    "SentimentData",
    "SentimentLevel",
    "MarketSentiment",
    "FearGreedIndexFetcher",
    "TextSentimentAnalyzer",
    "analyze_sentiment_for_trading",
]
