"""
SCIO Advanced Analytics

Mächtige Analysefähigkeiten für Daten und Muster.
"""

from scio.analytics.statistics import (
    StatisticsEngine,
    DescriptiveStats,
    HypothesisTest,
    CorrelationAnalysis,
)
from scio.analytics.patterns import (
    PatternDetector,
    AnomalyDetector,
    TrendAnalyzer,
    ClusterAnalyzer,
)
from scio.analytics.timeseries import (
    TimeSeriesAnalyzer,
    Forecaster,
    SeasonalDecomposer,
)
from scio.analytics.ml import (
    AutoMLPipeline,
    ModelTrainer,
    FeatureEngineering,
    ModelEvaluator,
)

__all__ = [
    # Statistics
    "StatisticsEngine",
    "DescriptiveStats",
    "HypothesisTest",
    "CorrelationAnalysis",
    # Patterns
    "PatternDetector",
    "AnomalyDetector",
    "TrendAnalyzer",
    "ClusterAnalyzer",
    # Time Series
    "TimeSeriesAnalyzer",
    "Forecaster",
    "SeasonalDecomposer",
    # Machine Learning
    "AutoMLPipeline",
    "ModelTrainer",
    "FeatureEngineering",
    "ModelEvaluator",
]
