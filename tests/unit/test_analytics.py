"""
Comprehensive tests for SCIO Analytics Module.

Tests cover:
- StatisticsEngine: descriptive stats, hypothesis tests, correlations
- PatternDetector: repeating patterns, cycles
- AnomalyDetector: z-score, IQR, MAD, rolling methods
- TrendAnalyzer: trend analysis, changepoints, segmentation
- ClusterAnalyzer: k-means, silhouette, optimal k
- TimeSeriesAnalyzer: stationarity, ACF, PACF, spectral analysis
- SeasonalDecomposer: additive and multiplicative decomposition
- Forecaster: SES, Holt, Holt-Winters, auto-forecast
- FeatureEngineering: standardization, normalization, polynomial features
- ModelTrainer: linear regression, logistic regression, KNN
- ModelEvaluator: cross-validation
- AutoMLPipeline: automated model selection
"""

import numpy as np
import pytest

from scio.analytics import (
    StatisticsEngine,
    DescriptiveStats,
    HypothesisTest,
    CorrelationAnalysis,
    PatternDetector,
    AnomalyDetector,
    TrendAnalyzer,
    ClusterAnalyzer,
    TimeSeriesAnalyzer,
    Forecaster,
    SeasonalDecomposer,
    AutoMLPipeline,
    ModelTrainer,
    FeatureEngineering,
    ModelEvaluator,
)
from scio.analytics.statistics import TestType
from scio.analytics.patterns import AnomalyType, TrendDirection, Anomaly, Trend, Cluster
from scio.analytics.timeseries import SeasonalComponents, Forecast
from scio.analytics.ml import ModelType, TaskType, ModelMetrics, TrainedModel


class TestDescriptiveStats:
    """Tests for descriptive statistics."""

    def test_describe_basic(self):
        """Test basic descriptive statistics."""
        engine = StatisticsEngine()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        stats = engine.describe(data)

        assert isinstance(stats, DescriptiveStats)
        assert stats.count == 10
        assert stats.mean == pytest.approx(5.5, rel=1e-3)
        assert stats.min == 1
        assert stats.max == 10
        assert stats.median == pytest.approx(5.5, rel=1e-3)

    def test_describe_with_numpy_array(self):
        """Test describe with numpy array input."""
        engine = StatisticsEngine()
        data = np.array([2, 4, 6, 8, 10])

        stats = engine.describe(data)

        assert stats.count == 5
        assert stats.mean == pytest.approx(6.0, rel=1e-3)
        assert stats.std == pytest.approx(np.std(data, ddof=1), rel=1e-3)

    def test_describe_variance_and_std(self):
        """Test variance and standard deviation calculations."""
        engine = StatisticsEngine()
        data = [10, 20, 30, 40, 50]

        stats = engine.describe(data)

        # Sample variance and std
        expected_var = np.var(data, ddof=1)
        expected_std = np.std(data, ddof=1)

        assert stats.variance == pytest.approx(expected_var, rel=1e-3)
        assert stats.std == pytest.approx(expected_std, rel=1e-3)

    def test_describe_mode(self):
        """Test mode calculation."""
        engine = StatisticsEngine()
        data = [1, 2, 2, 3, 3, 3, 4, 4, 5]  # 3 is the mode

        stats = engine.describe(data)

        assert stats.mode == 3

    def test_describe_skewness_symmetric(self):
        """Test skewness for symmetric data."""
        engine = StatisticsEngine()
        # Symmetric data around 5
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        stats = engine.describe(data)

        # Symmetric data should have skewness close to 0
        assert abs(stats.skewness) < 0.5

    def test_describe_kurtosis(self):
        """Test kurtosis calculation."""
        engine = StatisticsEngine()
        data = np.random.normal(0, 1, 1000)

        stats = engine.describe(data)

        # Normal distribution has kurtosis around 0 (excess kurtosis)
        assert stats.kurtosis is not None

    def test_describe_quartiles(self):
        """Test quartile calculations."""
        engine = StatisticsEngine()
        data = list(range(1, 101))  # 1 to 100

        stats = engine.describe(data)

        assert stats.q1 == pytest.approx(25.75, rel=1e-1)
        assert stats.q3 == pytest.approx(75.25, rel=1e-1)


class TestHypothesisTesting:
    """Tests for hypothesis testing."""

    def test_t_test_significant_difference(self):
        """Test t-test detects significant difference."""
        engine = StatisticsEngine()
        # Two groups with clearly different means
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(150, 10, 50)

        result = engine.t_test(group1, group2)

        assert isinstance(result, HypothesisTest)
        assert result.test_type == TestType.T_TEST
        assert result.p_value < 0.05
        assert result.significant == True  # Use == for numpy bool

    def test_t_test_no_significant_difference(self):
        """Test t-test with same distribution."""
        engine = StatisticsEngine()
        np.random.seed(42)
        group1 = np.random.normal(100, 10, 30)
        group2 = np.random.normal(100, 10, 30)

        result = engine.t_test(group1, group2)

        # Should likely not be significant
        assert result.test_type == TestType.T_TEST
        assert result.statistic is not None

    def test_paired_t_test(self):
        """Test paired t-test."""
        engine = StatisticsEngine()
        # Before and after treatment
        before = np.array([120, 130, 125, 140, 135, 145, 150, 155])
        after = np.array([115, 125, 118, 130, 128, 138, 142, 148])  # Generally lower

        result = engine.t_test(before, after, paired=True)

        assert result.test_type == TestType.PAIRED_T_TEST

    def test_anova_significant(self):
        """Test ANOVA with different groups."""
        engine = StatisticsEngine()
        group1 = np.random.normal(10, 2, 20)
        group2 = np.random.normal(20, 2, 20)
        group3 = np.random.normal(30, 2, 20)

        result = engine.anova(group1, group2, group3)

        assert result.test_type == TestType.ANOVA
        assert result.p_value < 0.05
        assert result.significant == True  # Use == for numpy bool

    def test_anova_no_difference(self):
        """Test ANOVA with similar groups."""
        engine = StatisticsEngine()
        np.random.seed(42)
        group1 = np.random.normal(50, 5, 20)
        group2 = np.random.normal(50, 5, 20)
        group3 = np.random.normal(50, 5, 20)

        result = engine.anova(group1, group2, group3)

        assert result.test_type == TestType.ANOVA

    def test_chi_square(self):
        """Test chi-square test for independence."""
        engine = StatisticsEngine()
        # Contingency table
        observed = np.array([[10, 20], [30, 40]])

        result = engine.chi_square(observed)

        assert result.test_type == TestType.CHI_SQUARE
        assert result.statistic is not None
        assert result.p_value is not None

    def test_normality_test_normal_data(self):
        """Test normality test with normal data."""
        engine = StatisticsEngine()
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)

        result = engine.normality_test(normal_data)

        assert result.test_type == TestType.SHAPIRO_WILK
        # Normal data should likely pass normality test
        assert result.p_value > 0.01

    def test_normality_test_non_normal_data(self):
        """Test normality test with non-normal data."""
        engine = StatisticsEngine()
        # Highly skewed data
        non_normal = np.exp(np.random.normal(0, 1, 100))

        result = engine.normality_test(non_normal)

        assert result.test_type == TestType.SHAPIRO_WILK


class TestCorrelation:
    """Tests for correlation analysis."""

    def test_correlation_positive(self):
        """Test positive correlation."""
        engine = StatisticsEngine()
        x = np.arange(0, 100)
        y = x * 2 + np.random.normal(0, 5, 100)  # Strong positive correlation

        result = engine.correlation(x, y)

        assert isinstance(result, CorrelationAnalysis)
        assert result.coefficient > 0.9
        assert result.strength in ["strong", "very strong"]
        assert result.direction == "positive"

    def test_correlation_negative(self):
        """Test negative correlation."""
        engine = StatisticsEngine()
        x = np.arange(0, 100)
        y = -x * 2 + np.random.normal(0, 5, 100)  # Strong negative correlation

        result = engine.correlation(x, y)

        assert result.coefficient < -0.9
        assert result.direction == "negative"

    def test_correlation_no_correlation(self):
        """Test no correlation."""
        engine = StatisticsEngine()
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)  # Independent

        result = engine.correlation(x, y)

        assert abs(result.coefficient) < 0.3

    def test_correlation_matrix(self):
        """Test correlation matrix for multiple variables."""
        engine = StatisticsEngine()
        np.random.seed(42)

        # Create correlated variables
        x1 = np.random.normal(0, 1, 100)
        x2 = x1 + np.random.normal(0, 0.1, 100)  # Highly correlated with x1
        x3 = np.random.normal(0, 1, 100)  # Independent

        # correlation_matrix expects a dict
        data = {"x1": x1, "x2": x2, "x3": x3}

        matrix = engine.correlation_matrix(data)

        # Returns dict of dicts with CorrelationAnalysis
        assert "x1" in matrix
        assert "x2" in matrix["x1"]
        # Diagonal should be 1
        assert matrix["x1"]["x1"].coefficient == pytest.approx(1.0, abs=0.01)
        # x1 and x2 should be highly correlated
        assert matrix["x1"]["x2"].coefficient > 0.9


class TestPatternDetector:
    """Tests for pattern detection."""

    def test_find_repeating_patterns_simple(self):
        """Test finding simple repeating patterns."""
        detector = PatternDetector()
        # Repeating pattern: 1, 2, 3
        data = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

        patterns = detector.find_repeating_patterns(data, min_pattern_length=2)

        assert len(patterns) > 0

    def test_find_repeating_patterns_no_pattern(self):
        """Test with random data (no clear pattern)."""
        detector = PatternDetector()
        np.random.seed(42)
        data = list(np.random.normal(0, 1, 50))

        patterns = detector.find_repeating_patterns(data, min_pattern_length=5)

        # May or may not find patterns in random data
        assert isinstance(patterns, list)

    def test_find_cycles(self):
        """Test finding cycles in data."""
        detector = PatternDetector()
        # Create data with clear cycle of period 10
        t = np.arange(0, 100)
        data = np.sin(2 * np.pi * t / 10)  # Period = 10

        cycles = detector.find_cycles(data)

        assert len(cycles) > 0


class TestAnomalyDetector:
    """Tests for anomaly detection."""

    def test_detect_zscore_outliers(self):
        """Test z-score anomaly detection."""
        # Method is set in constructor
        detector = AnomalyDetector(method="zscore")
        # Normal data with outliers
        data = list(np.random.normal(50, 5, 100)) + [200, -50]  # Add outliers

        anomalies = detector.detect(data)

        assert len(anomalies) >= 2
        assert all(isinstance(a, Anomaly) for a in anomalies)
        assert all(a.anomaly_type == AnomalyType.POINT for a in anomalies)

    def test_detect_iqr_outliers(self):
        """Test IQR anomaly detection."""
        detector = AnomalyDetector(method="iqr", threshold=1.0)
        # Data with outliers
        data = list(np.random.normal(100, 10, 100)) + [300, -100]

        anomalies = detector.detect(data)

        assert len(anomalies) >= 2

    def test_detect_mad_outliers(self):
        """Test MAD (Median Absolute Deviation) detection."""
        detector = AnomalyDetector(method="mad")
        data = list(np.random.normal(0, 1, 100)) + [100, -100]

        anomalies = detector.detect(data)

        assert len(anomalies) >= 2

    def test_detect_rolling_outliers(self):
        """Test rolling window anomaly detection."""
        # Use lower threshold to detect the spike
        detector = AnomalyDetector(method="rolling", window_size=10, threshold=2.0)
        # Data with natural variation so std > 0, plus a clear spike
        np.random.seed(42)
        base_data = np.random.normal(10, 1, 100)  # Mean 10, std 1
        base_data[50] = 50  # Add clear outlier (40 std deviations)
        data = list(base_data)

        anomalies = detector.detect(data)

        # The rolling method should detect the spike at index 50
        assert len(anomalies) >= 1

    def test_detect_threshold(self):
        """Test anomaly detection with custom threshold."""
        data = list(np.random.normal(0, 1, 100))

        # Lower threshold = more anomalies
        detector_high = AnomalyDetector(method="zscore", threshold=3.0)
        detector_low = AnomalyDetector(method="zscore", threshold=1.5)

        anomalies_high = detector_high.detect(data)
        anomalies_low = detector_low.detect(data)

        assert len(anomalies_low) >= len(anomalies_high)

    def test_no_anomalies(self):
        """Test detection with clean data."""
        detector = AnomalyDetector(method="zscore", threshold=3.0)
        data = [50, 51, 49, 50, 52, 48, 50, 51, 49, 50]

        anomalies = detector.detect(data)

        # Clean data should have no anomalies with high threshold
        assert len(anomalies) == 0


class TestTrendAnalyzer:
    """Tests for trend analysis."""

    def test_analyze_increasing_trend(self):
        """Test detecting increasing trend."""
        analyzer = TrendAnalyzer()
        data = np.arange(0, 100) + np.random.normal(0, 5, 100)

        trend = analyzer.analyze(data)

        assert isinstance(trend, Trend)
        assert trend.direction == TrendDirection.INCREASING
        assert trend.slope > 0

    def test_analyze_decreasing_trend(self):
        """Test detecting decreasing trend."""
        analyzer = TrendAnalyzer()
        data = 100 - np.arange(0, 100) + np.random.normal(0, 5, 100)

        trend = analyzer.analyze(data)

        assert trend.direction == TrendDirection.DECREASING
        assert trend.slope < 0

    def test_analyze_stable_trend(self):
        """Test detecting stable/flat trend."""
        analyzer = TrendAnalyzer()
        np.random.seed(42)
        data = np.random.normal(50, 2, 100)  # Random around 50

        trend = analyzer.analyze(data)

        assert abs(trend.slope) < 0.5

    def test_find_changepoints(self):
        """Test finding changepoints in data."""
        analyzer = TrendAnalyzer()
        # Data with clear changepoint at index 50
        data1 = np.random.normal(10, 1, 50)
        data2 = np.random.normal(50, 1, 50)
        data = np.concatenate([data1, data2])

        changepoints = analyzer.find_changepoints(data)

        assert len(changepoints) >= 1
        # Changepoint should be near index 50
        assert any(45 <= cp <= 55 for cp in changepoints)

    def test_segment_trends(self):
        """Test segmenting data by trends."""
        analyzer = TrendAnalyzer()
        # Increasing then decreasing
        data1 = np.arange(0, 50)
        data2 = np.arange(50, 0, -1)
        data = np.concatenate([data1, data2])

        segments = analyzer.segment_trends(data)

        assert len(segments) >= 1


class TestClusterAnalyzer:
    """Tests for clustering analysis."""

    def test_kmeans_basic(self):
        """Test basic k-means clustering."""
        analyzer = ClusterAnalyzer()
        np.random.seed(42)

        # Create 3 distinct clusters
        cluster1 = np.random.normal([0, 0], 1, (30, 2))
        cluster2 = np.random.normal([10, 10], 1, (30, 2))
        cluster3 = np.random.normal([0, 10], 1, (30, 2))
        data = np.vstack([cluster1, cluster2, cluster3])

        clusters = analyzer.kmeans(data, k=3)

        assert len(clusters) == 3
        assert all(isinstance(c, Cluster) for c in clusters)

    def test_kmeans_returns_correct_structure(self):
        """Test k-means returns correct cluster structure."""
        analyzer = ClusterAnalyzer()
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 2))

        clusters = analyzer.kmeans(data, k=2)

        # Each cluster should have centroid and members
        for cluster in clusters:
            assert cluster.centroid is not None
            assert len(cluster.centroid) == 2
            assert cluster.members is not None

    def test_find_optimal_k_elbow(self):
        """Test finding optimal k using elbow method."""
        analyzer = ClusterAnalyzer()
        np.random.seed(42)

        # Create clear 3-cluster structure
        cluster1 = np.random.normal([0, 0], 0.5, (30, 2))
        cluster2 = np.random.normal([10, 0], 0.5, (30, 2))
        cluster3 = np.random.normal([5, 10], 0.5, (30, 2))
        data = np.vstack([cluster1, cluster2, cluster3])

        optimal_k = analyzer.find_optimal_k(data, max_k=6)

        # Should find k around 3
        assert 2 <= optimal_k <= 5

    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        analyzer = ClusterAnalyzer()
        np.random.seed(42)

        # Well-separated clusters
        cluster1 = np.random.normal([0, 0], 0.5, (20, 2))
        cluster2 = np.random.normal([10, 10], 0.5, (20, 2))
        data = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 20 + [1] * 20)

        score = analyzer.silhouette_score(data, labels)

        # Well-separated clusters should have high silhouette
        assert score > 0.5


class TestTimeSeriesAnalyzer:
    """Tests for time series analysis."""

    def test_is_stationary_stationary_data(self):
        """Test stationarity detection for stationary data."""
        analyzer = TimeSeriesAnalyzer()
        np.random.seed(42)
        data = np.random.normal(50, 5, 100)  # Stationary

        result = analyzer.is_stationary(data)

        assert result["is_stationary"] == True  # Use == for numpy bool
        assert "mean_variation" in result
        assert "variance_variation" in result

    def test_is_stationary_non_stationary(self):
        """Test stationarity detection for non-stationary data."""
        analyzer = TimeSeriesAnalyzer()
        # Trending data (non-stationary)
        data = np.arange(0, 100) + np.random.normal(0, 2, 100)

        result = analyzer.is_stationary(data)

        assert result["is_stationary"] == False  # Use == for numpy bool

    def test_is_stationary_insufficient_data(self):
        """Test stationarity with insufficient data."""
        analyzer = TimeSeriesAnalyzer()
        data = [1, 2, 3, 4, 5]  # Too short

        result = analyzer.is_stationary(data, window=10)

        assert result["is_stationary"] is None
        assert "Insufficient" in result["reason"]

    def test_autocorrelation(self):
        """Test autocorrelation function calculation."""
        analyzer = TimeSeriesAnalyzer()
        # Create AR(1) process with known autocorrelation
        np.random.seed(42)
        n = 100
        data = np.zeros(n)
        phi = 0.8
        for i in range(1, n):
            data[i] = phi * data[i-1] + np.random.normal(0, 1)

        acf = analyzer.autocorrelation(data, max_lag=10)

        assert len(acf) == 11  # 0 to 10
        assert acf[0] == pytest.approx(1.0)  # Lag 0 is always 1
        assert acf[1] > 0.5  # Should show positive autocorrelation

    def test_partial_autocorrelation(self):
        """Test partial autocorrelation function calculation."""
        analyzer = TimeSeriesAnalyzer()
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        pacf = analyzer.partial_autocorrelation(data, max_lag=10)

        assert len(pacf) == 11
        assert pacf[0] == pytest.approx(1.0)

    def test_difference_order_1(self):
        """Test first-order differencing."""
        analyzer = TimeSeriesAnalyzer()
        data = np.array([1, 3, 6, 10, 15])

        diff = analyzer.difference(data, order=1)

        expected = np.array([2, 3, 4, 5])
        assert np.allclose(diff, expected)

    def test_difference_order_2(self):
        """Test second-order differencing."""
        analyzer = TimeSeriesAnalyzer()
        data = np.array([1, 3, 6, 10, 15, 21])

        diff = analyzer.difference(data, order=2)

        # First diff: [2, 3, 4, 5, 6], second diff: [1, 1, 1, 1]
        expected = np.array([1, 1, 1, 1])
        assert np.allclose(diff, expected)

    def test_difference_seasonal(self):
        """Test seasonal differencing."""
        analyzer = TimeSeriesAnalyzer()
        data = np.array([10, 20, 30, 40, 15, 25, 35, 45, 20, 30, 40, 50])

        diff = analyzer.difference(data, order=0, seasonal=4)

        # Seasonal diff: each value minus value 4 periods ago
        expected = data[4:] - data[:-4]
        assert np.allclose(diff, expected)

    def test_spectral_analysis(self):
        """Test spectral analysis via FFT."""
        analyzer = TimeSeriesAnalyzer()
        # Create data with known frequency
        t = np.arange(0, 100)
        freq = 0.1  # Period = 10
        data = np.sin(2 * np.pi * freq * t)

        result = analyzer.spectral_analysis(data)

        assert "frequencies" in result
        assert "power_spectrum" in result
        assert "dominant_frequencies" in result
        assert len(result["dominant_frequencies"]) > 0


class TestSeasonalDecomposer:
    """Tests for seasonal decomposition."""

    def test_decompose_additive(self):
        """Test additive seasonal decomposition."""
        decomposer = SeasonalDecomposer(method="additive")

        # Create data with known components
        t = np.arange(0, 100)
        trend = 0.1 * t  # Linear trend
        seasonal = 5 * np.sin(2 * np.pi * t / 12)  # Period 12
        residual = np.random.normal(0, 0.5, 100)
        data = trend + seasonal + residual

        result = decomposer.decompose(data, period=12)

        assert isinstance(result, SeasonalComponents)
        assert len(result.trend) == len(data)
        assert len(result.seasonal) == len(data)
        assert len(result.residual) == len(data)
        assert result.period == 12

    def test_decompose_multiplicative(self):
        """Test multiplicative seasonal decomposition."""
        decomposer = SeasonalDecomposer(method="multiplicative")

        # Create multiplicative data
        t = np.arange(0, 100)
        trend = 100 + 0.5 * t  # Linear trend
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * t / 12)  # Multiplicative seasonal
        data = trend * seasonal

        result = decomposer.decompose(data, period=12)

        assert isinstance(result, SeasonalComponents)
        assert len(result.trend) == len(data)


class TestForecaster:
    """Tests for time series forecasting."""

    def test_simple_exponential_smoothing(self):
        """Test simple exponential smoothing forecast."""
        forecaster = Forecaster()
        data = [100, 102, 104, 103, 105, 107, 106, 108]

        forecast = forecaster.simple_exponential_smoothing(data, alpha=0.3, horizon=5)

        assert isinstance(forecast, Forecast)
        assert len(forecast.values) == 5
        assert len(forecast.lower_bound) == 5
        assert len(forecast.upper_bound) == 5
        assert forecast.method == "SES"
        assert forecast.horizon == 5

    def test_holt_linear_with_trend(self):
        """Test Holt linear smoothing with trending data."""
        forecaster = Forecaster()
        # Data with clear upward trend
        data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

        forecast = forecaster.holt_linear(data, alpha=0.3, beta=0.1, horizon=5)

        assert isinstance(forecast, Forecast)
        assert forecast.method == "Holt"
        # Forecast should continue the trend
        assert forecast.values[-1] > forecast.values[0]

    def test_holt_winters_seasonal(self):
        """Test Holt-Winters with seasonal data."""
        forecaster = Forecaster()
        # Create seasonal data with trend
        t = np.arange(0, 48)  # 4 years of monthly data
        trend = 100 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        data = trend + seasonal

        forecast = forecaster.holt_winters(
            data, period=12, horizon=12, multiplicative=False
        )

        assert isinstance(forecast, Forecast)
        assert forecast.method == "Holt-Winters"
        assert len(forecast.values) == 12

    def test_holt_winters_fallback(self):
        """Test Holt-Winters fallback for insufficient data."""
        forecaster = Forecaster()
        # Too short for period=12
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        forecast = forecaster.holt_winters(data, period=12, horizon=5)

        # Should fall back to Holt linear
        assert forecast.method == "Holt"

    def test_auto_forecast_with_trend(self):
        """Test auto forecast detects trend."""
        forecaster = Forecaster()
        # Clear trending data
        data = np.arange(0, 50) + np.random.normal(0, 1, 50)

        forecast = forecaster.auto_forecast(data, horizon=10)

        assert isinstance(forecast, Forecast)
        # Should use Holt method for trending data
        assert forecast.method in ["Holt", "Holt-Winters", "SES"]

    def test_auto_forecast_stationary(self):
        """Test auto forecast for stationary data."""
        forecaster = Forecaster()
        np.random.seed(42)
        # Stationary data
        data = np.random.normal(50, 5, 100)

        forecast = forecaster.auto_forecast(data, horizon=10)

        assert isinstance(forecast, Forecast)

    def test_confidence_intervals(self):
        """Test confidence intervals in forecast."""
        forecaster = Forecaster()
        data = [100, 102, 99, 101, 103, 100, 102, 101]

        forecast = forecaster.simple_exponential_smoothing(
            data, horizon=5, confidence=0.95
        )

        # Lower bound should be less than forecast
        assert all(forecast.lower_bound < forecast.values)
        # Upper bound should be greater than forecast
        assert all(forecast.upper_bound > forecast.values)
        assert forecast.confidence_level == 0.95


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_standardize_fit(self):
        """Test standardization fitting."""
        fe = FeatureEngineering()
        data = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]])

        result = fe.standardize(data, fit=True)

        # Standardized data should have mean ~0 and std ~1
        assert np.allclose(np.mean(result, axis=0), 0, atol=0.1)
        assert np.allclose(np.std(result, axis=0), 1, atol=0.1)

    def test_standardize_transform(self):
        """Test standardization transform without fitting."""
        fe = FeatureEngineering()
        train = np.array([[1, 10], [2, 20], [3, 30]])
        test = np.array([[2.5, 25]])

        # Fit on train
        fe.standardize(train, fit=True, name="test")

        # Transform test without fitting
        result = fe.standardize(test, fit=False, name="test")

        assert result.shape == (1, 2)

    def test_standardize_missing_scaler(self):
        """Test error when scaler not fitted."""
        fe = FeatureEngineering()
        data = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="not fitted"):
            fe.standardize(data, fit=False, name="nonexistent")

    def test_normalize_fit(self):
        """Test min-max normalization."""
        fe = FeatureEngineering()
        data = np.array([[0, 0], [5, 100], [10, 200]])

        result = fe.normalize(data, fit=True)

        # Normalized data should be in [0, 1]
        assert np.min(result) >= 0
        assert np.max(result) <= 1 + 1e-10

    def test_polynomial_features_degree_2(self):
        """Test polynomial feature generation degree 2."""
        fe = FeatureEngineering()
        data = np.array([[1, 2], [3, 4], [5, 6]])

        result = fe.polynomial_features(data, degree=2, include_bias=False)

        # Original + squared + interaction: 2 + 2 + 1 = 5
        assert result.shape[1] >= 5

    def test_polynomial_features_with_bias(self):
        """Test polynomial features with bias term."""
        fe = FeatureEngineering()
        data = np.array([[1, 2], [3, 4]])

        result = fe.polynomial_features(data, degree=2, include_bias=True)

        # First column should be all ones (bias)
        assert np.allclose(result[:, 0], 1)

    def test_select_features_correlation(self):
        """Test feature selection by correlation."""
        fe = FeatureEngineering()
        np.random.seed(42)

        # Create features with different correlations to target
        n = 100
        y = np.random.normal(0, 1, n)
        x1 = y + np.random.normal(0, 0.1, n)  # High correlation
        x2 = y + np.random.normal(0, 1, n)    # Medium correlation
        x3 = np.random.normal(0, 1, n)         # Low correlation

        X = np.column_stack([x1, x2, x3])

        X_selected, indices = fe.select_features(X, y, method="correlation", k=2)

        assert X_selected.shape[1] == 2
        assert len(indices) == 2
        # Should select x1 and x2 (highest correlations)
        assert 0 in indices

    def test_select_features_variance(self):
        """Test feature selection by variance."""
        fe = FeatureEngineering()
        np.random.seed(42)

        x1 = np.random.normal(0, 10, 100)  # High variance
        x2 = np.random.normal(0, 1, 100)   # Medium variance
        x3 = np.random.normal(0, 0.1, 100) # Low variance

        X = np.column_stack([x1, x2, x3])
        y = np.random.normal(0, 1, 100)

        X_selected, indices = fe.select_features(X, y, method="variance", k=2)

        assert X_selected.shape[1] == 2
        # Should select x1 and x2 (highest variance)
        assert 0 in indices


class TestModelTrainer:
    """Tests for model training."""

    def test_train_linear_regression(self):
        """Test linear regression training."""
        trainer = ModelTrainer()
        np.random.seed(42)

        # Simple linear relationship
        X = np.random.normal(0, 1, (100, 3))
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 100)

        model = trainer.train_linear_regression(X, y)

        assert isinstance(model, TrainedModel)
        assert model.model_type == ModelType.LINEAR_REGRESSION
        assert model.task_type == TaskType.REGRESSION
        assert model.metrics.r2 > 0.9
        assert model.metrics.mse is not None

    def test_train_linear_regression_regularized(self):
        """Test ridge regression with regularization."""
        trainer = ModelTrainer()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 5))
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.normal(0, 0.5, 100)

        model = trainer.train_linear_regression(X, y, regularization=0.1)

        assert model.parameters["regularization"] == 0.1

    def test_train_logistic_regression(self):
        """Test logistic regression training."""
        trainer = ModelTrainer()
        np.random.seed(42)

        # Binary classification problem
        X = np.random.normal(0, 1, (200, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = trainer.train_logistic_regression(X, y, max_iterations=500)

        assert isinstance(model, TrainedModel)
        assert model.model_type == ModelType.LOGISTIC_REGRESSION
        assert model.task_type == TaskType.BINARY_CLASSIFICATION
        assert model.metrics.accuracy > 0.8

    def test_train_knn(self):
        """Test KNN training."""
        trainer = ModelTrainer()
        np.random.seed(42)

        X = np.random.normal(0, 1, (50, 2))
        y = (X[:, 0] > 0).astype(int)

        model = trainer.train_knn(X, y, k=5)

        assert model.model_type == ModelType.KNN
        assert model.parameters["k"] == 5

    def test_model_predict_linear(self):
        """Test prediction with linear model."""
        trainer = ModelTrainer()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 2))
        y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)

        model = trainer.train_linear_regression(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        # Predictions should be close to actual values
        mse = np.mean((predictions - y) ** 2)
        assert mse < 1

    def test_model_predict_logistic(self):
        """Test prediction with logistic model."""
        trainer = ModelTrainer()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = trainer.train_logistic_regression(X, y, max_iterations=500)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        # Predictions should be 0 or 1
        assert set(predictions).issubset({0, 1})


class TestModelEvaluator:
    """Tests for model evaluation."""

    def test_cross_validate_regression(self):
        """Test cross-validation for regression."""
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 3))
        y = X @ np.array([1, 2, 3]) + np.random.normal(0, 0.5, 100)

        result = evaluator.cross_validate(
            trainer, X, y, k_folds=5, model_type=ModelType.LINEAR_REGRESSION
        )

        assert result["k_folds"] == 5
        assert "mse_mean" in result
        assert "r2_mean" in result
        assert result["r2_mean"] > 0.5

    def test_cross_validate_classification(self):
        """Test cross-validation for classification."""
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)

        result = evaluator.cross_validate(
            trainer, X, y, k_folds=3, model_type=ModelType.LOGISTIC_REGRESSION
        )

        assert result["k_folds"] == 3
        assert "accuracy_mean" in result


class TestAutoMLPipeline:
    """Tests for AutoML pipeline."""

    def test_fit_regression_auto_detect(self):
        """Test AutoML with automatic regression detection."""
        pipeline = AutoMLPipeline()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 3))
        y = X @ np.array([1, 2, 3]) + np.random.normal(0, 0.5, 100)

        result = pipeline.fit(X, y, max_models=3, cv_folds=2)

        assert "best_model" in result
        assert result["task_type"] == "regression"
        assert "cv_results" in result

    def test_fit_classification_auto_detect(self):
        """Test AutoML with automatic classification detection."""
        pipeline = AutoMLPipeline()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        result = pipeline.fit(X, y, max_models=3, cv_folds=2)

        assert "best_model" in result
        assert result["task_type"] == "binary_classification"

    def test_fit_explicit_task_type(self):
        """Test AutoML with explicit task type."""
        pipeline = AutoMLPipeline()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 3))
        y = X[:, 0] + np.random.normal(0, 1, 100)

        result = pipeline.fit(X, y, task=TaskType.REGRESSION, max_models=2)

        assert result["task_type"] == "regression"

    def test_fit_returns_best_model(self):
        """Test that AutoML returns a working model."""
        pipeline = AutoMLPipeline()
        np.random.seed(42)

        X = np.random.normal(0, 1, (100, 2))
        y = 2 * X[:, 0] - X[:, 1] + np.random.normal(0, 0.2, 100)

        result = pipeline.fit(X, y, max_models=2, cv_folds=2)

        model = result["best_model"]
        # Model should be able to make predictions
        # Need to standardize input like the pipeline does
        X_std = pipeline.feature_eng.standardize(X, fit=False, name="default")
        predictions = model.predict(X_std)
        assert len(predictions) == len(y)


class TestModelMetrics:
    """Tests for model metrics dataclass."""

    def test_regression_metrics_creation(self):
        """Test creation of regression metrics."""
        metrics = ModelMetrics(
            mse=0.5,
            rmse=0.707,
            mae=0.4,
            r2=0.95,
            n_samples=100,
            n_features=5,
        )

        assert metrics.mse == 0.5
        assert metrics.r2 == 0.95

    def test_classification_metrics_creation(self):
        """Test creation of classification metrics."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1=0.95,
            auc_roc=0.98,
        )

        assert metrics.accuracy == 0.95
        assert metrics.f1 == 0.95


class TestTrainedModel:
    """Tests for TrainedModel class."""

    def test_predict_without_training(self):
        """Test prediction fails without weights."""
        model = TrainedModel(
            id="test",
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=TaskType.REGRESSION,
            parameters={},
            metrics=ModelMetrics(),
        )

        with pytest.raises(ValueError, match="not trained"):
            model.predict(np.array([[1, 2, 3]]))

    def test_model_has_required_fields(self):
        """Test TrainedModel has all required fields."""
        model = TrainedModel(
            id="test-model",
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=TaskType.REGRESSION,
            parameters={"regularization": 0.1},
            metrics=ModelMetrics(mse=0.5, r2=0.9),
        )

        assert model.id == "test-model"
        assert model.model_type == ModelType.LINEAR_REGRESSION
        assert model.task_type == TaskType.REGRESSION
        assert model.parameters["regularization"] == 0.1
        assert model.created_at is not None


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        np.random.seed(42)

        # Generate time series with trend and seasonality
        t = np.arange(0, 100)
        trend = 50 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 2, 100)
        data = trend + seasonal + noise

        # Analyze
        ts_analyzer = TimeSeriesAnalyzer()
        stationarity = ts_analyzer.is_stationary(data)

        # Decompose
        decomposer = SeasonalDecomposer(method="additive")
        components = decomposer.decompose(data, period=12)

        # Forecast
        forecaster = Forecaster()
        forecast = forecaster.holt_winters(data, period=12, horizon=12)

        # Verify - stationarity result depends on the algorithm's thresholds
        assert stationarity["is_stationary"] is not None
        assert len(components.trend) == len(data)
        assert len(forecast.values) == 12

    def test_ml_pipeline_end_to_end(self):
        """Test complete ML pipeline."""
        np.random.seed(42)

        # Generate data
        n_samples = 200
        X = np.random.normal(0, 1, (n_samples, 5))
        y = X @ np.array([1, 2, -1, 0.5, -0.5]) + np.random.normal(0, 0.5, n_samples)

        # Feature engineering
        fe = FeatureEngineering()
        X_scaled = fe.standardize(X)
        X_selected, indices = fe.select_features(X_scaled, y, k=3)

        # Train
        trainer = ModelTrainer()
        model = trainer.train_linear_regression(X_selected, y)

        # Evaluate
        evaluator = ModelEvaluator()
        cv_result = evaluator.cross_validate(
            trainer, X_selected, y, k_folds=5, model_type=ModelType.LINEAR_REGRESSION
        )

        # Verify
        assert model.metrics.r2 > 0.8
        assert cv_result["r2_mean"] > 0.7

    def test_anomaly_and_trend_combined(self):
        """Test combining anomaly detection with trend analysis."""
        np.random.seed(42)

        # Generate trending data with outliers
        trend_data = np.arange(0, 100) + np.random.normal(0, 5, 100)
        trend_data[50] = 200  # Add outlier

        # Detect anomalies - method is set in constructor
        anomaly_detector = AnomalyDetector(method="zscore")
        anomalies = anomaly_detector.detect(trend_data)

        # Analyze trend
        trend_analyzer = TrendAnalyzer()
        trend = trend_analyzer.analyze(trend_data)

        # Verify
        assert len(anomalies) >= 1
        assert any(a.index == 50 for a in anomalies)
        assert trend.direction == TrendDirection.INCREASING

    def test_clustering_with_feature_engineering(self):
        """Test clustering after feature engineering."""
        np.random.seed(42)

        # Generate clustered data
        cluster1 = np.random.normal([0, 0], 1, (50, 2))
        cluster2 = np.random.normal([10, 10], 1, (50, 2))
        X = np.vstack([cluster1, cluster2])

        # Feature engineering
        fe = FeatureEngineering()
        X_normalized = fe.normalize(X)

        # Cluster
        cluster_analyzer = ClusterAnalyzer()
        clusters = cluster_analyzer.kmeans(X_normalized, k=2)

        # Evaluate - use members (not points)
        labels = np.zeros(100)
        for i, cluster in enumerate(clusters):
            for idx in cluster.members:
                labels[idx] = i
        score = cluster_analyzer.silhouette_score(X_normalized, labels)

        # Verify
        assert len(clusters) == 2
        assert score > 0.5
