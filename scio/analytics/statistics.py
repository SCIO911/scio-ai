"""
SCIO Statistics Engine

Fortgeschrittene statistische Analysen.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
import numpy as np

from scio.core.logging import get_logger

logger = get_logger(__name__)


class TestType(str, Enum):
    """Arten von statistischen Tests."""
    T_TEST = "t_test"
    PAIRED_T_TEST = "paired_t_test"
    ANOVA = "anova"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    SHAPIRO_WILK = "shapiro_wilk"


@dataclass
class DescriptiveStats:
    """Deskriptive Statistiken für eine Datenreihe."""

    count: int
    mean: float
    median: float
    mode: Optional[float]
    std: float
    variance: float
    min: float
    max: float
    range: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    sum: float
    missing: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "std": self.std,
            "variance": self.variance,
            "min": self.min,
            "max": self.max,
            "range": self.range,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "sum": self.sum,
            "missing": self.missing,
        }

    def summary(self) -> str:
        """Gibt eine Zusammenfassung zurück."""
        return (
            f"n={self.count}, μ={self.mean:.4f}, σ={self.std:.4f}, "
            f"range=[{self.min:.4f}, {self.max:.4f}]"
        )


@dataclass
class HypothesisTest:
    """Ergebnis eines Hypothesentests."""

    test_type: TestType
    statistic: float
    p_value: float
    alpha: float = 0.05
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    sample_sizes: list[int] = field(default_factory=list)
    effect_size: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None

    @property
    def significant(self) -> bool:
        """Prüft ob das Ergebnis signifikant ist."""
        return self.p_value < self.alpha

    @property
    def stars(self) -> str:
        """Gibt Signifikanzsterne zurück."""
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        else:
            return ""

    def interpretation(self) -> str:
        """Gibt eine Interpretation zurück."""
        if self.significant:
            return (
                f"Signifikant (p={self.p_value:.4f}{self.stars}): "
                f"Nullhypothese wird abgelehnt."
            )
        else:
            return (
                f"Nicht signifikant (p={self.p_value:.4f}): "
                f"Nullhypothese kann nicht abgelehnt werden."
            )


@dataclass
class CorrelationAnalysis:
    """Korrelationsanalyse zwischen Variablen."""

    method: str  # pearson, spearman, kendall
    coefficient: float
    p_value: float
    n: int
    confidence_interval: Optional[tuple[float, float]] = None

    @property
    def strength(self) -> str:
        """Gibt die Stärke der Korrelation zurück."""
        r = abs(self.coefficient)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        else:
            return "very strong"

    @property
    def direction(self) -> str:
        """Gibt die Richtung der Korrelation zurück."""
        if self.coefficient > 0:
            return "positive"
        elif self.coefficient < 0:
            return "negative"
        else:
            return "none"


class StatisticsEngine:
    """
    Leistungsstarke Statistik-Engine für SCIO.

    Features:
    - Deskriptive Statistiken
    - Inferenzstatistik
    - Korrelationsanalysen
    - Hypothesentests
    - Effektstärkenberechnung
    """

    def __init__(self):
        logger.info("StatisticsEngine initialized")

    def describe(self, data: Union[list, np.ndarray]) -> DescriptiveStats:
        """Berechnet deskriptive Statistiken."""
        arr = np.array(data, dtype=float)
        valid = arr[~np.isnan(arr)]
        n = len(valid)

        if n == 0:
            return DescriptiveStats(
                count=0, mean=0, median=0, mode=None, std=0, variance=0,
                min=0, max=0, range=0, q1=0, q3=0, iqr=0,
                skewness=0, kurtosis=0, sum=0, missing=len(arr),
            )

        mean = float(np.mean(valid))
        std = float(np.std(valid, ddof=1)) if n > 1 else 0
        q1, median, q3 = np.percentile(valid, [25, 50, 75])

        # Mode (häufigster Wert)
        unique, counts = np.unique(valid, return_counts=True)
        mode = float(unique[np.argmax(counts)]) if len(unique) > 0 else None

        # Skewness
        if std > 0 and n > 2:
            skewness = float(np.mean(((valid - mean) / std) ** 3))
        else:
            skewness = 0.0

        # Kurtosis (excess kurtosis)
        if std > 0 and n > 3:
            kurtosis = float(np.mean(((valid - mean) / std) ** 4)) - 3
        else:
            kurtosis = 0.0

        return DescriptiveStats(
            count=n,
            mean=mean,
            median=float(median),
            mode=mode,
            std=std,
            variance=std ** 2,
            min=float(np.min(valid)),
            max=float(np.max(valid)),
            range=float(np.max(valid) - np.min(valid)),
            q1=float(q1),
            q3=float(q3),
            iqr=float(q3 - q1),
            skewness=skewness,
            kurtosis=kurtosis,
            sum=float(np.sum(valid)),
            missing=len(arr) - n,
        )

    def t_test(
        self,
        sample1: Union[list, np.ndarray],
        sample2: Optional[Union[list, np.ndarray]] = None,
        mu: float = 0,
        paired: bool = False,
        alpha: float = 0.05,
    ) -> HypothesisTest:
        """
        Führt einen t-Test durch.

        Args:
            sample1: Erste Stichprobe
            sample2: Zweite Stichprobe (für Zwei-Stichproben-Test)
            mu: Hypothesenwert (für Ein-Stichproben-Test)
            paired: Ob der Test gepaart ist
            alpha: Signifikanzniveau
        """
        arr1 = np.array(sample1, dtype=float)
        arr1 = arr1[~np.isnan(arr1)]
        n1 = len(arr1)

        if sample2 is None:
            # Ein-Stichproben t-Test
            mean1 = np.mean(arr1)
            se = np.std(arr1, ddof=1) / np.sqrt(n1)
            t_stat = (mean1 - mu) / se
            df = n1 - 1
            test_type = TestType.T_TEST
            null_h = f"μ = {mu}"
            alt_h = f"μ ≠ {mu}"
            sizes = [n1]
        else:
            arr2 = np.array(sample2, dtype=float)
            arr2 = arr2[~np.isnan(arr2)]
            n2 = len(arr2)

            if paired:
                # Gepaarter t-Test
                if n1 != n2:
                    raise ValueError("Für gepaarten Test müssen die Stichproben gleich groß sein")
                diff = arr1 - arr2
                mean_diff = np.mean(diff)
                se = np.std(diff, ddof=1) / np.sqrt(n1)
                t_stat = mean_diff / se
                df = n1 - 1
                test_type = TestType.PAIRED_T_TEST
                null_h = "μ_d = 0"
                alt_h = "μ_d ≠ 0"
                sizes = [n1]
            else:
                # Unabhängiger Zwei-Stichproben t-Test (Welch)
                mean1, mean2 = np.mean(arr1), np.mean(arr2)
                var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
                se = np.sqrt(var1/n1 + var2/n2)
                t_stat = (mean1 - mean2) / se

                # Welch-Satterthwaite Freiheitsgrade
                num = (var1/n1 + var2/n2)**2
                denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
                df = num / denom

                test_type = TestType.T_TEST
                null_h = "μ₁ = μ₂"
                alt_h = "μ₁ ≠ μ₂"
                sizes = [n1, n2]

        # P-Wert berechnen (zweiseitig)
        p_value = 2 * (1 - self._t_cdf(abs(float(t_stat)), df))

        # Effektstärke (Cohen's d)
        if sample2 is not None and not paired:
            pooled_std = np.sqrt(((n1-1)*np.var(arr1, ddof=1) + (n2-1)*np.var(arr2, ddof=1)) / (n1+n2-2))
            effect_size = (np.mean(arr1) - np.mean(arr2)) / pooled_std
        else:
            effect_size = float(t_stat) / np.sqrt(n1)

        return HypothesisTest(
            test_type=test_type,
            statistic=float(t_stat),
            p_value=p_value,
            alpha=alpha,
            null_hypothesis=null_h,
            alternative_hypothesis=alt_h,
            sample_sizes=sizes,
            effect_size=float(effect_size),
        )

    def _t_cdf(self, t: float, df: float) -> float:
        """Approximation der t-Verteilung CDF."""
        x = df / (df + t * t)
        return 1 - 0.5 * self._incomplete_beta(df / 2, 0.5, x)

    def _incomplete_beta(self, a: float, b: float, x: float) -> float:
        """Approximation der unvollständigen Beta-Funktion."""
        if x == 0:
            return 0
        if x == 1:
            return 1

        # Continued fraction approximation
        max_iter = 100
        eps = 1e-10

        qab = a + b
        qap = a + 1
        qam = a - 1
        c = 1.0
        d = 1 - qab * x / qap
        if abs(d) < eps:
            d = eps
        d = 1 / d
        h = d

        for m in range(1, max_iter):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d
            if abs(d) < eps:
                d = eps
            c = 1 + aa / c
            if abs(c) < eps:
                c = eps
            d = 1 / d
            h *= d * c

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d
            if abs(d) < eps:
                d = eps
            c = 1 + aa / c
            if abs(c) < eps:
                c = eps
            d = 1 / d
            delta = d * c
            h *= delta

            if abs(delta - 1) < eps:
                break

        front = math.exp(
            a * math.log(x) + b * math.log(1 - x) +
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        )
        return front * h / a

    def correlation(
        self,
        x: Union[list, np.ndarray],
        y: Union[list, np.ndarray],
        method: str = "pearson",
    ) -> CorrelationAnalysis:
        """
        Berechnet die Korrelation zwischen zwei Variablen.

        Args:
            x: Erste Variable
            y: Zweite Variable
            method: pearson, spearman, oder kendall
        """
        arr_x = np.array(x, dtype=float)
        arr_y = np.array(y, dtype=float)

        # Entferne NaN-Paare
        valid = ~(np.isnan(arr_x) | np.isnan(arr_y))
        arr_x = arr_x[valid]
        arr_y = arr_y[valid]
        n = len(arr_x)

        if n < 3:
            return CorrelationAnalysis(method=method, coefficient=0, p_value=1, n=n)

        if method == "pearson":
            r = float(np.corrcoef(arr_x, arr_y)[0, 1])
        elif method == "spearman":
            # Rangkorrelation
            rank_x = np.argsort(np.argsort(arr_x))
            rank_y = np.argsort(np.argsort(arr_y))
            r = float(np.corrcoef(rank_x, rank_y)[0, 1])
        elif method == "kendall":
            # Kendall's Tau
            concordant = 0
            discordant = 0
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi = arr_x[i], arr_y[i]
                    xj, yj = arr_x[j], arr_y[j]
                    if (xi < xj and yi < yj) or (xi > xj and yi > yj):
                        concordant += 1
                    elif (xi < xj and yi > yj) or (xi > xj and yi < yj):
                        discordant += 1
            r = (concordant - discordant) / (n * (n - 1) / 2)
        else:
            raise ValueError(f"Unbekannte Methode: {method}")

        # P-Wert (Approximation für Pearson/Spearman)
        if abs(r) >= 1:
            p_value = 0.0
        else:
            t_stat = r * np.sqrt((n - 2) / (1 - r * r))
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 2))

        # Konfidenzintervall (Fisher z-Transformation)
        if abs(r) < 1:
            z = 0.5 * np.log((1 + r) / (1 - r))
            se = 1 / np.sqrt(n - 3)
            z_crit = 1.96  # 95% CI
            z_low, z_high = z - z_crit * se, z + z_crit * se
            r_low = (np.exp(2 * z_low) - 1) / (np.exp(2 * z_low) + 1)
            r_high = (np.exp(2 * z_high) - 1) / (np.exp(2 * z_high) + 1)
            ci = (float(r_low), float(r_high))
        else:
            ci = (r, r)

        return CorrelationAnalysis(
            method=method,
            coefficient=r,
            p_value=p_value,
            n=n,
            confidence_interval=ci,
        )

    def correlation_matrix(
        self,
        data: dict[str, Union[list, np.ndarray]],
        method: str = "pearson",
    ) -> dict[str, dict[str, CorrelationAnalysis]]:
        """Berechnet eine Korrelationsmatrix für mehrere Variablen."""
        keys = list(data.keys())
        matrix = {}

        for k1 in keys:
            matrix[k1] = {}
            for k2 in keys:
                matrix[k1][k2] = self.correlation(data[k1], data[k2], method)

        return matrix

    def anova(
        self,
        *groups: Union[list, np.ndarray],
        alpha: float = 0.05,
    ) -> HypothesisTest:
        """
        Führt eine einfaktorielle ANOVA durch.

        Args:
            *groups: Zwei oder mehr Gruppen zum Vergleich
            alpha: Signifikanzniveau
        """
        k = len(groups)
        if k < 2:
            raise ValueError("ANOVA benötigt mindestens 2 Gruppen")

        arrays = [np.array(g, dtype=float) for g in groups]
        arrays = [arr[~np.isnan(arr)] for arr in arrays]
        ns = [len(arr) for arr in arrays]
        n_total = sum(ns)

        # Gesamtmittel
        grand_mean = np.mean(np.concatenate(arrays))

        # Between-Groups Varianz
        ss_between = sum(n * (np.mean(arr) - grand_mean) ** 2 for n, arr in zip(ns, arrays))
        df_between = k - 1

        # Within-Groups Varianz
        ss_within = sum(np.sum((arr - np.mean(arr)) ** 2) for arr in arrays)
        df_within = n_total - k

        # F-Statistik
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_stat = ms_between / ms_within

        # P-Wert (F-Verteilung Approximation)
        p_value = 1 - self._f_cdf(f_stat, df_between, df_within)

        # Effektstärke (Eta-squared)
        eta_squared = ss_between / (ss_between + ss_within)

        return HypothesisTest(
            test_type=TestType.ANOVA,
            statistic=f_stat,
            p_value=p_value,
            alpha=alpha,
            null_hypothesis="Alle Gruppenmittelwerte sind gleich",
            alternative_hypothesis="Mindestens ein Gruppenmittelwert ist unterschiedlich",
            sample_sizes=ns,
            effect_size=eta_squared,
        )

    def _f_cdf(self, f: float, df1: float, df2: float) -> float:
        """Approximation der F-Verteilung CDF."""
        x = df2 / (df2 + df1 * f)
        return 1 - self._incomplete_beta(df2 / 2, df1 / 2, x)

    def chi_square(
        self,
        observed: Union[list, np.ndarray],
        expected: Optional[Union[list, np.ndarray]] = None,
        alpha: float = 0.05,
    ) -> HypothesisTest:
        """
        Chi-Quadrat-Test.

        Args:
            observed: Beobachtete Häufigkeiten
            expected: Erwartete Häufigkeiten (wenn None, gleichverteilt)
            alpha: Signifikanzniveau
        """
        obs = np.array(observed, dtype=float)
        n = np.sum(obs)

        if expected is None:
            exp = np.full_like(obs, n / len(obs))
        else:
            exp = np.array(expected, dtype=float)

        # Chi-Quadrat-Statistik
        chi2 = float(np.sum((obs - exp) ** 2 / exp))
        df = len(obs) - 1

        # P-Wert (Chi-Quadrat-Verteilung Approximation)
        p_value = 1 - self._chi2_cdf(chi2, df)

        return HypothesisTest(
            test_type=TestType.CHI_SQUARE,
            statistic=chi2,
            p_value=p_value,
            alpha=alpha,
            null_hypothesis="Beobachtete = Erwartete Häufigkeiten",
            alternative_hypothesis="Beobachtete ≠ Erwartete Häufigkeiten",
            sample_sizes=[int(n)],
        )

    def _chi2_cdf(self, x: float, df: float) -> float:
        """Approximation der Chi-Quadrat-Verteilung CDF."""
        if x <= 0:
            return 0
        return self._incomplete_gamma(df / 2, x / 2)

    def _incomplete_gamma(self, a: float, x: float) -> float:
        """Approximation der unvollständigen Gamma-Funktion (regularisiert)."""
        if x < 0 or a <= 0:
            return 0

        if x < a + 1:
            # Reihenentwicklung
            sum_val = 1 / a
            term = 1 / a
            for n in range(1, 100):
                term *= x / (a + n)
                sum_val += term
                if abs(term) < 1e-10:
                    break
            return sum_val * math.exp(-x + a * math.log(x) - math.lgamma(a))
        else:
            # Kettenbruch
            b = x + 1 - a
            c = 1e30
            d = 1 / b
            h = d
            for i in range(1, 100):
                an = -i * (i - a)
                b += 2
                d = an * d + b
                if abs(d) < 1e-30:
                    d = 1e-30
                c = b + an / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1 / d
                delta = d * c
                h *= delta
                if abs(delta - 1) < 1e-10:
                    break
            return 1 - h * math.exp(-x + a * math.log(x) - math.lgamma(a))

    def normality_test(
        self,
        data: Union[list, np.ndarray],
        alpha: float = 0.05,
    ) -> HypothesisTest:
        """
        Prüft auf Normalverteilung (Shapiro-Wilk-ähnlich).
        """
        arr = np.array(data, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 3:
            return HypothesisTest(
                test_type=TestType.SHAPIRO_WILK,
                statistic=0,
                p_value=1,
                alpha=alpha,
                null_hypothesis="Daten sind normalverteilt",
                alternative_hypothesis="Daten sind nicht normalverteilt",
                sample_sizes=[n],
            )

        # Sortierte Werte
        x = np.sort(arr)

        # Erwartete Werte unter Normalverteilung
        m = np.zeros(n)
        for i in range(n):
            p = (i + 0.5) / n
            m[i] = self._norm_ppf(p)

        # Korrelation als Test-Statistik
        r = np.corrcoef(x, m)[0, 1]
        w = r ** 2

        # Approximation des p-Werts
        mean_w = 0.0038 * n + 0.474
        std_w = 0.0008 * n + 0.067
        z = (w - mean_w) / std_w
        p_value = 2 * (1 - self._norm_cdf(abs(z)))

        return HypothesisTest(
            test_type=TestType.SHAPIRO_WILK,
            statistic=w,
            p_value=p_value,
            alpha=alpha,
            null_hypothesis="Daten sind normalverteilt",
            alternative_hypothesis="Daten sind nicht normalverteilt",
            sample_sizes=[n],
        )

    def _norm_cdf(self, x: float) -> float:
        """Standardnormalverteilung CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_ppf(self, p: float) -> float:
        """Inverse der Standardnormalverteilung (Approximation)."""
        if p <= 0:
            return float('-inf')
        if p >= 1:
            return float('inf')

        # Rational approximation
        a = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00
        ]
        b = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01
        ]
        c = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00
        ]
        d = [
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00
        ]

        p_low = 0.02425
        p_high = 1 - p_low

        if p < p_low:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
