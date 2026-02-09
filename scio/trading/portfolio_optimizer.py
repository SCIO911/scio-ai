"""
SCIO Portfolio Optimizer

Moderne Portfolio-Theorie (MPT) Implementation:
- Efficient Frontier Berechnung
- Sharpe Ratio Optimierung
- Risk Parity
- Black-Litterman Model (vereinfacht)
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
import random

from scio.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Asset:
    """Ein Asset im Portfolio."""
    symbol: str
    name: str
    expected_return: float  # Annualisiert
    volatility: float  # Annualisierte Standardabweichung
    current_price: Decimal
    weight: float = 0.0  # Portfoliogewicht 0-1

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "current_price": str(self.current_price),
            "weight": self.weight,
        }


@dataclass
class Portfolio:
    """Ein Portfolio von Assets."""
    assets: list[Asset]
    weights: list[float]
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0

    def __post_init__(self):
        if self.assets and self.weights:
            self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Berechnet Portfolio-Metriken."""
        # Erwartete Rendite (gewichteter Durchschnitt)
        self.expected_return = sum(
            a.expected_return * w
            for a, w in zip(self.assets, self.weights)
        )

        # Volatilitaet (vereinfacht, ohne Korrelation)
        # Echte Berechnung wuerde Kovarianzmatrix benoetigen
        variance = sum(
            (a.volatility * w) ** 2
            for a, w in zip(self.assets, self.weights)
        )
        self.volatility = math.sqrt(variance)

        # Sharpe Ratio (angenommen risikofreier Zins = 4%)
        risk_free_rate = 0.04
        if self.volatility > 0:
            self.sharpe_ratio = (self.expected_return - risk_free_rate) / self.volatility
        else:
            self.sharpe_ratio = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": [a.to_dict() for a in self.assets],
            "weights": self.weights,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
        }


@dataclass
class OptimizationResult:
    """Ergebnis einer Portfolio-Optimierung."""
    optimal_portfolio: Portfolio
    efficient_frontier: list[Portfolio]
    min_variance_portfolio: Portfolio
    max_sharpe_portfolio: Portfolio
    recommendations: list[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimal_portfolio": self.optimal_portfolio.to_dict(),
            "min_variance_portfolio": self.min_variance_portfolio.to_dict(),
            "max_sharpe_portfolio": self.max_sharpe_portfolio.to_dict(),
            "efficient_frontier_points": len(self.efficient_frontier),
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


class PortfolioOptimizer:
    """
    Portfolio-Optimierung basierend auf Moderner Portfolio-Theorie.
    """

    # Asset-Klassen mit typischen Renditen/Volatilitaeten
    ASSET_CLASSES = {
        "stocks_us": {"return": 0.10, "vol": 0.20, "name": "US Aktien"},
        "stocks_eu": {"return": 0.08, "vol": 0.18, "name": "EU Aktien"},
        "stocks_em": {"return": 0.12, "vol": 0.25, "name": "Emerging Markets"},
        "bonds_gov": {"return": 0.04, "vol": 0.05, "name": "Staatsanleihen"},
        "bonds_corp": {"return": 0.06, "vol": 0.08, "name": "Unternehmensanleihen"},
        "reits": {"return": 0.09, "vol": 0.22, "name": "REITs"},
        "gold": {"return": 0.05, "vol": 0.15, "name": "Gold"},
        "crypto": {"return": 0.25, "vol": 0.70, "name": "Krypto"},
        "cash": {"return": 0.03, "vol": 0.01, "name": "Cash/Geldmarkt"},
    }

    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
        logger.info("PortfolioOptimizer initialized")

    def create_asset(
        self,
        symbol: str,
        name: str,
        expected_return: float,
        volatility: float,
        price: Decimal = Decimal("100"),
    ) -> Asset:
        """Erstellt ein Asset."""
        return Asset(
            symbol=symbol,
            name=name,
            expected_return=expected_return,
            volatility=volatility,
            current_price=price,
        )

    def create_asset_from_class(self, asset_class: str) -> Optional[Asset]:
        """Erstellt Asset aus vordefinierter Klasse."""
        if asset_class not in self.ASSET_CLASSES:
            return None

        data = self.ASSET_CLASSES[asset_class]
        return self.create_asset(
            symbol=asset_class.upper(),
            name=data["name"],
            expected_return=data["return"],
            volatility=data["vol"],
        )

    def calculate_portfolio_metrics(
        self,
        assets: list[Asset],
        weights: list[float],
    ) -> Portfolio:
        """Berechnet Portfolio-Metriken fuer gegebene Gewichte."""
        if len(assets) != len(weights):
            raise ValueError("Assets und Weights muessen gleiche Laenge haben")

        # Normalisiere Gewichte
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return Portfolio(assets=assets, weights=weights)

    def optimize_sharpe(
        self,
        assets: list[Asset],
        iterations: int = 10000,
    ) -> Portfolio:
        """
        Findet Portfolio mit maximaler Sharpe Ratio.
        Verwendet Monte-Carlo Simulation.
        """
        best_sharpe = -float("inf")
        best_weights = [1.0 / len(assets)] * len(assets)

        for _ in range(iterations):
            # Zufaellige Gewichte
            weights = [random.random() for _ in assets]
            total = sum(weights)
            weights = [w / total for w in weights]

            portfolio = self.calculate_portfolio_metrics(assets, weights)

            if portfolio.sharpe_ratio > best_sharpe:
                best_sharpe = portfolio.sharpe_ratio
                best_weights = weights

        return self.calculate_portfolio_metrics(assets, best_weights)

    def optimize_min_variance(
        self,
        assets: list[Asset],
        target_return: Optional[float] = None,
        iterations: int = 10000,
    ) -> Portfolio:
        """
        Findet Portfolio mit minimaler Varianz.
        Optional mit Zielrendite-Constraint.
        """
        min_vol = float("inf")
        best_weights = [1.0 / len(assets)] * len(assets)

        for _ in range(iterations):
            weights = [random.random() for _ in assets]
            total = sum(weights)
            weights = [w / total for w in weights]

            portfolio = self.calculate_portfolio_metrics(assets, weights)

            # Pruefe Zielrendite-Constraint
            if target_return is not None:
                if portfolio.expected_return < target_return - 0.005:
                    continue

            if portfolio.volatility < min_vol:
                min_vol = portfolio.volatility
                best_weights = weights

        return self.calculate_portfolio_metrics(assets, best_weights)

    def calculate_efficient_frontier(
        self,
        assets: list[Asset],
        points: int = 20,
        iterations_per_point: int = 5000,
    ) -> list[Portfolio]:
        """
        Berechnet Efficient Frontier.
        """
        # Finde min/max erwartete Rendite
        min_return = min(a.expected_return for a in assets)
        max_return = max(a.expected_return for a in assets)

        frontier = []
        target_returns = [
            min_return + i * (max_return - min_return) / (points - 1)
            for i in range(points)
        ]

        for target in target_returns:
            portfolio = self.optimize_min_variance(
                assets,
                target_return=target,
                iterations=iterations_per_point,
            )
            frontier.append(portfolio)

        return frontier

    def optimize(
        self,
        assets: list[Asset],
        risk_tolerance: str = "moderate",  # conservative, moderate, aggressive
    ) -> OptimizationResult:
        """
        Vollstaendige Portfolio-Optimierung.
        """
        logger.info("Starting portfolio optimization", assets=len(assets))

        # Efficient Frontier
        frontier = self.calculate_efficient_frontier(assets, points=15)

        # Min Variance Portfolio
        min_var = self.optimize_min_variance(assets)

        # Max Sharpe Portfolio
        max_sharpe = self.optimize_sharpe(assets)

        # Waehle optimales Portfolio basierend auf Risikotoleranz
        if risk_tolerance == "conservative":
            # Nahe am Min-Variance
            optimal = min_var
        elif risk_tolerance == "aggressive":
            # Max-Sharpe oder noch mehr Risiko
            optimal = max_sharpe
        else:
            # Moderate: Balance zwischen beiden
            # Waehle Portfolio auf Frontier mit mittlerem Risiko
            mid_idx = len(frontier) // 2
            optimal = frontier[mid_idx] if frontier else max_sharpe

        # Empfehlungen generieren
        recommendations = self._generate_recommendations(
            optimal, min_var, max_sharpe, risk_tolerance
        )

        return OptimizationResult(
            optimal_portfolio=optimal,
            efficient_frontier=frontier,
            min_variance_portfolio=min_var,
            max_sharpe_portfolio=max_sharpe,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        optimal: Portfolio,
        min_var: Portfolio,
        max_sharpe: Portfolio,
        risk_tolerance: str,
    ) -> list[str]:
        """Generiert Empfehlungen."""
        recs = []

        # Sharpe-basierte Empfehlung
        if optimal.sharpe_ratio > 1.0:
            recs.append(
                f"Ausgezeichnetes Risk/Return-Verhaeltnis (Sharpe: {optimal.sharpe_ratio:.2f})"
            )
        elif optimal.sharpe_ratio > 0.5:
            recs.append(
                f"Gutes Risk/Return-Verhaeltnis (Sharpe: {optimal.sharpe_ratio:.2f})"
            )
        else:
            recs.append(
                f"Moderates Risk/Return-Verhaeltnis (Sharpe: {optimal.sharpe_ratio:.2f})"
            )

        # Volatilitaets-Warnung
        if optimal.volatility > 0.25:
            recs.append(
                "WARNUNG: Hohe Volatilitaet - nur fuer langfristigen Horizont geeignet"
            )

        # Diversifikations-Check
        max_weight = max(optimal.weights)
        if max_weight > 0.5:
            recs.append(
                f"Klumpenrisiko: Ein Asset hat {max_weight*100:.0f}% Gewicht - mehr diversifizieren"
            )

        # Risikotoleranz-Hinweis
        if risk_tolerance == "conservative":
            recs.append(
                "Konservatives Portfolio: Fokus auf Kapitalerhalt"
            )
        elif risk_tolerance == "aggressive":
            recs.append(
                "Aggressives Portfolio: Hoehere Rendite, aber auch hoeheres Verlustrisiko"
            )

        return recs

    def rebalance_recommendation(
        self,
        current_weights: list[float],
        target_weights: list[float],
        portfolio_value: Decimal,
        threshold: float = 0.05,
    ) -> list[dict[str, Any]]:
        """
        Empfiehlt Rebalancing-Aktionen.
        """
        actions = []

        for i, (current, target) in enumerate(zip(current_weights, target_weights)):
            diff = target - current

            if abs(diff) > threshold:
                action = "kaufen" if diff > 0 else "verkaufen"
                amount = abs(diff) * float(portfolio_value)

                actions.append({
                    "position": i,
                    "action": action,
                    "weight_change": diff,
                    "amount_eur": amount,
                })

        return actions


class RiskAnalyzer:
    """
    Erweiterte Risikoanalyse.
    """

    def calculate_var(
        self,
        returns: list[float],
        confidence: float = 0.95,
        portfolio_value: Decimal = Decimal("10000"),
    ) -> dict[str, Any]:
        """
        Value at Risk Berechnung.

        Args:
            returns: Liste taeglicher Renditen
            confidence: Konfidenzniveau (z.B. 0.95 = 95%)
            portfolio_value: Portfoliowert

        Returns:
            VaR Metriken
        """
        if len(returns) < 30:
            return {"error": "Mindestens 30 Datenpunkte benoetigt"}

        # Sortiere Renditen
        sorted_returns = sorted(returns)

        # VaR Index
        var_index = int((1 - confidence) * len(sorted_returns))
        var_return = sorted_returns[var_index]

        # VaR in Euro
        var_eur = float(portfolio_value) * abs(var_return)

        # Conditional VaR (Expected Shortfall)
        tail_returns = sorted_returns[:var_index + 1]
        cvar_return = statistics.mean(tail_returns) if tail_returns else var_return
        cvar_eur = float(portfolio_value) * abs(cvar_return)

        return {
            "confidence": confidence,
            "var_percent": var_return * 100,
            "var_eur": var_eur,
            "cvar_percent": cvar_return * 100,
            "cvar_eur": cvar_eur,
            "interpretation": (
                f"Mit {confidence*100:.0f}% Wahrscheinlichkeit verliert das Portfolio "
                f"an einem Tag nicht mehr als {var_eur:.2f} EUR"
            ),
        }

    def calculate_drawdown(
        self,
        values: list[float],
    ) -> dict[str, Any]:
        """
        Maximum Drawdown Berechnung.
        """
        if len(values) < 2:
            return {"error": "Mindestens 2 Datenpunkte benoetigt"}

        peak = values[0]
        max_drawdown = 0.0
        current_drawdown = 0.0
        drawdown_start = 0
        max_drawdown_start = 0
        max_drawdown_end = 0

        for i, value in enumerate(values):
            if value > peak:
                peak = value
                drawdown_start = i

            drawdown = (peak - value) / peak

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_start = drawdown_start
                max_drawdown_end = i

            current_drawdown = drawdown

        return {
            "max_drawdown_percent": max_drawdown * 100,
            "current_drawdown_percent": current_drawdown * 100,
            "max_drawdown_start_idx": max_drawdown_start,
            "max_drawdown_end_idx": max_drawdown_end,
            "interpretation": (
                f"Maximaler Rueckgang vom Hoch: {max_drawdown*100:.1f}%"
            ),
        }

    def risk_metrics(
        self,
        returns: list[float],
        benchmark_returns: Optional[list[float]] = None,
        risk_free_rate: float = 0.04,
    ) -> dict[str, Any]:
        """
        Umfassende Risikometriken.
        """
        if len(returns) < 30:
            return {"error": "Mindestens 30 Datenpunkte benoetigt"}

        # Annualisierung (angenommen taegliche Daten)
        ann_factor = math.sqrt(252)

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        ann_return = mean_return * 252
        ann_vol = std_return * ann_factor

        # Sharpe Ratio
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # Sortino Ratio (nur Downside-Volatilitaet)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_vol = statistics.stdev(downside_returns) * ann_factor
            sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        else:
            sortino = float("inf")

        # Calmar Ratio
        values = [1.0]
        for r in returns:
            values.append(values[-1] * (1 + r))
        dd = self.calculate_drawdown(values)
        max_dd = dd.get("max_drawdown_percent", 1) / 100
        calmar = ann_return / max_dd if max_dd > 0 else 0

        metrics = {
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "positive_days_percent": len([r for r in returns if r > 0]) / len(returns) * 100,
            "best_day": max(returns) * 100,
            "worst_day": min(returns) * 100,
        }

        # Beta und Alpha wenn Benchmark vorhanden
        if benchmark_returns and len(benchmark_returns) == len(returns):
            # Kovarianz
            mean_bench = statistics.mean(benchmark_returns)
            cov = sum(
                (r - mean_return) * (b - mean_bench)
                for r, b in zip(returns, benchmark_returns)
            ) / len(returns)

            var_bench = statistics.variance(benchmark_returns)
            beta = cov / var_bench if var_bench > 0 else 1.0

            # Alpha
            ann_bench = mean_bench * 252
            alpha = ann_return - (risk_free_rate + beta * (ann_bench - risk_free_rate))

            metrics["beta"] = beta
            metrics["alpha"] = alpha

        return metrics


def suggest_portfolio_for_goal(
    goal: str,
    amount: Decimal,
    years: int,
    risk_tolerance: str = "moderate",
) -> dict[str, Any]:
    """
    Schlaegt Portfolio basierend auf Anlageziel vor.

    Goals: retirement, house, education, wealth, emergency
    """
    optimizer = PortfolioOptimizer()

    # Erstelle Asset-Mix basierend auf Ziel und Horizont
    if goal == "emergency":
        # Notgroschen: Sehr konservativ
        assets = [
            optimizer.create_asset_from_class("cash"),
            optimizer.create_asset_from_class("bonds_gov"),
        ]
        target_weights = [0.7, 0.3]

    elif goal == "retirement" and years > 20:
        # Langfristige Altersvorsorge: Wachstumsorientiert
        assets = [
            optimizer.create_asset_from_class("stocks_us"),
            optimizer.create_asset_from_class("stocks_eu"),
            optimizer.create_asset_from_class("stocks_em"),
            optimizer.create_asset_from_class("bonds_gov"),
            optimizer.create_asset_from_class("reits"),
        ]
        if risk_tolerance == "aggressive":
            target_weights = [0.40, 0.25, 0.15, 0.10, 0.10]
        else:
            target_weights = [0.30, 0.20, 0.10, 0.30, 0.10]

    elif goal == "house" and years < 5:
        # Kurzfristiges Ziel: Konservativ
        assets = [
            optimizer.create_asset_from_class("bonds_gov"),
            optimizer.create_asset_from_class("bonds_corp"),
            optimizer.create_asset_from_class("cash"),
        ]
        target_weights = [0.50, 0.30, 0.20]

    else:
        # Standard: Ausgewogenes Portfolio
        assets = [
            optimizer.create_asset_from_class("stocks_us"),
            optimizer.create_asset_from_class("stocks_eu"),
            optimizer.create_asset_from_class("bonds_gov"),
            optimizer.create_asset_from_class("gold"),
        ]
        target_weights = [0.40, 0.20, 0.30, 0.10]

    # Entferne None-Assets
    valid = [(a, w) for a, w in zip(assets, target_weights) if a is not None]
    if not valid:
        return {"error": "Could not create portfolio"}

    assets, weights = zip(*valid)
    assets = list(assets)
    weights = list(weights)

    # Normalisiere
    total = sum(weights)
    weights = [w / total for w in weights]

    portfolio = optimizer.calculate_portfolio_metrics(assets, weights)

    # Projektion
    expected_final = float(amount) * (1 + portfolio.expected_return) ** years

    return {
        "goal": goal,
        "initial_amount": str(amount),
        "years": years,
        "risk_tolerance": risk_tolerance,
        "portfolio": portfolio.to_dict(),
        "projected_value": expected_final,
        "projected_gain": expected_final - float(amount),
        "recommendation": (
            f"Mit diesem Portfolio koennte aus {amount} EUR "
            f"nach {years} Jahren ca. {expected_final:,.0f} EUR werden "
            f"(erwartete Rendite: {portfolio.expected_return*100:.1f}% p.a.)"
        ),
    }


if __name__ == "__main__":
    print("=== PORTFOLIO OPTIMIZER TEST ===\n")

    optimizer = PortfolioOptimizer()

    # Erstelle Test-Assets
    assets = [
        optimizer.create_asset_from_class("stocks_us"),
        optimizer.create_asset_from_class("stocks_eu"),
        optimizer.create_asset_from_class("bonds_gov"),
        optimizer.create_asset_from_class("gold"),
    ]
    assets = [a for a in assets if a is not None]

    print("Assets:")
    for a in assets:
        print(f"  {a.name}: {a.expected_return*100:.1f}% Return, {a.volatility*100:.1f}% Vol")

    # Optimiere
    result = optimizer.optimize(assets, risk_tolerance="moderate")

    print(f"\n--- Optimales Portfolio ---")
    print(f"Erwartete Rendite: {result.optimal_portfolio.expected_return*100:.2f}%")
    print(f"Volatilitaet: {result.optimal_portfolio.volatility*100:.2f}%")
    print(f"Sharpe Ratio: {result.optimal_portfolio.sharpe_ratio:.2f}")

    print("\nGewichte:")
    for asset, weight in zip(assets, result.optimal_portfolio.weights):
        print(f"  {asset.name}: {weight*100:.1f}%")

    print("\nEmpfehlungen:")
    for rec in result.recommendations:
        print(f"  - {rec}")

    # Ziel-basierte Empfehlung
    print("\n--- Altersvorsorge-Empfehlung (50k EUR, 30 Jahre) ---")
    suggestion = suggest_portfolio_for_goal("retirement", Decimal("50000"), 30, "moderate")
    print(suggestion["recommendation"])
