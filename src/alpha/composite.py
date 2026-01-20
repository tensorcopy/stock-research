"""
Composite Alpha Module
Combine multiple alpha factors into a unified score
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from .factors import AlphaFactors, AlphaFactor


@dataclass
class CompositeAlphaScore:
    """Container for composite alpha score"""
    score: float
    z_score: float
    percentile: float
    factor_contributions: dict[str, float]
    factors: dict[str, AlphaFactor]


class CompositeAlpha:
    """
    Combine multiple alpha factors for stock selection

    Key considerations:
    1. Factor diversification - don't over-concentrate
    2. Factor timing - some factors work better in certain regimes
    3. Factor decay - signals weaken over time
    4. Transaction costs - higher turnover factors need stronger signals
    """

    # Default factor weights based on research efficacy
    DEFAULT_WEIGHTS = {
        # Momentum (strong, persistent)
        "momentum_12_1": 0.15,
        "momentum_6_1": 0.10,
        # Value (classic, but cyclical)
        "earnings_yield": 0.10,
        "fcf_yield": 0.10,
        "book_to_market": 0.05,
        # Quality (defensive, stable)
        "roe": 0.10,
        "gross_profitability": 0.10,
        "accruals": 0.05,
        # Low volatility (risk premium)
        "ivol": 0.10,
        "beta": 0.05,
        # Size (small cap premium)
        "size": 0.05,
        # Growth (can be good or bad)
        "revenue_growth": 0.05,
    }

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """
        Initialize with custom or default factor weights

        Args:
            weights: Custom factor weights (should sum to 1)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.alpha_factors = AlphaFactors()

    def normalize_factor(
        self,
        values: pd.Series,
        method: str = "z_score"
    ) -> pd.Series:
        """
        Normalize factor values for comparison

        Args:
            values: Raw factor values
            method: 'z_score', 'rank', or 'minmax'
        """
        if method == "z_score":
            mean = values.mean()
            std = values.std()
            if std == 0:
                return pd.Series(0, index=values.index)
            return (values - mean) / std

        elif method == "rank":
            return values.rank(pct=True)

        elif method == "minmax":
            min_val = values.min()
            max_val = values.max()
            if max_val == min_val:
                return pd.Series(0.5, index=values.index)
            return (values - min_val) / (max_val - min_val)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def winsorize(
        self,
        values: pd.Series,
        lower: float = 0.01,
        upper: float = 0.99
    ) -> pd.Series:
        """
        Winsorize extreme values to reduce outlier impact

        Args:
            values: Raw values
            lower: Lower percentile threshold
            upper: Upper percentile threshold
        """
        lower_bound = values.quantile(lower)
        upper_bound = values.quantile(upper)
        return values.clip(lower_bound, upper_bound)

    def calculate_composite_score(
        self,
        factor_values: dict[str, float],
        factor_universe: Optional[dict[str, pd.Series]] = None
    ) -> CompositeAlphaScore:
        """
        Calculate composite alpha score for a single stock

        Args:
            factor_values: Dict of factor name to raw value
            factor_universe: Optional universe data for z-score calculation
        """
        contributions = {}
        weighted_sum = 0
        total_weight = 0

        for factor_name, weight in self.weights.items():
            if factor_name not in factor_values:
                continue

            value = factor_values[factor_name]
            if pd.isna(value):
                continue

            # Normalize if universe provided
            if factor_universe and factor_name in factor_universe:
                universe_values = factor_universe[factor_name]
                z_score = (value - universe_values.mean()) / universe_values.std()
                value = z_score

            contribution = value * weight
            contributions[factor_name] = contribution
            weighted_sum += contribution
            total_weight += weight

        # Rescale if not all factors available
        if total_weight > 0 and total_weight < 1:
            weighted_sum = weighted_sum / total_weight

        # Create AlphaFactor objects
        factors = {
            name: AlphaFactor(name, val)
            for name, val in factor_values.items()
        }

        return CompositeAlphaScore(
            score=weighted_sum,
            z_score=weighted_sum,  # Already normalized
            percentile=0,  # Filled when ranking universe
            factor_contributions=contributions,
            factors=factors
        )

    def rank_universe(
        self,
        universe_factors: dict[str, dict[str, float]]
    ) -> pd.DataFrame:
        """
        Rank all stocks in universe by composite alpha

        Args:
            universe_factors: Dict mapping symbol to factor values dict

        Returns:
            DataFrame with scores and ranks
        """
        # First pass: collect all factor values for normalization
        factor_series = {}
        for factor_name in self.weights:
            values = []
            symbols = []
            for symbol, factors in universe_factors.items():
                if factor_name in factors and not pd.isna(factors[factor_name]):
                    values.append(factors[factor_name])
                    symbols.append(symbol)
            if values:
                factor_series[factor_name] = pd.Series(values, index=symbols)

        # Normalize factors
        normalized_factors = {}
        for factor_name, series in factor_series.items():
            # Winsorize then z-score
            winsorized = self.winsorize(series)
            normalized = self.normalize_factor(winsorized, "z_score")
            normalized_factors[factor_name] = normalized

        # Calculate composite scores
        results = []
        for symbol, factors in universe_factors.items():
            normalized_values = {}
            for factor_name in self.weights:
                if factor_name in normalized_factors and symbol in normalized_factors[factor_name].index:
                    normalized_values[factor_name] = normalized_factors[factor_name][symbol]

            if normalized_values:
                score = self.calculate_composite_score(normalized_values)
                results.append({
                    "symbol": symbol,
                    "composite_score": score.score,
                    **{f"factor_{k}": v for k, v in score.factor_contributions.items()}
                })

        df = pd.DataFrame(results)

        if not df.empty:
            df["rank"] = df["composite_score"].rank(ascending=False)
            df["percentile"] = df["composite_score"].rank(pct=True) * 100

        return df.sort_values("composite_score", ascending=False)

    def get_top_stocks(
        self,
        universe_factors: dict[str, dict[str, float]],
        n: int = 20,
        min_score: Optional[float] = None
    ) -> list[str]:
        """
        Get top N stocks by composite alpha

        Args:
            universe_factors: Factor values for all stocks
            n: Number of stocks to return
            min_score: Optional minimum score threshold

        Returns:
            List of top stock symbols
        """
        rankings = self.rank_universe(universe_factors)

        if rankings.empty:
            return []

        if min_score is not None:
            rankings = rankings[rankings["composite_score"] >= min_score]

        return rankings.head(n)["symbol"].tolist()

    def get_factor_exposures(
        self,
        portfolio_symbols: list[str],
        universe_factors: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """
        Calculate portfolio's exposure to each factor

        Useful for understanding what's driving portfolio returns
        """
        exposures = {factor: [] for factor in self.weights}

        for symbol in portfolio_symbols:
            if symbol not in universe_factors:
                continue

            factors = universe_factors[symbol]
            for factor_name in self.weights:
                if factor_name in factors and not pd.isna(factors[factor_name]):
                    exposures[factor_name].append(factors[factor_name])

        # Average exposure per factor
        avg_exposures = {}
        for factor_name, values in exposures.items():
            if values:
                avg_exposures[factor_name] = np.mean(values)

        return avg_exposures

    def analyze_factor_performance(
        self,
        factor_returns: pd.DataFrame
    ) -> dict:
        """
        Analyze historical factor performance

        Args:
            factor_returns: DataFrame with factor return series

        Returns:
            Performance statistics for each factor
        """
        stats = {}

        for factor in factor_returns.columns:
            returns = factor_returns[factor].dropna()
            if len(returns) < 10:
                continue

            stats[factor] = {
                "mean_return": returns.mean() * 252,  # Annualized
                "volatility": returns.std() * np.sqrt(252),
                "sharpe": (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                "max_drawdown": (returns.cumsum() - returns.cumsum().cummax()).min(),
                "hit_rate": (returns > 0).mean(),
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis()
            }

        return stats
