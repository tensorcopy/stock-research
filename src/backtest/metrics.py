"""
Performance Metrics Module
Calculate comprehensive performance statistics
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class RiskMetrics:
    """Risk-related metrics"""
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional VaR
    beta: Optional[float] = None


@dataclass
class ReturnMetrics:
    """Return-related metrics"""
    total_return: float
    annualized_return: float
    best_day: float
    worst_day: float
    best_month: float
    worst_month: float
    positive_days_pct: float


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for backtests

    Key metrics for alpha evaluation:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Alpha and beta vs benchmark
    - Information ratio
    - Win rate and profit factor
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    def calculate_all(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> dict:
        """
        Calculate all performance metrics

        Args:
            equity_curve: Daily portfolio value series
            daily_returns: Daily return series
            benchmark_returns: Optional benchmark return series

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # Basic return metrics
        metrics.update(self.calculate_return_metrics(equity_curve, daily_returns))

        # Risk metrics
        metrics.update(self.calculate_risk_metrics(daily_returns))

        # Risk-adjusted metrics
        metrics.update(self.calculate_risk_adjusted_metrics(daily_returns))

        # Drawdown metrics
        metrics.update(self.calculate_drawdown_metrics(equity_curve))

        # Alpha/Beta vs benchmark
        if benchmark_returns is not None:
            metrics.update(self.calculate_alpha_beta(daily_returns, benchmark_returns))

        return metrics

    def calculate_return_metrics(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series
    ) -> dict:
        """Calculate return-related metrics"""
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized return
        n_years = len(daily_returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Monthly returns
        monthly_returns = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "best_day": daily_returns.max(),
            "worst_day": daily_returns.min(),
            "best_month": monthly_returns.max() if len(monthly_returns) > 0 else 0,
            "worst_month": monthly_returns.min() if len(monthly_returns) > 0 else 0,
            "positive_days_pct": (daily_returns > 0).mean(),
            "positive_months_pct": (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0,
            "avg_daily_return": daily_returns.mean(),
            "avg_monthly_return": monthly_returns.mean() if len(monthly_returns) > 0 else 0
        }

    def calculate_risk_metrics(self, daily_returns: pd.Series) -> dict:
        """Calculate risk-related metrics"""
        volatility = daily_returns.std() * np.sqrt(252)

        # Value at Risk (parametric)
        var_95 = daily_returns.quantile(0.05)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()

        # Skewness and kurtosis
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()

        # Downside deviation (for Sortino)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0

        return {
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "downside_deviation": downside_deviation
        }

    def calculate_risk_adjusted_metrics(self, daily_returns: pd.Series) -> dict:
        """Calculate risk-adjusted return metrics"""
        # Sharpe Ratio
        excess_returns = daily_returns - self.daily_rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino Ratio
        negative_returns = daily_returns[daily_returns < self.daily_rf]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 1
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino
        }

    def calculate_drawdown_metrics(self, equity_curve: pd.Series) -> dict:
        """Calculate drawdown-related metrics"""
        # Running maximum
        running_max = equity_curve.cummax()

        # Drawdown series
        drawdown = (equity_curve - running_max) / running_max

        # Max drawdown
        max_drawdown = drawdown.min()

        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            drawdown_periods.append(current_duration)

        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0

        # Calmar ratio (annualized return / max drawdown)
        n_years = len(equity_curve) / 252
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        calmar = -annualized_return / max_drawdown if max_drawdown < 0 else 0

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_days": max_drawdown_duration,
            "avg_drawdown_duration_days": avg_drawdown_duration,
            "calmar_ratio": calmar,
            "num_drawdowns": len(drawdown_periods)
        }

    def calculate_alpha_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> dict:
        """Calculate alpha and beta vs benchmark"""
        # Align series
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 20:
            return {"alpha": None, "beta": None, "information_ratio": None}

        port_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]

        # Beta
        covariance = port_ret.cov(bench_ret)
        benchmark_variance = bench_ret.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1

        # Alpha (Jensen's alpha, annualized)
        excess_port = port_ret.mean() - self.daily_rf
        excess_bench = bench_ret.mean() - self.daily_rf
        daily_alpha = excess_port - beta * excess_bench
        alpha = daily_alpha * 252

        # Information Ratio
        active_returns = port_ret - bench_ret
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

        # Correlation
        correlation = port_ret.corr(bench_ret)

        return {
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "correlation_to_benchmark": correlation,
            "tracking_error": tracking_error
        }

    def calculate_trade_metrics(self, trades: list) -> dict:
        """Calculate trade-level metrics"""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0
            }

        # Group by symbol to calculate P&L per position
        # This is a simplified version - full implementation would track entry/exit pairs

        total_trades = len(trades)
        buys = [t for t in trades if t.side == "buy"]
        sells = [t for t in trades if t.side == "sell"]

        total_commission = sum(t.commission for t in trades)
        total_slippage = sum(t.slippage for t in trades)

        return {
            "total_trades": total_trades,
            "buy_trades": len(buys),
            "sell_trades": len(sells),
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "avg_trade_size": np.mean([t.quantity * t.price for t in trades]) if trades else 0
        }

    def generate_report(
        self,
        metrics: dict,
        equity_curve: pd.Series
    ) -> str:
        """Generate a text report of performance"""
        report = []
        report.append("=" * 50)
        report.append("BACKTEST PERFORMANCE REPORT")
        report.append("=" * 50)

        report.append(f"\nPeriod: {equity_curve.index[0].date()} to {equity_curve.index[-1].date()}")
        report.append(f"Trading Days: {len(equity_curve)}")

        report.append("\n--- RETURNS ---")
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"Best Day: {metrics.get('best_day', 0):.2%}")
        report.append(f"Worst Day: {metrics.get('worst_day', 0):.2%}")

        report.append("\n--- RISK ---")
        report.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"VaR (95%): {metrics.get('var_95', 0):.2%}")

        report.append("\n--- RISK-ADJUSTED ---")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")

        if metrics.get('alpha') is not None:
            report.append("\n--- VS BENCHMARK ---")
            report.append(f"Alpha: {metrics.get('alpha', 0):.2%}")
            report.append(f"Beta: {metrics.get('beta', 0):.2f}")
            report.append(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")

        report.append("\n" + "=" * 50)

        return "\n".join(report)
