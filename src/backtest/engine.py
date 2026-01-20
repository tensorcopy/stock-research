"""
Backtesting Engine
Core backtesting functionality for strategy validation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
import pandas as pd
import numpy as np

from .portfolio import Portfolio, Position
from .metrics import PerformanceMetrics


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    initial_capital: float = 100_000
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_positions: int = 20
    position_size: str = "equal"  # 'equal', 'volatility', 'alpha'
    rebalance_frequency: str = "monthly"  # 'daily', 'weekly', 'monthly'
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class Trade:
    """Record of a single trade"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    slippage: float

    @property
    def total_cost(self) -> float:
        if self.side == "buy":
            return self.quantity * self.price * (1 + self.slippage) + self.commission
        else:
            return self.quantity * self.price * (1 - self.slippage) - self.commission


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    equity_curve: pd.Series
    trades: list[Trade]
    positions_history: list[dict]
    metrics: dict
    daily_returns: pd.Series


class BacktestEngine:
    """
    Event-driven backtesting engine for strategy validation

    Features:
    - Transaction cost modeling (commission + slippage)
    - Position sizing algorithms
    - Flexible rebalancing schedules
    - Comprehensive performance metrics
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.portfolio: Optional[Portfolio] = None
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.positions_history: list[dict] = []

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        signal_generator: Callable[[datetime, dict], dict[str, float]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest with given price data and signal generator

        Args:
            price_data: Dict mapping symbols to OHLCV DataFrames
            signal_generator: Function(date, prices) -> {symbol: signal_strength}
            start_date: Override config start date
            end_date: Override config end date

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        # Initialize
        self.portfolio = Portfolio(self.config.initial_capital)
        self.trades = []
        self.equity_curve = []
        self.positions_history = []

        # Get common date range
        all_dates = self._get_common_dates(price_data)

        start = start_date or self.config.start_date or all_dates[0]
        end = end_date or self.config.end_date or all_dates[-1]

        dates = [d for d in all_dates if start <= d <= end]

        # Main backtest loop
        rebalance_dates = self._get_rebalance_dates(dates)

        for date in dates:
            # Get current prices
            current_prices = self._get_prices_at_date(price_data, date)

            # Update portfolio value
            portfolio_value = self.portfolio.calculate_value(current_prices)
            self.equity_curve.append((date, portfolio_value))

            # Rebalance if needed
            if date in rebalance_dates:
                # Generate signals
                signals = signal_generator(date, price_data)

                # Execute rebalancing
                self._rebalance(date, signals, current_prices)

            # Record positions
            self.positions_history.append({
                "date": date,
                "positions": self.portfolio.get_positions_summary(),
                "cash": self.portfolio.cash,
                "total_value": portfolio_value
            })

        # Calculate metrics
        equity_series = pd.Series(
            [e[1] for e in self.equity_curve],
            index=[e[0] for e in self.equity_curve]
        )
        daily_returns = equity_series.pct_change().dropna()

        metrics_calculator = PerformanceMetrics()
        metrics = metrics_calculator.calculate_all(equity_series, daily_returns)

        return BacktestResult(
            equity_curve=equity_series,
            trades=self.trades,
            positions_history=self.positions_history,
            metrics=metrics,
            daily_returns=daily_returns
        )

    def _get_common_dates(self, price_data: dict[str, pd.DataFrame]) -> list[datetime]:
        """Get dates common to all price series"""
        date_sets = []
        for symbol, df in price_data.items():
            date_sets.append(set(df.index))

        if not date_sets:
            return []

        common = date_sets[0]
        for ds in date_sets[1:]:
            common = common.intersection(ds)

        return sorted(list(common))

    def _get_prices_at_date(
        self,
        price_data: dict[str, pd.DataFrame],
        date: datetime
    ) -> dict[str, float]:
        """Get closing prices at specific date"""
        prices = {}
        for symbol, df in price_data.items():
            if date in df.index:
                prices[symbol] = df.loc[date, "close"]
        return prices

    def _get_rebalance_dates(self, dates: list[datetime]) -> set[datetime]:
        """Determine which dates are rebalancing dates"""
        if not dates:
            return set()

        rebalance_dates = set()

        if self.config.rebalance_frequency == "daily":
            return set(dates)

        elif self.config.rebalance_frequency == "weekly":
            # First trading day of each week
            current_week = None
            for date in dates:
                week = date.isocalendar()[1]
                if week != current_week:
                    rebalance_dates.add(date)
                    current_week = week

        elif self.config.rebalance_frequency == "monthly":
            # First trading day of each month
            current_month = None
            for date in dates:
                month = (date.year, date.month)
                if month != current_month:
                    rebalance_dates.add(date)
                    current_month = month

        return rebalance_dates

    def _rebalance(
        self,
        date: datetime,
        signals: dict[str, float],
        prices: dict[str, float]
    ):
        """Execute portfolio rebalancing based on signals"""
        # Filter to stocks we have prices for
        valid_signals = {
            s: v for s, v in signals.items()
            if s in prices and not np.isnan(v)
        }

        if not valid_signals:
            return

        # Rank and select top stocks
        sorted_signals = sorted(
            valid_signals.items(),
            key=lambda x: x[1],
            reverse=True
        )
        target_symbols = [s for s, _ in sorted_signals[:self.config.max_positions]]

        # Calculate target weights
        target_weights = self._calculate_weights(target_symbols, valid_signals, prices)

        # Get current portfolio value
        portfolio_value = self.portfolio.calculate_value(prices)

        # Determine trades needed
        current_positions = {
            p.symbol: p.quantity * prices.get(p.symbol, 0) / portfolio_value
            for p in self.portfolio.positions.values()
            if p.symbol in prices
        }

        # Sell positions not in target
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in target_symbols and symbol in prices:
                self._execute_sell(date, symbol, prices[symbol], self.portfolio.positions[symbol].quantity)

        # Adjust existing positions and buy new ones
        for symbol in target_symbols:
            if symbol not in prices:
                continue

            current_weight = current_positions.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)

            weight_diff = target_weight - current_weight
            target_value = weight_diff * portfolio_value
            target_quantity = target_value / prices[symbol]

            if abs(weight_diff) > 0.01:  # 1% threshold to avoid tiny trades
                if target_quantity > 0:
                    self._execute_buy(date, symbol, prices[symbol], target_quantity)
                elif target_quantity < 0 and symbol in self.portfolio.positions:
                    sell_qty = min(-target_quantity, self.portfolio.positions[symbol].quantity)
                    self._execute_sell(date, symbol, prices[symbol], sell_qty)

    def _calculate_weights(
        self,
        symbols: list[str],
        signals: dict[str, float],
        prices: dict[str, float]
    ) -> dict[str, float]:
        """Calculate position weights based on sizing method"""
        n = len(symbols)
        if n == 0:
            return {}

        if self.config.position_size == "equal":
            weight = 1.0 / n
            return {s: weight for s in symbols}

        elif self.config.position_size == "alpha":
            # Weight by signal strength (positive signals only)
            positive_signals = {s: max(0, signals[s]) for s in symbols}
            total_signal = sum(positive_signals.values())
            if total_signal == 0:
                return {s: 1.0 / n for s in symbols}
            return {s: v / total_signal for s, v in positive_signals.items()}

        else:
            # Default to equal weight
            return {s: 1.0 / n for s in symbols}

    def _execute_buy(
        self,
        date: datetime,
        symbol: str,
        price: float,
        quantity: float
    ):
        """Execute a buy order"""
        slippage_cost = price * self.config.slippage
        execution_price = price + slippage_cost
        commission = execution_price * quantity * self.config.commission

        total_cost = execution_price * quantity + commission

        if total_cost > self.portfolio.cash:
            # Reduce quantity to fit available cash
            quantity = (self.portfolio.cash - commission) / execution_price
            if quantity <= 0:
                return
            total_cost = execution_price * quantity + commission

        # Record trade
        trade = Trade(
            timestamp=date,
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage_cost * quantity
        )
        self.trades.append(trade)

        # Update portfolio
        self.portfolio.cash -= total_cost
        self.portfolio.add_position(symbol, quantity, execution_price)

    def _execute_sell(
        self,
        date: datetime,
        symbol: str,
        price: float,
        quantity: float
    ):
        """Execute a sell order"""
        slippage_cost = price * self.config.slippage
        execution_price = price - slippage_cost
        commission = execution_price * quantity * self.config.commission

        proceeds = execution_price * quantity - commission

        # Record trade
        trade = Trade(
            timestamp=date,
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage_cost * quantity
        )
        self.trades.append(trade)

        # Update portfolio
        self.portfolio.cash += proceeds
        self.portfolio.remove_position(symbol, quantity)
