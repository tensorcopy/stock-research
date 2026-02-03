#!/usr/bin/env python3
"""
Simple Momentum Strategy Example
Demonstrates how to use the stock research framework for a basic momentum strategy
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import MarketDataFetcher, get_sp500_symbols
from src.signals.momentum import MomentumSignals
from src.alpha.composite import CompositeAlpha
from src.backtest.engine import BacktestEngine, BacktestConfig


def run_momentum_strategy():
    """
    Run a simple momentum strategy

    Strategy:
    1. Universe: Top 100 S&P 500 stocks by market cap
    2. Signal: 12-1 month momentum (classic Jegadeesh-Titman)
    3. Selection: Top 20 stocks by momentum
    4. Rebalancing: Monthly
    """
    print("=" * 60)
    print("Simple Momentum Strategy")
    print("=" * 60)

    # Initialize components
    data_fetcher = MarketDataFetcher()
    momentum_signals = MomentumSignals()

    # Get universe
    print("\n1. Fetching universe...")
    symbols = get_sp500_symbols()[:50]  # Use subset for demo
    print(f"   Universe size: {len(symbols)} stocks")

    # Fetch historical data
    print("\n2. Fetching historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years

    price_data = data_fetcher.get_ohlcv(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    print(f"   Fetched data for {len(price_data)} symbols")

    # Calculate momentum scores
    print("\n3. Calculating momentum scores...")

    def momentum_signal_generator(date, data):
        """Generate momentum signals for given date"""
        signals = {}

        for symbol, df in data.items():
            # Get data up to current date
            mask = df.index <= date
            if mask.sum() < 252:  # Need at least 1 year of data
                continue

            historical = df[mask]
            close = historical["close"]

            # Calculate 12-1 momentum
            if len(close) >= 252:
                mom_12_1 = (close.iloc[-21] / close.iloc[-252]) - 1
                signals[symbol] = mom_12_1

        return signals

    # Configure and run backtest
    print("\n4. Running backtest...")
    config = BacktestConfig(
        initial_capital=100_000,
        commission=0.001,
        slippage=0.0005,
        max_positions=10,
        position_size="equal",
        rebalance_frequency="monthly"
    )

    engine = BacktestEngine(config)

    # Run backtest
    result = engine.run(
        price_data=price_data,
        signal_generator=momentum_signal_generator
    )

    # Display results
    print("\n5. Results:")
    print("-" * 40)

    metrics = result.metrics
    print(f"   Total Return: {metrics['total_return']:.2%}")
    print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"   Volatility: {metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"   Total Trades: {len(result.trades)}")

    print("\n" + "=" * 60)
    print("Strategy completed!")
    print("=" * 60)

    return result


def analyze_factor_returns():
    """
    Analyze returns of different momentum factors

    Compares:
    - 1-month momentum (short-term reversal)
    - 6-month momentum
    - 12-month momentum
    - 12-1 month momentum (classic)
    """
    print("\nFactor Analysis")
    print("-" * 40)

    # This would typically load actual data
    # For demo, we'll show the analysis structure

    factors = {
        "mom_1m": "Short-term (1 month) - typically negative",
        "mom_6m": "Medium-term (6 month) - strong momentum",
        "mom_12_1": "Classic (12-1 month) - strongest signal",
        "mom_24m": "Long-term (24 month) - reversal starts"
    }

    print("\nMomentum Factor Horizon Analysis:")
    for factor, description in factors.items():
        print(f"  {factor}: {description}")

    print("\nKey Insights:")
    print("  - Short-term (1-4 weeks): Mean reversion dominates")
    print("  - Medium-term (3-12 months): Momentum strongest")
    print("  - Long-term (3-5 years): Mean reversion returns")
    print("  - Skip last month to avoid reversal effect")


if __name__ == "__main__":
    print("\nStock Research Framework Demo")
    print("=" * 60)

    # Note: This requires yfinance and internet connection
    # For actual runs, uncomment the following:

    # result = run_momentum_strategy()

    # For demo without data:
    print("\nTo run the full strategy with live data:")
    print("1. Install requirements: uv sync")
    print("2. Uncomment the run_momentum_strategy() call")
    print("3. Run: uv run python examples/simple_momentum_strategy.py")

    analyze_factor_returns()
