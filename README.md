# Stock Research Framework

A comprehensive Python framework for market research, stock selection, and alpha generation focused on capturing trends to maximize returns.

## Overview

This framework provides tools for:
- **Data Collection**: Market data, fundamentals, and sentiment analysis
- **Signal Generation**: Trend detection, momentum signals, and breakout identification
- **Alpha Factors**: A library of proven quantitative factors for stock selection
- **Backtesting**: Event-driven backtesting with transaction cost modeling

## Key Features

### Trend Capture Strategy

The framework implements multiple methods to identify and capture emerging trends:

1. **Moving Average Crossovers** - Golden/Death Cross detection
2. **ADX Trend Strength** - Measure trend intensity
3. **Linear Regression Channels** - Statistical trend detection
4. **Higher Highs/Lows Pattern** - Classic technical analysis
5. **Breakout Detection** - Price, volume, and volatility breakouts

### Alpha Factor Library

Research-backed factors for stock selection:

| Category | Factors | Description |
|----------|---------|-------------|
| **Momentum** | 12-1 month, 6-1 month | Price momentum (Jegadeesh-Titman) |
| **Value** | E/P, B/M, FCF Yield | Cheap stocks outperform |
| **Quality** | ROE, Gross Profitability, Accruals | Profitable, stable companies |
| **Low Vol** | IVOL, Beta | Low volatility anomaly |
| **Size** | Market Cap | Small cap premium |

## Installation

```bash
# Clone the repository
cd projects/stock-research

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Automated Daily Research (GitHub Actions)

This repository includes a GitHub Actions workflow that automatically runs market research every weekday after market close and sends email notifications with the results.

### Quick Setup

1. **Configure GitHub Secrets** (required for email notifications):
   - Go to your repository **Settings** → **Secrets and variables** → **Actions**
   - Add the following secrets:
     - `EMAIL_SMTP_HOST` (e.g., `smtp.gmail.com`)
     - `EMAIL_SMTP_PORT` (e.g., `587`)
     - `EMAIL_FROM` (your email address)
     - `EMAIL_TO` (recipient email address)
     - `EMAIL_PASSWORD` (app password or account password)

2. **Enable GitHub Actions**:
   - The workflow runs automatically Mon-Fri at 5:00 PM ET (22:00 UTC)
   - Or trigger manually via **Actions** tab → **Daily Market Research** → **Run workflow**

3. **Receive Daily Reports**:
   - HTML email with market overview, top momentum stocks, breakouts, and strong buy candidates
   - Attached JSON and CSV files with complete analysis

For detailed setup instructions, email provider configuration, and troubleshooting, see **[GITHUB_ACTIONS_SETUP.md](GITHUB_ACTIONS_SETUP.md)**.

## Quick Start

### 1. Fetch Market Data

```python
from src.data.market_data import MarketDataFetcher, get_sp500_symbols
from datetime import datetime, timedelta

fetcher = MarketDataFetcher()

# Get S&P 500 symbols
symbols = get_sp500_symbols()

# Fetch OHLCV data
data = fetcher.get_ohlcv(
    symbols=symbols[:50],  # Top 50 stocks
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now()
)
```

### 2. Generate Signals

```python
from src.signals.trend_detector import TrendDetector
from src.signals.momentum import MomentumSignals

# Trend analysis
trend_detector = TrendDetector()
trend = trend_detector.get_composite_trend(data["AAPL"])
print(f"AAPL Trend: {trend.direction.name}, Strength: {trend.strength:.2f}")

# Momentum scoring
momentum = MomentumSignals()
signal = momentum.calculate_momentum_score(data["AAPL"])
print(f"Momentum Score: {signal.score:.2f}")
```

### 3. Calculate Alpha Factors

```python
from src.alpha.composite import CompositeAlpha

alpha = CompositeAlpha()

# Calculate factors for universe
universe_factors = {}
for symbol, df in data.items():
    # Add fundamental data here
    universe_factors[symbol] = {
        "momentum_12_1": calculate_momentum(df),
        # ... other factors
    }

# Rank stocks
rankings = alpha.rank_universe(universe_factors)
top_stocks = rankings.head(20)["symbol"].tolist()
```

### 4. Backtest Strategy

```python
from src.backtest.engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    initial_capital=100_000,
    max_positions=20,
    rebalance_frequency="monthly"
)

engine = BacktestEngine(config)
result = engine.run(
    price_data=data,
    signal_generator=my_signal_function
)

print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Total Return: {result.metrics['total_return']:.2%}")
```

## Project Structure

```
stock-research/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml         # Default configuration
├── src/
│   ├── data/
│   │   ├── market_data.py   # Price/volume data fetching
│   │   ├── fundamentals.py  # Financial statements
│   │   └── sentiment.py     # News/sentiment analysis
│   ├── signals/
│   │   ├── trend_detector.py   # Trend identification
│   │   ├── momentum.py         # Momentum indicators
│   │   └── breakout.py         # Breakout detection
│   ├── alpha/
│   │   ├── factors.py       # Individual alpha factors
│   │   └── composite.py     # Factor combination
│   ├── backtest/
│   │   ├── engine.py        # Backtesting engine
│   │   ├── portfolio.py     # Portfolio management
│   │   └── metrics.py       # Performance metrics
│   └── utils/
│       └── helpers.py       # Utility functions
├── tests/
│   ├── test_momentum.py
│   └── test_alpha_factors.py
├── examples/
│   └── simple_momentum_strategy.py
├── notebooks/               # Jupyter notebooks
└── results/                 # Backtest results
```

## Alpha Generation Philosophy

### What Makes a Good Alpha Factor?

1. **Persistent** - Works across different time periods
2. **Pervasive** - Works across markets and asset classes
3. **Robust** - Survives different definitions
4. **Investable** - Can be implemented at scale
5. **Intuitive** - Has economic rationale

### Momentum: The King of Factors

The 12-1 month momentum factor (Jegadeesh-Titman, 1993) remains one of the strongest:

- **Why it works**: Behavioral biases (underreaction, herding)
- **Best implementation**: Skip last month (short-term reversal)
- **Combine with**: Quality factors to avoid "momentum crashes"
- **Watch out for**: Factor crowding, transaction costs

### Trend Following vs. Mean Reversion

| Timeframe | Dominant Effect |
|-----------|-----------------|
| 1-4 weeks | Mean reversion |
| 1-12 months | **Momentum** |
| 3-5 years | Mean reversion |

## Configuration

Edit `config/default.yaml` to customize:

```yaml
# Universe selection
universe:
  base: sp500
  min_market_cap: 1000000000

# Factor weights
alpha:
  weights:
    momentum_12_1: 0.20
    earnings_yield: 0.15
    roe: 0.10

# Backtest settings
backtest:
  initial_capital: 100000
  max_positions: 20
  rebalance_frequency: monthly
```

## Performance Metrics

The framework calculates comprehensive metrics:

- **Returns**: Total, annualized, monthly
- **Risk**: Volatility, VaR, max drawdown
- **Risk-Adjusted**: Sharpe, Sortino, Calmar ratios
- **Alpha/Beta**: Jensen's alpha, information ratio

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## References

Key academic papers on factors and momentum:

- Jegadeesh & Titman (1993) - Returns to Buying Winners
- Fama & French (1993) - Three-Factor Model
- Novy-Marx (2013) - Gross Profitability
- Asness et al. (2013) - Value and Momentum Everywhere

## License

MIT License - See LICENSE file for details.

## Disclaimer

This framework is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consult with financial professionals before making investment decisions.
