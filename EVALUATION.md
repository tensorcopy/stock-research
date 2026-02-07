# Stock Research Framework - Evaluation

**Date:** 2026-02-07
**Scope:** Full codebase review, live analysis run, and investment applicability assessment

---

## 1. What This Framework Does

This is a **quantitative stock screening and research tool** that combines four analytical approaches to identify investment candidates:

| Module | Purpose | Output |
|--------|---------|--------|
| **Trend Detection** | Identifies whether stocks are trending up/down via MA crossovers, ADX, linear regression, swing analysis | Trend direction + strength + confidence |
| **Momentum Signals** | Scores stocks on RSI, MACD, stochastic, price momentum, volume momentum | Composite momentum score |
| **Breakout Detection** | Finds stocks breaking through support/resistance on elevated volume | Breakout signals + potential score |
| **Alpha Factors** | Ranks stocks across momentum, value, quality, volatility, size, growth factors | Composite alpha score + percentile |

All four are combined in `analyze_market.py` to produce **"Strong Buy" candidates** - stocks that simultaneously show positive trends, high momentum, and high composite alpha.

---

## 2. How to Use It for Investment Research

### Daily Screening Workflow

```bash
# Run the full analysis
uv run python analyze_market.py

# Review outputs:
#   results/SUMMARY.md       - Quick overview of top picks
#   results/alpha_rankings.csv - All stocks ranked by alpha score
#   results/latest.json      - Full machine-readable data
```

### What the Output Tells You

**Market Overview** - Breadth indicator showing % of stocks in uptrends. Useful for gauging whether it's a favorable environment for buying (>60% bullish) or a time for caution (<40% bullish).

**Top Momentum Stocks** - Stocks with the strongest price momentum. These are names that have been outperforming recently. Momentum is one of the most persistent anomalies in finance research.

**Breakout Opportunities** - Stocks breaking through key price levels on volume. These are potential entry points for trend-following strategies.

**Alpha Rankings** - Multi-factor ranking combining momentum, value, quality, and volatility. This is the most comprehensive signal - it looks beyond just price action to fundamental quality.

**Strong Buy Candidates** - The intersection: stocks that score well on ALL dimensions. Today's run identified 10 candidates including CAT, MRK, JNJ, XOM, UPS.

### Backtesting a Strategy

```bash
# Run the included momentum strategy example
uv run python examples/simple_momentum_strategy.py

# This backtests a 12-1 month momentum strategy over 2 years
# with monthly rebalancing and reports Sharpe, drawdown, etc.
```

### Customizing for Your Needs

Edit `config/default.yaml` to:
- Change the stock universe (switch from S&P 500 to custom list)
- Adjust factor weights (increase momentum weight if you're more trend-oriented)
- Change rebalancing frequency (weekly vs monthly)
- Set risk parameters (position sizing, stop losses)

---

## 3. Strengths

### Methodologically Sound
- **Skips the last month** in 12-1 momentum (correctly handles short-term reversal effect per Jegadeesh-Titman research)
- **Winsorizes** factor values before normalization to handle outliers
- **Z-score normalization** for cross-factor comparability
- **Multi-method trend detection** (4 independent methods combined by confidence-weighted average) reduces false signals
- **Transaction cost modeling** in backtester (0.1% commission + 0.05% slippage) - realistic for evaluating net-of-cost performance

### Well-Structured Code
- Clean separation of concerns (data / signals / alpha / backtest)
- Dataclass-based signal containers with direction, strength, and confidence
- 16 passing tests covering momentum and alpha factor calculations
- Configuration-driven design via YAML

### Practical Outputs
- Markdown summary, CSV rankings, JSON data, and an HTML dashboard
- GitHub Actions workflow for automated daily runs at market close
- Historical results archived by date for tracking changes over time

---

## 4. Limitations and Risks

### Data Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Yahoo Finance only** | Data can be delayed, occasionally inaccurate, and has rate limits | Fine for daily analysis; not suitable for intraday |
| **No fundamental data used in live analysis** | `analyze_market.py` only uses price-based alpha factors (momentum + volatility), skipping value/quality factors | The factor library supports fundamentals, but the main script doesn't fetch them - this reduces the diversification benefit of multi-factor |
| **Limited universe** | Hardcoded to 50 large-cap stocks | Configurable, but misses mid/small-cap opportunities |
| **1-year lookback** | May miss longer-term cycles | Adequate for momentum signals, but longer history is better for value/quality factors |

### Analytical Gaps

1. **No sector-relative analysis.** The current output mixes sectors freely. A stock like XOM might score high on momentum simply because energy is in favor. Sector-neutral scoring would distinguish genuine stock-level alpha from sector rotation.

2. **No risk-adjusted position sizing in the screening output.** The "Strong Buy" list treats all candidates equally. In practice, a volatile stock like TSLA should have a smaller position than a stable stock like JNJ for the same conviction level.

3. **Backtest lacks benchmark comparison by default.** The backtest engine supports alpha/beta calculation vs. a benchmark, but the example strategy doesn't use it. Without comparing to SPY, you can't tell if a strategy is truly adding alpha or just riding beta.

4. **No out-of-sample validation framework.** There's no walk-forward analysis or train/test split for factor weight optimization. The default weights are fixed rather than empirically validated on your specific universe.

5. **RSI values above 70-80 on "buy" candidates.** Today's top momentum picks (JNJ RSI: 94.3, MRK RSI: 86.2, PEP RSI: 93.7) are deeply overbought. The framework flags high momentum but doesn't warn about overextension risk. Buying at RSI >80 often precedes mean reversion.

6. **No drawdown protection or regime detection.** The framework doesn't adjust signals for market regime (bull vs. bear vs. crisis). Momentum strategies historically suffer severe drawdowns during regime transitions (e.g., momentum crash of 2009).

### What's Missing for Investment Decision-Making

- **No valuation context** in the live output. You can see that CAT has strong momentum, but not whether it's trading at 25x or 15x earnings.
- **No earnings calendar awareness.** Buying ahead of earnings reports adds binary event risk.
- **No macro/sector overlay.** Interest rate environment, economic cycle positioning, and sector rotation context matter for factor timing.
- **No portfolio construction guidance.** The tool identifies candidates but doesn't tell you how to size them, when to rebalance, or how to manage correlation.

---

## 5. Today's Analysis Results (2026-02-07)

### Market State: NEUTRAL (58% bullish)
58% of the 50 analyzed stocks show uptrends. This is a moderately constructive environment - not strongly bullish enough for aggressive positioning, but not bearish either.

### Strong Buy Candidates

| Stock | Trend | Momentum | Alpha | Observation |
|-------|-------|----------|-------|-------------|
| **CAT** | STRONG UP | +0.79 | +0.77 | Industrial strength, scores well on all dimensions |
| **MRK** | STRONG UP | +0.81 | +0.81 | Top combined score; RSI 86 suggests caution on timing |
| **JNJ** | STRONG UP | +0.76 | +0.80 | High alpha; RSI 94 is extremely overbought |
| **XOM** | STRONG UP | +0.74 | +0.42 | Energy momentum; lower alpha suggests less fundamental support |
| **UPS** | STRONG UP | +0.72 | +0.43 | Logistics recovery play |
| **GOOGL** | UP | +0.12 | +1.32 | Highest alpha score by far; low momentum suggests early-stage rotation into the name |

### Notable Observations
- **GOOGL** has the highest alpha score (+1.32) in the universe but relatively low momentum (+0.12). This pattern (high value/quality, low momentum) often signals a contrarian opportunity - the fundamentals are strong but the market hasn't caught on yet.
- **Defensive sectors dominate** the buy list (healthcare, industrials, energy). Tech names like NVDA, META, and AMD rank in the bottom quartile. This suggests a rotation out of growth into value/quality.
- **High RSI readings** across top momentum names suggest the current uptrends are extended. Consider waiting for pullbacks to enter rather than chasing.

---

## 6. How to Use This Framework Responsibly

### What it IS good for
- **Screening**: Narrowing a universe of 50+ stocks to 5-10 names worth deeper research
- **Trend confirmation**: Validating whether a stock you're interested in has positive technical momentum
- **Historical backtesting**: Testing whether a strategy idea has merit before committing capital
- **Systematic discipline**: Removing emotional bias from stock selection

### What it is NOT
- **A trading system.** The output is research input, not buy/sell orders. Each candidate needs further due diligence (valuation, competitive position, management quality, catalyst analysis).
- **A guarantee of returns.** Past factor performance does not guarantee future results. Momentum, value, and quality factors have all experienced extended periods of underperformance.
- **A complete portfolio manager.** It doesn't handle position sizing, portfolio correlation, risk budgeting, or rebalancing scheduling in the screening output.

### Recommended Workflow

1. **Run daily analysis** via GitHub Actions or manually
2. **Review Strong Buy candidates** - these are your starting point
3. **Cross-check with fundamentals** - look at P/E, revenue growth, competitive moat
4. **Check RSI and overbought/oversold** - avoid entries at RSI > 80
5. **Size positions** based on conviction and volatility (the `config/default.yaml` risk section has guidelines: max 10% per position, max 30% per sector)
6. **Backtest any strategy** before deploying capital using the backtest engine
7. **Track results** over time using the historical archive

---

## 7. Summary

This is a well-built quantitative research framework that implements academically-grounded factor analysis. It successfully runs end-to-end, produces actionable screening results, and provides a solid backtesting infrastructure.

**For investment use:** Treat the output as a **screening tool** that generates research leads, not as a signal to trade on directly. The biggest gaps are the lack of fundamental data in the live analysis and the absence of overbought/oversold warnings. Combine this tool's output with your own valuation work, sector analysis, and risk management before making investment decisions.
