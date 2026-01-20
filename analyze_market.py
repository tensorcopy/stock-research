#!/usr/bin/env python3
"""
Market Analysis Script
Comprehensive market analysis using the stock research framework

Analyzes:
1. Market trends across a stock universe
2. Momentum signals and rankings
3. Breakout opportunities
4. Composite alpha scores for stock selection
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from tqdm import tqdm

from data.market_data import MarketDataFetcher, get_sp500_symbols
from signals.trend_detector import TrendDetector, TrendDirection
from signals.momentum import MomentumSignals
from signals.breakout import BreakoutDetector
from alpha.composite import CompositeAlpha
from alpha.factors import AlphaFactors


def analyze_market(
    symbols: list[str] = None,
    lookback_days: int = 365,
    top_n: int = 20
) -> dict:
    """
    Run comprehensive market analysis

    Args:
        symbols: List of stock symbols (defaults to S&P 500 subset)
        lookback_days: Days of historical data to fetch
        top_n: Number of top stocks to highlight

    Returns:
        Dictionary with analysis results
    """
    print("=" * 60)
    print("MARKET ANALYSIS REPORT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Default to a diversified subset of major stocks
    if symbols is None:
        symbols = [
            # Technology
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "ADBE",
            # Finance
            "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK", "C", "WFC",
            # Healthcare
            "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
            # Consumer
            "WMT", "HD", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "TGT",
            # Energy & Industrial
            "XOM", "CVX", "CAT", "BA", "UNP", "HON", "GE", "MMM", "UPS", "LMT"
        ]

    print(f"\nAnalyzing {len(symbols)} stocks...")

    # Initialize components
    fetcher = MarketDataFetcher()
    trend_detector = TrendDetector()
    momentum_signals = MomentumSignals()
    breakout_detector = BreakoutDetector()
    alpha_factors = AlphaFactors()
    composite_alpha = CompositeAlpha()

    # Fetch data
    print("\n[1/5] Fetching market data...")
    start_date = datetime.now() - timedelta(days=lookback_days)
    end_date = datetime.now()

    price_data = fetcher.get_ohlcv(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )

    successful_symbols = list(price_data.keys())
    print(f"Successfully fetched data for {len(successful_symbols)}/{len(symbols)} symbols")

    if not successful_symbols:
        print("ERROR: No data fetched. Check network connection and symbol validity.")
        return {"error": "No data available"}

    # Analyze trends
    print("\n[2/5] Analyzing market trends...")
    trend_results = []

    for symbol in tqdm(successful_symbols, desc="Trends"):
        try:
            ohlcv = price_data[symbol]
            trend = trend_detector.get_composite_trend(ohlcv)
            trend_results.append({
                "symbol": symbol,
                "trend_direction": trend.direction.name,
                "trend_value": trend.direction.value,
                "trend_strength": trend.strength,
                "trend_confidence": trend.confidence
            })
        except Exception as e:
            print(f"Trend error for {symbol}: {e}")

    trend_df = pd.DataFrame(trend_results)

    # Analyze momentum
    print("\n[3/5] Calculating momentum signals...")
    momentum_results = []

    for symbol in tqdm(successful_symbols, desc="Momentum"):
        try:
            ohlcv = price_data[symbol]
            mom_signal = momentum_signals.calculate_momentum_score(ohlcv)
            momentum_results.append({
                "symbol": symbol,
                "momentum_score": mom_signal.score,
                "rsi": mom_signal.components.get("rsi", np.nan)
            })
        except Exception as e:
            print(f"Momentum error for {symbol}: {e}")

    momentum_df = pd.DataFrame(momentum_results)

    # Detect breakouts
    print("\n[4/5] Scanning for breakout opportunities...")
    breakout_results = []

    for symbol in tqdm(successful_symbols, desc="Breakouts"):
        try:
            ohlcv = price_data[symbol]
            breakouts = breakout_detector.get_all_breakouts(ohlcv)
            potential = breakout_detector.score_breakout_potential(ohlcv)

            breakout_results.append({
                "symbol": symbol,
                "breakout_count": len(breakouts),
                "breakout_types": [b.breakout_type.value for b in breakouts],
                "breakout_directions": [b.direction for b in breakouts],
                "breakout_potential": potential["breakout_potential_score"],
                "distance_to_resistance": potential.get("distance_to_resistance", np.nan),
                "volume_trend": potential.get("volume_trend", np.nan)
            })
        except Exception as e:
            print(f"Breakout error for {symbol}: {e}")

    breakout_df = pd.DataFrame(breakout_results)

    # Calculate alpha factors
    print("\n[5/5] Computing composite alpha scores...")
    factor_data = {}

    for symbol in tqdm(successful_symbols, desc="Alpha"):
        try:
            ohlcv = price_data[symbol]
            prices = ohlcv["close"]

            # Calculate momentum-based factors
            mom_12_1 = alpha_factors.momentum_12_1(prices)
            mom_6_1 = alpha_factors.momentum_6_1(prices)
            mom_1m = alpha_factors.momentum_1m(prices)

            # Calculate volatility
            returns = prices.pct_change().dropna()
            ivol = returns.std() * np.sqrt(252)

            factor_data[symbol] = {
                "momentum_12_1": mom_12_1.value if not np.isnan(mom_12_1.value) else None,
                "momentum_6_1": mom_6_1.value if not np.isnan(mom_6_1.value) else None,
                "momentum_1m": mom_1m.value if not np.isnan(mom_1m.value) else None,
                "ivol": -ivol  # Negative because low vol is preferred
            }
        except Exception as e:
            print(f"Factor error for {symbol}: {e}")

    # Filter out None values and rank universe
    factor_data_clean = {
        k: {fk: fv for fk, fv in v.items() if fv is not None}
        for k, v in factor_data.items()
        if any(fv is not None for fv in v.values())
    }

    alpha_rankings = composite_alpha.rank_universe(factor_data_clean)

    # Merge all results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    # Market Overview
    print("\n### MARKET OVERVIEW ###")
    if not trend_df.empty:
        strong_up = len(trend_df[trend_df["trend_value"] == 2])
        up = len(trend_df[trend_df["trend_value"] == 1])
        neutral = len(trend_df[trend_df["trend_value"] == 0])
        down = len(trend_df[trend_df["trend_value"] == -1])
        strong_down = len(trend_df[trend_df["trend_value"] == -2])

        print(f"Strong Uptrend:   {strong_up:3d} ({strong_up/len(trend_df)*100:.1f}%)")
        print(f"Uptrend:          {up:3d} ({up/len(trend_df)*100:.1f}%)")
        print(f"Neutral:          {neutral:3d} ({neutral/len(trend_df)*100:.1f}%)")
        print(f"Downtrend:        {down:3d} ({down/len(trend_df)*100:.1f}%)")
        print(f"Strong Downtrend: {strong_down:3d} ({strong_down/len(trend_df)*100:.1f}%)")

        # Market breadth indicator
        bullish_pct = (strong_up + up) / len(trend_df) * 100
        if bullish_pct > 60:
            market_sentiment = "BULLISH"
        elif bullish_pct > 40:
            market_sentiment = "NEUTRAL"
        else:
            market_sentiment = "BEARISH"

        print(f"\nMarket Sentiment: {market_sentiment} ({bullish_pct:.1f}% bullish)")

    # Top Momentum Stocks
    print("\n### TOP MOMENTUM STOCKS ###")
    if not momentum_df.empty:
        top_momentum = momentum_df.nlargest(top_n, "momentum_score")
        for i, row in top_momentum.head(10).iterrows():
            rsi_val = row.get('rsi', 'N/A')
            rsi_str = f"{rsi_val:.1f}" if isinstance(rsi_val, (int, float)) and not np.isnan(rsi_val) else "N/A"
            print(f"{row['symbol']:6s} | Score: {row['momentum_score']:+.3f} | RSI: {rsi_str}")

    # Breakout Opportunities
    print("\n### BREAKOUT OPPORTUNITIES ###")
    if not breakout_df.empty:
        # Stocks with active breakouts
        active_breakouts = breakout_df[breakout_df["breakout_count"] > 0].copy()
        if not active_breakouts.empty:
            print("Active Breakouts:")
            for i, row in active_breakouts.head(10).iterrows():
                directions = row["breakout_directions"]
                direction_str = "BULLISH" if sum(directions) > 0 else "BEARISH" if sum(directions) < 0 else "MIXED"
                types = ", ".join(row["breakout_types"])
                print(f"  {row['symbol']:6s} | {direction_str:7s} | Types: {types}")

        # High potential breakout candidates
        print("\nHigh Potential Candidates:")
        high_potential = breakout_df.nlargest(10, "breakout_potential")
        for i, row in high_potential.head(5).iterrows():
            dist = row.get('distance_to_resistance', np.nan)
            dist_str = f"{dist*100:.1f}%" if not np.isnan(dist) else "N/A"
            print(f"  {row['symbol']:6s} | Potential: {row['breakout_potential']:.1f} | Dist to Resistance: {dist_str}")

    # Alpha Rankings
    print("\n### TOP ALPHA RANKED STOCKS ###")
    if not alpha_rankings.empty:
        print("(Based on momentum, value, quality, and volatility factors)")
        for i, row in alpha_rankings.head(top_n).iterrows():
            percentile = row.get('percentile', 0)
            print(f"{row['symbol']:6s} | Alpha Score: {row['composite_score']:+.3f} | Percentile: {percentile:.1f}")

    # Strong Buy Candidates (intersection of signals)
    print("\n### STRONG BUY CANDIDATES ###")
    print("(Stocks with positive trends, high momentum, AND high alpha)")

    # Merge data for final rankings
    if not trend_df.empty and not momentum_df.empty and not alpha_rankings.empty:
        merged = trend_df.merge(momentum_df, on="symbol").merge(
            alpha_rankings[["symbol", "composite_score", "percentile"]],
            on="symbol",
            how="inner"
        )

        # Filter for strong candidates
        strong_candidates = merged[
            (merged["trend_value"] >= 1) &  # At least uptrending
            (merged["momentum_score"] > 0) &  # Positive momentum
            (merged["composite_score"] > 0)  # Positive alpha
        ].copy()

        if not strong_candidates.empty:
            # Score combined signals
            strong_candidates["combined_score"] = (
                strong_candidates["trend_strength"] * 0.3 +
                strong_candidates["momentum_score"] * 0.4 +
                strong_candidates["composite_score"].clip(-1, 1) * 0.3
            )
            strong_candidates = strong_candidates.sort_values("combined_score", ascending=False)

            for i, row in strong_candidates.head(10).iterrows():
                print(f"{row['symbol']:6s} | Trend: {row['trend_direction']:12s} | "
                      f"Mom: {row['momentum_score']:+.2f} | Alpha: {row['composite_score']:+.2f}")
        else:
            print("No stocks currently meet all strong buy criteria.")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total stocks analyzed: {len(successful_symbols)}")
    print(f"Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Compile results
    results = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "symbols_analyzed": len(successful_symbols),
            "lookback_days": lookback_days
        },
        "market_overview": {
            "sentiment": market_sentiment if 'market_sentiment' in dir() else "UNKNOWN",
            "bullish_pct": bullish_pct if 'bullish_pct' in dir() else 0
        },
        "trends": trend_df.to_dict(orient="records") if not trend_df.empty else [],
        "momentum": momentum_df.to_dict(orient="records") if not momentum_df.empty else [],
        "breakouts": breakout_df.to_dict(orient="records") if not breakout_df.empty else [],
        "alpha_rankings": alpha_rankings.to_dict(orient="records") if not alpha_rankings.empty else [],
        "strong_candidates": strong_candidates.to_dict(orient="records") if 'strong_candidates' in dir() and not strong_candidates.empty else []
    }

    return results


def save_results(results: dict, output_dir: str = "./results"):
    """Save analysis results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results as JSON
    import json
    json_path = output_path / f"market_analysis_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {json_path}")

    # Save summary CSV
    if results.get("alpha_rankings"):
        csv_path = output_path / f"alpha_rankings_{timestamp}.csv"
        pd.DataFrame(results["alpha_rankings"]).to_csv(csv_path, index=False)
        print(f"Alpha rankings saved to: {csv_path}")

    return str(json_path)


if __name__ == "__main__":
    # Run analysis
    results = analyze_market()

    # Save results
    save_results(results)
