#!/usr/bin/env python3
"""
Market Analysis Script - Actionable Signals Edition

Produces direct buy/hold/avoid signals by combining:
1. Market trends (MA crossover, ADX, linear regression, swing analysis)
2. Momentum signals with overbought/oversold filtering
3. Breakout opportunities
4. Multi-factor composite alpha (momentum + value + quality + volatility)
5. Sector-relative scoring
6. SPY benchmark comparison
7. Risk-adjusted position sizing
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
from data.fundamentals import FundamentalsFetcher
from signals.trend_detector import TrendDetector, TrendDirection
from signals.momentum import MomentumSignals
from signals.breakout import BreakoutDetector
from alpha.composite import CompositeAlpha
from alpha.factors import AlphaFactors


# ---------------------------------------------------------------------------
# Sector map fallback for the default 50-stock universe
# Used when yfinance sector lookup is unavailable
# ---------------------------------------------------------------------------
DEFAULT_SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "NVDA": "Technology", "META": "Technology",
    "TSLA": "Technology", "AMD": "Technology", "CRM": "Technology",
    "ADBE": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "V": "Financials", "MA": "Financials",
    "AXP": "Financials", "BLK": "Financials", "C": "Financials",
    "WFC": "Financials",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "BMY": "Healthcare",
    "AMGN": "Healthcare",
    # Consumer
    "WMT": "Consumer", "HD": "Consumer", "PG": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "COST": "Consumer",
    "NKE": "Consumer", "MCD": "Consumer", "SBUX": "Consumer",
    "TGT": "Consumer",
    # Energy
    "XOM": "Energy", "CVX": "Energy",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "UNP": "Industrials",
    "HON": "Industrials", "GE": "Industrials", "MMM": "Industrials",
    "UPS": "Industrials", "LMT": "Industrials",
}


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

class SignalRating:
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD_OVERBOUGHT = "HOLD_OB"
    HOLD = "HOLD"
    AVOID = "AVOID"


def classify_signal(
    trend_value: int,
    momentum_score: float,
    alpha_percentile: float,
    rsi: float,
    relative_strength: float,
    sector_rank: int,
    sector_size: int,
) -> tuple[str, float, str]:
    """
    Classify a stock into an actionable signal rating.

    Returns:
        (signal_rating, confidence 0-1, reason string)
    """
    # --- score confidence from individual factors ---
    confidence = 0.0
    positive_trend = trend_value >= 1
    strong_trend = trend_value >= 2
    good_momentum = momentum_score > 0.2
    strong_momentum = momentum_score > 0.4
    good_alpha = alpha_percentile >= 60
    strong_alpha = alpha_percentile >= 75
    outperforming_spy = relative_strength > 0
    top_in_sector = sector_rank <= max(1, sector_size // 3)

    if positive_trend:
        confidence += 0.20
    if strong_trend:
        confidence += 0.10
    if good_momentum:
        confidence += 0.15
    if strong_momentum:
        confidence += 0.10
    if good_alpha:
        confidence += 0.15
    if strong_alpha:
        confidence += 0.10
    if outperforming_spy:
        confidence += 0.10
    if top_in_sector:
        confidence += 0.10

    overbought = rsi >= 75
    severely_overbought = rsi >= 85

    # --- classify ---
    reasons = []

    # Overbought stocks that would otherwise qualify
    if overbought and positive_trend and (good_momentum or good_alpha):
        if severely_overbought:
            reasons.append(f"RSI {rsi:.0f} severely overbought - wait for pullback to <65")
        else:
            reasons.append(f"RSI {rsi:.0f} overbought - wait for pullback to <70")
        return SignalRating.HOLD_OVERBOUGHT, round(confidence * 0.5, 2), "; ".join(reasons)

    # STRONG_BUY: all factors aligned
    if (strong_trend and strong_momentum and good_alpha
            and outperforming_spy and not overbought):
        reasons.append("All signals aligned: strong trend + momentum + alpha + beating SPY")
        if top_in_sector:
            reasons.append(f"#{sector_rank} in sector")
        return SignalRating.STRONG_BUY, round(confidence, 2), "; ".join(reasons)

    # BUY: most factors aligned
    if positive_trend and (good_momentum or good_alpha) and not overbought:
        if good_momentum and good_alpha:
            reasons.append("Positive trend with good momentum and alpha")
        elif good_momentum:
            reasons.append("Positive trend with good momentum")
        else:
            reasons.append("Positive trend with strong alpha")
        if outperforming_spy:
            reasons.append("Outperforming SPY")
        return SignalRating.BUY, round(confidence, 2), "; ".join(reasons)

    # AVOID: weak on most dimensions
    if trend_value <= 0 and alpha_percentile < 40:
        reasons.append("Weak/no trend and below-average alpha")
        return SignalRating.AVOID, round(confidence, 2), "; ".join(reasons)

    reasons.append("Mixed signals - insufficient conviction")
    return SignalRating.HOLD, round(confidence, 2), "; ".join(reasons)


def calculate_position_size(
    stock_volatility: float,
    median_volatility: float,
    confidence: float,
    n_positions: int,
    max_size: float = 0.10,
) -> float:
    """
    Inverse-volatility weighted position size, adjusted by confidence.

    Returns:
        Suggested allocation as fraction of portfolio (e.g. 0.05 = 5%).
    """
    base_size = 1.0 / max(1, n_positions)
    vol_adj = min(2.0, median_volatility / stock_volatility) if stock_volatility > 0 else 1.0
    conf_adj = 0.5 + 0.5 * confidence
    return min(base_size * vol_adj * conf_adj, max_size)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_market(
    symbols: list[str] = None,
    lookback_days: int = 365,
    top_n: int = 20
) -> dict:
    """
    Run comprehensive market analysis and produce actionable signals.

    Returns:
        Dictionary with all analysis results.
    """
    print("=" * 70)
    print("MARKET ANALYSIS REPORT - ACTIONABLE SIGNALS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

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

    print(f"\nAnalyzing {len(symbols)} stocks + SPY benchmark...")

    # Initialize components
    fetcher = MarketDataFetcher()
    fundamentals_fetcher = FundamentalsFetcher()
    trend_detector = TrendDetector()
    momentum_signals = MomentumSignals()
    breakout_detector = BreakoutDetector()
    alpha_factors = AlphaFactors()
    composite_alpha = CompositeAlpha()

    start_date = datetime.now() - timedelta(days=lookback_days)
    end_date = datetime.now()

    # ── [1/7] Fetch market data (including SPY benchmark) ─────────────────
    print("\n[1/7] Fetching market data...")
    all_symbols = list(dict.fromkeys(symbols + ["SPY"]))  # deduplicate
    price_data = fetcher.get_ohlcv(
        symbols=all_symbols,
        start_date=start_date,
        end_date=end_date
    )

    spy_data = price_data.pop("SPY", None)
    successful_symbols = list(price_data.keys())
    spy_status = "OK" if spy_data is not None else "UNAVAILABLE"
    print(f"Stocks: {len(successful_symbols)}/{len(symbols)} | SPY benchmark: {spy_status}")

    if not successful_symbols:
        print("ERROR: No data fetched. Check network connection.")
        return {"error": "No data available"}

    # Compute SPY returns for later use
    spy_returns = None
    spy_total_return = None
    if spy_data is not None:
        spy_prices = spy_data["close"]
        spy_returns = spy_prices.pct_change().dropna()
        spy_total_return = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1

    # ── [2/7] Fetch fundamental data & sector classification ──────────────
    print("\n[2/7] Fetching fundamental data & sectors...")
    fundamentals_df = fundamentals_fetcher.get_key_metrics(successful_symbols)

    # Build lookup dicts
    fund_map: dict[str, dict] = {}
    sector_map: dict[str, str] = {}

    if not fundamentals_df.empty:
        for _, row in fundamentals_df.iterrows():
            sym = row["symbol"]
            fund_map[sym] = row.to_dict()
            sector = row.get("sector", None)
            if sector and sector != "Unknown":
                sector_map[sym] = sector

    # Fill in missing sectors from hardcoded fallback
    for sym in successful_symbols:
        if sym not in sector_map:
            sector_map[sym] = DEFAULT_SECTOR_MAP.get(sym, "Other")

    n_with_fundamentals = sum(1 for s in successful_symbols if s in fund_map)
    print(f"Fundamentals: {n_with_fundamentals}/{len(successful_symbols)} stocks")

    # ── [3/7] Analyze market trends ───────────────────────────────────────
    print("\n[3/7] Analyzing market trends...")
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

    # ── [4/7] Calculate momentum signals ──────────────────────────────────
    print("\n[4/7] Calculating momentum signals...")
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

    # ── [5/7] Scan for breakout opportunities ─────────────────────────────
    print("\n[5/7] Scanning for breakout opportunities...")
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

    # ── [6/7] Compute composite alpha WITH fundamentals ───────────────────
    print("\n[6/7] Computing composite alpha scores (price + fundamental factors)...")
    factor_data = {}
    volatilities = {}

    for symbol in tqdm(successful_symbols, desc="Alpha"):
        try:
            ohlcv = price_data[symbol]
            prices = ohlcv["close"]
            returns = prices.pct_change().dropna()

            # --- Price-based factors ---
            mom_12_1 = alpha_factors.momentum_12_1(prices)
            mom_6_1 = alpha_factors.momentum_6_1(prices)

            ann_vol = returns.std() * np.sqrt(252)
            volatilities[symbol] = ann_vol

            sym_factors: dict = {
                "momentum_12_1": mom_12_1.value if not np.isnan(mom_12_1.value) else None,
                "momentum_6_1": mom_6_1.value if not np.isnan(mom_6_1.value) else None,
            }

            # Proper IVOL & beta vs SPY (instead of raw volatility)
            if spy_returns is not None and len(returns) >= 60:
                ivol_factor = alpha_factors.idiosyncratic_volatility(returns, spy_returns)
                beta_factor = alpha_factors.beta(returns, spy_returns)
                sym_factors["ivol"] = ivol_factor.value  # already negated
                sym_factors["beta"] = beta_factor.value  # already negated
            else:
                sym_factors["ivol"] = -ann_vol

            # --- Fundamental factors (when available) ---
            fd = fund_map.get(symbol, {})

            pe = fd.get("pe_ratio")
            if pe and isinstance(pe, (int, float)) and pe > 0:
                sym_factors["earnings_yield"] = 1.0 / pe

            pb = fd.get("pb_ratio")
            if pb and isinstance(pb, (int, float)) and pb > 0:
                sym_factors["book_to_market"] = 1.0 / pb

            ev_eb = fd.get("ev_ebitda")
            if ev_eb and isinstance(ev_eb, (int, float)) and ev_eb > 0:
                sym_factors["fcf_yield"] = 1.0 / ev_eb

            roe = fd.get("roe")
            if roe is not None and isinstance(roe, (int, float)) and not np.isnan(roe):
                sym_factors["roe"] = roe

            gm = fd.get("gross_margin")
            if gm is not None and isinstance(gm, (int, float)) and not np.isnan(gm):
                sym_factors["gross_profitability"] = gm

            rg = fd.get("revenue_growth")
            if rg is not None and isinstance(rg, (int, float)) and not np.isnan(rg):
                sym_factors["revenue_growth"] = rg

            mktcap = fd.get("market_cap")
            if mktcap and isinstance(mktcap, (int, float)) and mktcap > 0:
                sym_factors["size"] = -np.log(mktcap)

            # Clean None values
            factor_data[symbol] = {k: v for k, v in sym_factors.items() if v is not None}

        except Exception as e:
            print(f"Factor error for {symbol}: {e}")

    # Rank universe with full multi-factor model
    alpha_rankings = composite_alpha.rank_universe(factor_data)

    # ── [7/7] Generate actionable signals ─────────────────────────────────
    print("\n[7/7] Generating actionable signals...")

    # -- Sector-relative scoring --
    if not alpha_rankings.empty:
        alpha_rankings["sector"] = alpha_rankings["symbol"].map(sector_map).fillna("Other")

        # Sector z-score (only for sectors with 3+ stocks)
        def _sector_z(group):
            if len(group) < 3 or group.std() == 0:
                return pd.Series(0.0, index=group.index)
            return (group - group.mean()) / group.std()

        alpha_rankings["sector_z_score"] = alpha_rankings.groupby("sector")[
            "composite_score"
        ].transform(_sector_z)

        alpha_rankings["sector_rank"] = alpha_rankings.groupby("sector")[
            "composite_score"
        ].rank(ascending=False).astype(int)

        sector_sizes = alpha_rankings.groupby("sector")["symbol"].transform("count")
        alpha_rankings["sector_size"] = sector_sizes.astype(int)

    # -- Relative strength vs SPY --
    relative_strengths = {}
    stock_betas = {}
    if spy_data is not None and spy_total_return is not None:
        for symbol in successful_symbols:
            try:
                prices = price_data[symbol]["close"]
                stock_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
                relative_strengths[symbol] = stock_ret - spy_total_return

                stock_returns = prices.pct_change().dropna()
                aligned = pd.concat([stock_returns, spy_returns], axis=1).dropna()
                if len(aligned) > 20:
                    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
                    var = aligned.iloc[:, 1].var()
                    stock_betas[symbol] = cov / var if var > 0 else 1.0
            except Exception:
                pass

    # -- Build merged DataFrame with all signals --
    merged = pd.DataFrame()
    if not trend_df.empty and not momentum_df.empty and not alpha_rankings.empty:
        merged = trend_df.merge(momentum_df, on="symbol").merge(
            alpha_rankings[["symbol", "composite_score", "percentile",
                            "sector", "sector_rank", "sector_size",
                            "sector_z_score"]],
            on="symbol", how="inner"
        )
        merged["relative_strength"] = merged["symbol"].map(relative_strengths).fillna(0)
        merged["beta"] = merged["symbol"].map(stock_betas).fillna(1.0)
        merged["volatility"] = merged["symbol"].map(volatilities).fillna(0.20)

    # -- Classify every stock --
    signal_rows = []
    if not merged.empty:
        for _, row in merged.iterrows():
            rating, conf, reason = classify_signal(
                trend_value=int(row["trend_value"]),
                momentum_score=float(row["momentum_score"]),
                alpha_percentile=float(row["percentile"]),
                rsi=float(row["rsi"]) if not np.isnan(row["rsi"]) else 50.0,
                relative_strength=float(row["relative_strength"]),
                sector_rank=int(row["sector_rank"]),
                sector_size=int(row["sector_size"]),
            )
            signal_rows.append({
                **row.to_dict(),
                "signal": rating,
                "confidence": conf,
                "reason": reason,
            })

    signals_df = pd.DataFrame(signal_rows) if signal_rows else pd.DataFrame()

    # -- Position sizing for BUY / STRONG_BUY --
    buy_signals = pd.DataFrame()
    if not signals_df.empty:
        buy_mask = signals_df["signal"].isin([SignalRating.STRONG_BUY, SignalRating.BUY])
        buy_signals = signals_df[buy_mask].copy()

        if not buy_signals.empty:
            median_vol = np.median(list(volatilities.values())) if volatilities else 0.20
            n_pos = len(buy_signals)
            buy_signals["position_size"] = buy_signals.apply(
                lambda r: calculate_position_size(
                    stock_volatility=r["volatility"],
                    median_volatility=median_vol,
                    confidence=r["confidence"],
                    n_positions=n_pos,
                ),
                axis=1,
            )
            # Normalize so sizes sum to <= 1.0
            total = buy_signals["position_size"].sum()
            if total > 1.0:
                buy_signals["position_size"] = buy_signals["position_size"] / total

            # Enforce max sector exposure (30%)
            for sector in buy_signals["sector"].unique():
                sec_mask = buy_signals["sector"] == sector
                sec_total = buy_signals.loc[sec_mask, "position_size"].sum()
                if sec_total > 0.30:
                    buy_signals.loc[sec_mask, "position_size"] *= 0.30 / sec_total

            buy_signals = buy_signals.sort_values("confidence", ascending=False)

    # ══════════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    # --- Market Overview ---
    print("\n### MARKET OVERVIEW ###")
    market_sentiment = "UNKNOWN"
    bullish_pct = 0.0
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

        bullish_pct = (strong_up + up) / len(trend_df) * 100
        if bullish_pct > 60:
            market_sentiment = "BULLISH"
        elif bullish_pct > 40:
            market_sentiment = "NEUTRAL"
        else:
            market_sentiment = "BEARISH"

        print(f"\nMarket Sentiment: {market_sentiment} ({bullish_pct:.1f}% bullish)")

    # --- SPY Benchmark ---
    print("\n### SPY BENCHMARK ###")
    if spy_total_return is not None:
        print(f"SPY Total Return ({lookback_days}d): {spy_total_return:+.1%}")

        if not merged.empty:
            n_beating = sum(1 for v in relative_strengths.values() if v > 0)
            print(f"Stocks Beating SPY: {n_beating}/{len(relative_strengths)} "
                  f"({n_beating/max(1,len(relative_strengths))*100:.0f}%)")

            top_rel = sorted(relative_strengths.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop Relative Performers vs SPY:")
            for sym, rs in top_rel:
                print(f"  {sym:6s} | Excess Return: {rs:+.1%}")
    else:
        print("SPY data unavailable - benchmark comparison skipped.")

    # --- Breakout Opportunities ---
    print("\n### BREAKOUT OPPORTUNITIES ###")
    if not breakout_df.empty:
        active_breakouts = breakout_df[breakout_df["breakout_count"] > 0].copy()
        if not active_breakouts.empty:
            print("Active Breakouts:")
            for _, row in active_breakouts.head(10).iterrows():
                directions = row["breakout_directions"]
                direction_str = "BULLISH" if sum(directions) > 0 else "BEARISH" if sum(directions) < 0 else "MIXED"
                types = ", ".join(row["breakout_types"])
                print(f"  {row['symbol']:6s} | {direction_str:7s} | Types: {types}")

    # --- Alpha Rankings (with sector context) ---
    print("\n### TOP ALPHA RANKED STOCKS ###")
    print("(Multi-factor: momentum + value + quality + volatility)")
    if not alpha_rankings.empty:
        for _, row in alpha_rankings.head(top_n).iterrows():
            sec = row.get("sector", "?")
            sec_rk = row.get("sector_rank", "?")
            print(f"{row['symbol']:6s} | Alpha: {row['composite_score']:+.3f} "
                  f"| Pctl: {row['percentile']:.0f} "
                  f"| {sec} #{sec_rk}")

    # --- Sector-Relative Top Picks ---
    print("\n### TOP STOCKS BY SECTOR ###")
    if not alpha_rankings.empty:
        for sector in sorted(alpha_rankings["sector"].unique()):
            sec_df = alpha_rankings[alpha_rankings["sector"] == sector].head(3)
            print(f"  {sector}:")
            for _, row in sec_df.iterrows():
                print(f"    {row['symbol']:6s} | Alpha: {row['composite_score']:+.3f} "
                      f"| Sector Z: {row['sector_z_score']:+.2f}")

    # ═══════════════════════════════════════════════════════════════════
    #  ACTIONABLE SIGNALS (the main output)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ACTIONABLE SIGNALS")
    print("=" * 70)

    if not signals_df.empty:
        # STRONG BUY
        sb = signals_df[signals_df["signal"] == SignalRating.STRONG_BUY].sort_values(
            "confidence", ascending=False
        )
        print(f"\n  STRONG BUY ({len(sb)}):")
        if sb.empty:
            print("    (none)")
        for _, r in sb.iterrows():
            print(f"    {r['symbol']:6s} | Conf: {r['confidence']:.0%} | Trend: {r['trend_direction']:12s} "
                  f"| Mom: {r['momentum_score']:+.2f} | Alpha: {r['composite_score']:+.2f} "
                  f"| RSI: {r['rsi']:.0f} | vs SPY: {r['relative_strength']:+.1%}")
            print(f"           {r['reason']}")

        # BUY
        b = signals_df[signals_df["signal"] == SignalRating.BUY].sort_values(
            "confidence", ascending=False
        )
        print(f"\n  BUY ({len(b)}):")
        if b.empty:
            print("    (none)")
        for _, r in b.iterrows():
            print(f"    {r['symbol']:6s} | Conf: {r['confidence']:.0%} | Trend: {r['trend_direction']:12s} "
                  f"| Mom: {r['momentum_score']:+.2f} | Alpha: {r['composite_score']:+.2f} "
                  f"| RSI: {r['rsi']:.0f} | vs SPY: {r['relative_strength']:+.1%}")
            print(f"           {r['reason']}")

        # HOLD (Overbought)
        ho = signals_df[signals_df["signal"] == SignalRating.HOLD_OVERBOUGHT].sort_values(
            "composite_score", ascending=False
        )
        if not ho.empty:
            print(f"\n  HOLD - OVERBOUGHT ({len(ho)}):")
            print("  (Would qualify but RSI too high - wait for pullback)")
            for _, r in ho.iterrows():
                print(f"    {r['symbol']:6s} | RSI: {r['rsi']:.0f} | Trend: {r['trend_direction']:12s} "
                      f"| Mom: {r['momentum_score']:+.2f} | Alpha: {r['composite_score']:+.2f}")

        # AVOID
        av = signals_df[signals_df["signal"] == SignalRating.AVOID].sort_values(
            "composite_score", ascending=True
        )
        if not av.empty:
            print(f"\n  AVOID ({len(av)}):")
            for _, r in av.head(10).iterrows():
                print(f"    {r['symbol']:6s} | Trend: {r['trend_direction']:12s} "
                      f"| Alpha Pctl: {r['percentile']:.0f}")

        # Count HOLD
        h = signals_df[signals_df["signal"] == SignalRating.HOLD]
        if not h.empty:
            hold_syms = ", ".join(h["symbol"].tolist())
            print(f"\n  HOLD ({len(h)}): {hold_syms}")
    else:
        print("\n  No signals generated - check data availability.")

    # --- Position Sizing ---
    print("\n" + "-" * 70)
    print("POSITION SIZING GUIDANCE")
    print("-" * 70)
    if not buy_signals.empty:
        capital = 100_000
        print(f"Method: Inverse-Volatility | Capital: ${capital:,.0f} "
              f"| Max Position: 10% | Max Sector: 30%")
        print()
        print(f"{'Symbol':<8} {'Signal':<12} {'Sector':<14} {'Weight':>7} {'$ Amount':>10} {'Conf':>6}")
        print("-" * 62)
        total_alloc = 0
        for _, r in buy_signals.iterrows():
            wt = r["position_size"]
            amt = wt * capital
            total_alloc += wt
            print(f"{r['symbol']:<8} {r['signal']:<12} {r['sector']:<14} {wt:>6.1%} {amt:>9,.0f}  {r['confidence']:>5.0%}")

        cash_pct = 1.0 - total_alloc
        print("-" * 62)
        print(f"{'Allocated':<36} {total_alloc:>6.1%} {total_alloc*capital:>9,.0f}")
        print(f"{'Cash Reserve':<36} {cash_pct:>6.1%} {cash_pct*capital:>9,.0f}")
    else:
        print("No BUY or STRONG_BUY signals generated. Stay in cash or hold existing positions.")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total stocks analyzed: {len(successful_symbols)}")
    print(f"Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    if not signals_df.empty:
        for rating in [SignalRating.STRONG_BUY, SignalRating.BUY,
                       SignalRating.HOLD_OVERBOUGHT, SignalRating.HOLD, SignalRating.AVOID]:
            cnt = len(signals_df[signals_df["signal"] == rating])
            if cnt > 0:
                print(f"  {rating:<14s}: {cnt}")
    print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Compile results dict ──────────────────────────────────────────────
    results = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "symbols_analyzed": len(successful_symbols),
            "lookback_days": lookback_days,
            "fundamentals_available": n_with_fundamentals,
        },
        "market_overview": {
            "sentiment": market_sentiment,
            "bullish_pct": bullish_pct,
        },
        "benchmark": {
            "symbol": "SPY",
            "total_return": spy_total_return,
            "stocks_beating_spy": sum(1 for v in relative_strengths.values() if v > 0),
        },
        "trends": trend_df.to_dict(orient="records") if not trend_df.empty else [],
        "momentum": momentum_df.to_dict(orient="records") if not momentum_df.empty else [],
        "breakouts": breakout_df.to_dict(orient="records") if not breakout_df.empty else [],
        "alpha_rankings": alpha_rankings.to_dict(orient="records") if not alpha_rankings.empty else [],
        "signals": signals_df.to_dict(orient="records") if not signals_df.empty else [],
        "buy_signals": buy_signals.to_dict(orient="records") if not buy_signals.empty else [],
    }

    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: str = "./results"):
    """Save analysis results to files."""
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Historical archive
    history_path = output_path / date_str
    history_path.mkdir(parents=True, exist_ok=True)

    market = results.get("market_overview", {})
    benchmark = results.get("benchmark", {})
    buy_signals = results.get("buy_signals", [])
    alpha_rankings = results.get("alpha_rankings", [])
    all_signals = results.get("signals", [])

    # ── Build summary markdown ────────────────────────────────────────────
    lines = [
        "# Market Analysis - Actionable Signals",
        "",
        f"**Date:** {date_str}  ",
        f"**Sentiment:** {market.get('sentiment', 'N/A')} ({market.get('bullish_pct', 0):.1f}% bullish)  ",
    ]

    spy_ret = benchmark.get("total_return")
    if spy_ret is not None:
        lines.append(f"**SPY Benchmark:** {spy_ret:+.1%}  ")

    lines += ["", "---", ""]

    # Actionable signals table
    strong_buys = [s for s in all_signals if s.get("signal") == "STRONG_BUY"]
    buys = [s for s in all_signals if s.get("signal") == "BUY"]
    holds_ob = [s for s in all_signals if s.get("signal") == "HOLD_OB"]

    if strong_buys or buys:
        lines += ["## Actionable Signals", ""]
        lines += [
            "| Signal | Symbol | Sector | Trend | Mom | Alpha | RSI | vs SPY | Conf |",
            "|--------|--------|--------|-------|-----|-------|-----|--------|------|",
        ]
        for s in strong_buys + buys:
            rsi_val = s.get("rsi", 0)
            rsi_str = f"{rsi_val:.0f}" if isinstance(rsi_val, (int, float)) and not np.isnan(rsi_val) else "N/A"
            rs = s.get("relative_strength", 0)
            lines.append(
                f"| **{s['signal']}** | **{s['symbol']}** | {s.get('sector', '?')} "
                f"| {s['trend_direction']} | {s['momentum_score']:+.2f} "
                f"| {s['composite_score']:+.2f} | {rsi_str} | {rs:+.1%} "
                f"| {s.get('confidence', 0):.0%} |"
            )

    if holds_ob:
        lines += ["", "### Overbought - Wait for Pullback", ""]
        lines += [
            "| Symbol | RSI | Trend | Alpha | Reason |",
            "|--------|-----|-------|-------|--------|",
        ]
        for s in holds_ob:
            rsi_val = s.get("rsi", 0)
            rsi_str = f"{rsi_val:.0f}" if isinstance(rsi_val, (int, float)) and not np.isnan(rsi_val) else "N/A"
            lines.append(
                f"| {s['symbol']} | {rsi_str} | {s['trend_direction']} "
                f"| {s['composite_score']:+.2f} | {s.get('reason', '')} |"
            )

    # Position sizing
    if buy_signals:
        lines += ["", "---", "", "## Position Sizing (Inverse-Volatility Weighted)", ""]
        lines += [
            "| Symbol | Signal | Sector | Weight | Est. $100K |",
            "|--------|--------|--------|--------|------------|",
        ]
        for s in buy_signals:
            wt = s.get("position_size", 0)
            lines.append(
                f"| {s['symbol']} | {s['signal']} | {s.get('sector', '?')} "
                f"| {wt:.1%} | ${wt * 100_000:,.0f} |"
            )

    # Full alpha rankings
    lines += ["", "---", "", "## Full Alpha Rankings", ""]
    lines += [
        "| Symbol | Alpha Score | Pctl | Sector | Sector Rank |",
        "|--------|-------------|------|--------|-------------|",
    ]
    for r in alpha_rankings[:20]:
        lines.append(
            f"| {r['symbol']} | {r['composite_score']:+.3f} | {r.get('percentile', 0):.0f} "
            f"| {r.get('sector', '?')} | #{r.get('sector_rank', '?')} |"
        )

    lines += ["", "---", "", f"*Updated: {time_str} UTC*"]
    summary_content = "\n".join(lines)

    # ── Write files ───────────────────────────────────────────────────────
    with open(history_path / "analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if alpha_rankings:
        pd.DataFrame(alpha_rankings).to_csv(history_path / "alpha_rankings.csv", index=False)

    with open(history_path / "SUMMARY.md", "w") as f:
        f.write(summary_content)

    # Latest copies
    with open(output_path / "latest.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if alpha_rankings:
        pd.DataFrame(alpha_rankings).to_csv(output_path / "alpha_rankings.csv", index=False)

    with open(output_path / "SUMMARY.md", "w") as f:
        f.write(summary_content)

    print(f"\nResults saved:")
    print(f"  Latest:  {output_path}/")
    print(f"  History: {history_path}/")

    return str(history_path)


if __name__ == "__main__":
    # Run analysis
    results = analyze_market()

    # Save results
    save_results(results)
