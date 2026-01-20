"""
Momentum Signals Module
Generate momentum-based trading signals for alpha capture
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class MomentumSignal:
    """Container for momentum signal"""
    score: float  # -1 to 1
    rank_percentile: float  # 0 to 100
    components: dict


class MomentumSignals:
    """
    Calculate momentum signals for stock selection

    Momentum factors are among the strongest alpha sources:
    - Price momentum (past returns predict future returns)
    - Earnings momentum (revision trends)
    - Volume momentum (accumulation/distribution)

    Time frames matter:
    - Short-term (1-4 weeks): Mean reversion dominates
    - Medium-term (3-12 months): Momentum strongest
    - Long-term (3-5 years): Mean reversion returns
    """

    def __init__(self):
        pass

    def calculate_price_momentum(
        self,
        prices: pd.Series,
        periods: list[int] = None
    ) -> dict[str, float]:
        """
        Calculate price momentum across multiple timeframes

        Classic momentum: 12-month return excluding last month
        (to avoid short-term reversal)
        """
        if periods is None:
            periods = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m

        momentum = {}

        for period in periods:
            if len(prices) > period:
                ret = (prices.iloc[-1] / prices.iloc[-period]) - 1
                momentum[f"mom_{period}d"] = ret

        # Classic momentum (12-1 month)
        if len(prices) > 252:
            ret_12m = (prices.iloc[-21] / prices.iloc[-252]) - 1
            momentum["mom_12m_1m"] = ret_12m

        return momentum

    def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        RSI > 70: Overbought (potential reversal down)
        RSI < 30: Oversold (potential reversal up)
        """
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Bullish: MACD crosses above signal line
        Bearish: MACD crosses below signal line
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        })

    def calculate_rate_of_change(
        self,
        prices: pd.Series,
        period: int = 10
    ) -> pd.Series:
        """
        Calculate Rate of Change (ROC)

        Simple momentum measure: percentage change over period
        """
        return (prices / prices.shift(period) - 1) * 100

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator

        %K > 80: Overbought
        %K < 20: Oversold
        Crossovers of %K and %D generate signals
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return pd.DataFrame({
            "stoch_k": stoch_k,
            "stoch_d": stoch_d
        })

    def calculate_volume_momentum(
        self,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> dict[str, float]:
        """
        Calculate volume-based momentum indicators

        Volume confirms price movements:
        - Rising price + rising volume = strong uptrend
        - Rising price + falling volume = weak uptrend (potential reversal)
        """
        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).cumsum()

        # Volume Rate of Change
        vol_roc = (volume / volume.shift(period) - 1).iloc[-1]

        # Price-Volume Trend
        pvt = ((close.diff() / close.shift(1)) * volume).cumsum()

        # Money Flow Index components
        typical_price = close  # Simplified (should be (H+L+C)/3)
        money_flow = typical_price * volume

        # Volume-weighted momentum
        recent_vol = volume.tail(period)
        recent_close = close.tail(period)
        vol_weighted_return = (recent_close.pct_change() * recent_vol).sum() / recent_vol.sum()

        return {
            "obv_slope": (obv.iloc[-1] - obv.iloc[-period]) / period if len(obv) > period else 0,
            "volume_roc": vol_roc,
            "vol_weighted_return": vol_weighted_return,
            "volume_trend": volume.tail(period).mean() / volume.tail(period * 2).head(period).mean() - 1
        }

    def calculate_momentum_score(
        self,
        ohlcv: pd.DataFrame,
        weights: Optional[dict] = None
    ) -> MomentumSignal:
        """
        Calculate composite momentum score

        Combines multiple momentum indicators into single score
        """
        default_weights = {
            "price_momentum": 0.4,
            "rsi": 0.15,
            "macd": 0.2,
            "stochastic": 0.1,
            "volume": 0.15
        }
        weights = weights or default_weights

        components = {}
        scores = {}

        # Price momentum (normalize to -1 to 1)
        price_mom = self.calculate_price_momentum(ohlcv["close"])
        mom_score = price_mom.get("mom_12m_1m", price_mom.get("mom_126d", 0))
        scores["price_momentum"] = np.clip(mom_score * 2, -1, 1)  # 50% return = max score
        components["price_momentum"] = price_mom

        # RSI (normalize: 30-70 range to -1 to 1)
        rsi = self.calculate_rsi(ohlcv["close"])
        current_rsi = rsi.iloc[-1]
        scores["rsi"] = (current_rsi - 50) / 50  # 0 at 50, ±1 at extremes
        components["rsi"] = current_rsi

        # MACD (use histogram for momentum direction)
        macd = self.calculate_macd(ohlcv["close"])
        macd_hist = macd["histogram"].iloc[-1]
        macd_norm = macd_hist / ohlcv["close"].iloc[-1] * 100
        scores["macd"] = np.clip(macd_norm * 10, -1, 1)
        components["macd"] = {
            "macd": macd["macd"].iloc[-1],
            "signal": macd["signal"].iloc[-1],
            "histogram": macd_hist
        }

        # Stochastic
        stoch = self.calculate_stochastic(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        stoch_k = stoch["stoch_k"].iloc[-1]
        scores["stochastic"] = (stoch_k - 50) / 50
        components["stochastic"] = {
            "k": stoch_k,
            "d": stoch["stoch_d"].iloc[-1]
        }

        # Volume momentum
        vol_mom = self.calculate_volume_momentum(ohlcv["close"], ohlcv["volume"])
        scores["volume"] = np.clip(vol_mom["vol_weighted_return"] * 10, -1, 1)
        components["volume"] = vol_mom

        # Calculate weighted composite score
        composite_score = sum(
            scores[k] * weights[k]
            for k in weights
            if k in scores
        )

        return MomentumSignal(
            score=composite_score,
            rank_percentile=0,  # To be filled when ranking across universe
            components=components
        )

    def rank_momentum(
        self,
        universe: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Rank stocks by momentum within a universe

        Args:
            universe: Dict mapping symbols to OHLCV DataFrames

        Returns:
            DataFrame with momentum scores and ranks
        """
        results = []

        for symbol, ohlcv in universe.items():
            try:
                signal = self.calculate_momentum_score(ohlcv)
                results.append({
                    "symbol": symbol,
                    "momentum_score": signal.score,
                    **{f"comp_{k}": v if not isinstance(v, dict) else str(v)
                       for k, v in signal.components.items()}
                })
            except Exception as e:
                print(f"Error calculating momentum for {symbol}: {e}")

        df = pd.DataFrame(results)

        if not df.empty:
            df["momentum_rank"] = df["momentum_score"].rank(ascending=False)
            df["momentum_percentile"] = df["momentum_score"].rank(pct=True) * 100

        return df.sort_values("momentum_score", ascending=False)
