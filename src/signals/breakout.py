"""
Breakout Detection Module
Identify price breakouts and potential trend initiations
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


class BreakoutType(Enum):
    """Type of breakout detected"""
    RESISTANCE_BREAK = "resistance"
    SUPPORT_BREAK = "support"
    RANGE_EXPANSION = "range_expansion"
    VOLUME_BREAKOUT = "volume_breakout"
    VOLATILITY_EXPANSION = "volatility_expansion"


@dataclass
class BreakoutSignal:
    """Container for breakout signal"""
    breakout_type: BreakoutType
    direction: int  # 1 for bullish, -1 for bearish
    strength: float  # 0-1 scale
    price_level: float
    volume_confirmation: bool


class BreakoutDetector:
    """
    Detect breakouts for trend capture

    Breakouts are key for catching new trends early:
    - Price breaking key levels (support/resistance)
    - Volume expansion confirming the move
    - Volatility expansion indicating regime change
    """

    def __init__(self):
        pass

    def find_support_resistance(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 50,
        tolerance: float = 0.02
    ) -> dict:
        """
        Identify key support and resistance levels

        Uses pivot points and price clustering
        """
        # Find local highs and lows
        local_highs = []
        local_lows = []

        window = 5
        for i in range(window, len(high) - window):
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                local_highs.append(high.iloc[i])
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                local_lows.append(low.iloc[i])

        # Cluster nearby levels
        def cluster_levels(levels: list, tolerance: float) -> list:
            if not levels:
                return []

            sorted_levels = sorted(levels)
            clusters = [[sorted_levels[0]]]

            for level in sorted_levels[1:]:
                if (level - clusters[-1][-1]) / clusters[-1][-1] < tolerance:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])

            return [np.mean(cluster) for cluster in clusters]

        resistance_levels = cluster_levels(local_highs, tolerance)
        support_levels = cluster_levels(local_lows, tolerance)

        current_price = close.iloc[-1]

        # Find nearest levels
        resistance_above = [r for r in resistance_levels if r > current_price]
        support_below = [s for s in support_levels if s < current_price]

        return {
            "resistance_levels": resistance_levels,
            "support_levels": support_levels,
            "nearest_resistance": min(resistance_above) if resistance_above else None,
            "nearest_support": max(support_below) if support_below else None,
            "current_price": current_price
        }

    def detect_price_breakout(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 20
    ) -> Optional[BreakoutSignal]:
        """
        Detect if price is breaking out of recent range

        Breakout criteria:
        1. Price exceeds lookback high/low
        2. Volume above average
        3. Close near the extreme (not just wick)
        """
        if len(ohlcv) < lookback:
            return None

        recent = ohlcv.tail(lookback)
        current = ohlcv.iloc[-1]

        lookback_high = recent["high"].max()
        lookback_low = recent["low"].min()
        avg_volume = recent["volume"].mean()

        # Check for breakout
        breakout_up = current["close"] > lookback_high
        breakout_down = current["close"] < lookback_low
        volume_confirm = current["volume"] > avg_volume * 1.5

        if breakout_up:
            # Calculate strength based on how far above the level
            strength = (current["close"] - lookback_high) / lookback_high
            return BreakoutSignal(
                breakout_type=BreakoutType.RESISTANCE_BREAK,
                direction=1,
                strength=min(1.0, strength * 20),  # 5% = max strength
                price_level=lookback_high,
                volume_confirmation=volume_confirm
            )
        elif breakout_down:
            strength = (lookback_low - current["close"]) / lookback_low
            return BreakoutSignal(
                breakout_type=BreakoutType.SUPPORT_BREAK,
                direction=-1,
                strength=min(1.0, strength * 20),
                price_level=lookback_low,
                volume_confirmation=volume_confirm
            )

        return None

    def detect_volume_breakout(
        self,
        ohlcv: pd.DataFrame,
        threshold: float = 2.0,
        lookback: int = 20
    ) -> Optional[BreakoutSignal]:
        """
        Detect volume breakouts

        Unusually high volume often precedes or confirms price moves
        """
        if len(ohlcv) < lookback:
            return None

        current = ohlcv.iloc[-1]
        avg_volume = ohlcv["volume"].tail(lookback).mean()

        if current["volume"] > avg_volume * threshold:
            # Determine direction from price action
            price_change = current["close"] - current["open"]
            direction = 1 if price_change > 0 else -1

            strength = min(1.0, (current["volume"] / avg_volume - 1) / 3)

            return BreakoutSignal(
                breakout_type=BreakoutType.VOLUME_BREAKOUT,
                direction=direction,
                strength=strength,
                price_level=current["close"],
                volume_confirmation=True
            )

        return None

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def detect_volatility_breakout(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 20,
        expansion_threshold: float = 1.5
    ) -> Optional[BreakoutSignal]:
        """
        Detect volatility expansion breakouts

        When volatility expands significantly after contraction,
        it often signals the start of a new trend
        """
        if len(ohlcv) < lookback * 2:
            return None

        atr = self.calculate_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])

        current_atr = atr.iloc[-1]
        recent_atr = atr.tail(lookback).mean()
        prior_atr = atr.iloc[-lookback*2:-lookback].mean()

        # Check for volatility expansion
        if current_atr > recent_atr * expansion_threshold:
            # Determine direction
            price_change = ohlcv["close"].iloc[-1] - ohlcv["close"].iloc[-lookback]
            direction = 1 if price_change > 0 else -1

            expansion_ratio = current_atr / recent_atr
            strength = min(1.0, (expansion_ratio - 1) / 2)

            return BreakoutSignal(
                breakout_type=BreakoutType.VOLATILITY_EXPANSION,
                direction=direction,
                strength=strength,
                price_level=ohlcv["close"].iloc[-1],
                volume_confirmation=ohlcv["volume"].iloc[-1] > ohlcv["volume"].tail(lookback).mean()
            )

        return None

    def detect_squeeze_breakout(
        self,
        ohlcv: pd.DataFrame,
        bb_period: int = 20,
        kc_period: int = 20,
        kc_mult: float = 1.5
    ) -> Optional[BreakoutSignal]:
        """
        Detect TTM Squeeze breakout

        Squeeze occurs when Bollinger Bands are inside Keltner Channels
        Breakout from squeeze often leads to strong moves
        """
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]

        # Bollinger Bands
        bb_sma = close.rolling(window=bb_period).mean()
        bb_std = close.rolling(window=bb_period).std()
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)

        # Keltner Channels
        kc_ma = close.rolling(window=kc_period).mean()
        atr = self.calculate_atr(high, low, close, kc_period)
        kc_upper = kc_ma + (atr * kc_mult)
        kc_lower = kc_ma - (atr * kc_mult)

        # Squeeze detection (BB inside KC)
        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)

        # Check for squeeze release
        if len(squeeze) < 2:
            return None

        was_in_squeeze = squeeze.iloc[-2]
        now_released = not squeeze.iloc[-1]

        if was_in_squeeze and now_released:
            # Momentum direction
            momentum = close.iloc[-1] - close.iloc[-bb_period]
            direction = 1 if momentum > 0 else -1

            # Strength from momentum magnitude
            strength = min(1.0, abs(momentum) / close.iloc[-1] * 20)

            return BreakoutSignal(
                breakout_type=BreakoutType.RANGE_EXPANSION,
                direction=direction,
                strength=strength,
                price_level=close.iloc[-1],
                volume_confirmation=ohlcv["volume"].iloc[-1] > ohlcv["volume"].tail(20).mean()
            )

        return None

    def get_all_breakouts(
        self,
        ohlcv: pd.DataFrame
    ) -> list[BreakoutSignal]:
        """
        Run all breakout detection methods

        Returns list of all detected breakouts
        """
        breakouts = []

        detectors = [
            lambda: self.detect_price_breakout(ohlcv),
            lambda: self.detect_volume_breakout(ohlcv),
            lambda: self.detect_volatility_breakout(ohlcv),
            lambda: self.detect_squeeze_breakout(ohlcv)
        ]

        for detector in detectors:
            try:
                signal = detector()
                if signal is not None:
                    breakouts.append(signal)
            except Exception as e:
                print(f"Breakout detection error: {e}")

        return breakouts

    def score_breakout_potential(
        self,
        ohlcv: pd.DataFrame
    ) -> dict:
        """
        Score the breakout potential of a stock

        High scores indicate stock is likely to break out soon
        """
        # Volatility contraction (precedes breakouts)
        atr = self.calculate_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        current_atr = atr.iloc[-1]
        historical_atr = atr.mean()
        volatility_ratio = current_atr / historical_atr

        # Price near key levels
        sr_levels = self.find_support_resistance(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        current = ohlcv["close"].iloc[-1]

        distance_to_resistance = (
            (sr_levels["nearest_resistance"] - current) / current
            if sr_levels["nearest_resistance"] else 1.0
        )
        distance_to_support = (
            (current - sr_levels["nearest_support"]) / current
            if sr_levels["nearest_support"] else 1.0
        )

        # Volume trend (accumulation before breakout)
        volume_sma = ohlcv["volume"].rolling(20).mean()
        volume_trend = volume_sma.iloc[-1] / volume_sma.iloc[-20] - 1

        # Consolidation pattern (price range narrowing)
        recent_range = (ohlcv["high"].tail(10).max() - ohlcv["low"].tail(10).min()) / current
        prior_range = (ohlcv["high"].tail(30).head(20).max() - ohlcv["low"].tail(30).head(20).min()) / current
        range_contraction = 1 - (recent_range / prior_range) if prior_range > 0 else 0

        # Composite score
        score = 0
        score += max(0, 1 - volatility_ratio) * 25  # Low vol = higher score
        score += max(0, 1 - distance_to_resistance * 10) * 25  # Close to resistance
        score += max(0, volume_trend) * 25  # Increasing volume
        score += max(0, range_contraction) * 25  # Consolidating

        return {
            "breakout_potential_score": score,
            "volatility_ratio": volatility_ratio,
            "distance_to_resistance": distance_to_resistance,
            "distance_to_support": distance_to_support,
            "volume_trend": volume_trend,
            "range_contraction": range_contraction,
            "support_resistance": sr_levels
        }
