"""
Trend Detection Module
Identify market trends using technical and statistical methods
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


class TrendDirection(Enum):
    """Trend direction classification"""
    STRONG_UP = 2
    UP = 1
    NEUTRAL = 0
    DOWN = -1
    STRONG_DOWN = -2


@dataclass
class TrendSignal:
    """Container for trend signal data"""
    direction: TrendDirection
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    start_date: Optional[pd.Timestamp] = None
    duration_days: int = 0


class TrendDetector:
    """
    Detect and analyze price trends for alpha generation

    Methods implemented:
    1. Moving Average Crossovers (Golden/Death Cross)
    2. ADX (Average Directional Index)
    3. Linear Regression Channel
    4. Higher Highs/Higher Lows Pattern
    5. Ichimoku Cloud
    """

    def __init__(self):
        self.signals_history: list[TrendSignal] = []

    def detect_ma_crossover(
        self,
        prices: pd.Series,
        fast_period: int = 50,
        slow_period: int = 200
    ) -> TrendSignal:
        """
        Detect Golden Cross (bullish) or Death Cross (bearish)

        Golden Cross: 50-day MA crosses above 200-day MA
        Death Cross: 50-day MA crosses below 200-day MA
        """
        if len(prices) < slow_period:
            return TrendSignal(TrendDirection.NEUTRAL, 0, 0)

        fast_ma = prices.rolling(window=fast_period).mean()
        slow_ma = prices.rolling(window=slow_period).mean()

        # Current state
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]

        # Previous state (for crossover detection)
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]

        # Determine trend
        if current_fast > current_slow:
            direction = TrendDirection.UP
            # Check for recent crossover (stronger signal)
            if prev_fast <= prev_slow:
                direction = TrendDirection.STRONG_UP
        elif current_fast < current_slow:
            direction = TrendDirection.DOWN
            if prev_fast >= prev_slow:
                direction = TrendDirection.STRONG_DOWN
        else:
            direction = TrendDirection.NEUTRAL

        # Calculate strength based on MA separation
        ma_diff_pct = (current_fast - current_slow) / current_slow
        strength = min(1.0, abs(ma_diff_pct) * 10)  # Scale: 10% diff = max strength

        # Confidence based on trend consistency
        recent_prices = prices.tail(20)
        above_fast = (recent_prices > fast_ma.tail(20)).mean()
        confidence = abs(above_fast - 0.5) * 2  # Convert to 0-1 scale

        return TrendSignal(direction, strength, confidence)

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate ADX (Average Directional Index)

        ADX measures trend strength (not direction):
        - 0-25: Weak/No trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return pd.DataFrame({
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "trend_strength": adx / 100  # Normalized 0-1
        })

    def detect_trend_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        threshold: float = 25
    ) -> TrendSignal:
        """
        Detect trend using ADX indicator

        Args:
            threshold: ADX value to consider a trend exists (default 25)
        """
        adx_data = self.calculate_adx(high, low, close, period)

        current_adx = adx_data["adx"].iloc[-1]
        current_plus_di = adx_data["plus_di"].iloc[-1]
        current_minus_di = adx_data["minus_di"].iloc[-1]

        # Determine direction
        if current_adx < threshold:
            direction = TrendDirection.NEUTRAL
        elif current_plus_di > current_minus_di:
            direction = TrendDirection.STRONG_UP if current_adx > 50 else TrendDirection.UP
        else:
            direction = TrendDirection.STRONG_DOWN if current_adx > 50 else TrendDirection.DOWN

        # Strength from ADX
        strength = min(1.0, current_adx / 75)

        # Confidence from DI spread
        di_spread = abs(current_plus_di - current_minus_di)
        confidence = min(1.0, di_spread / 40)

        return TrendSignal(direction, strength, confidence)

    def detect_linear_regression_trend(
        self,
        prices: pd.Series,
        period: int = 20
    ) -> TrendSignal:
        """
        Detect trend using linear regression slope

        Measures the rate of price change over the period
        """
        if len(prices) < period:
            return TrendSignal(TrendDirection.NEUTRAL, 0, 0)

        recent = prices.tail(period)
        x = np.arange(period)
        y = recent.values

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Normalize slope by price level (percentage change per day)
        avg_price = y.mean()
        slope_pct = (slope / avg_price) * 100

        # Determine direction
        if slope_pct > 0.5:
            direction = TrendDirection.STRONG_UP
        elif slope_pct > 0.1:
            direction = TrendDirection.UP
        elif slope_pct < -0.5:
            direction = TrendDirection.STRONG_DOWN
        elif slope_pct < -0.1:
            direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.NEUTRAL

        # Strength from slope magnitude
        strength = min(1.0, abs(slope_pct) / 2)

        # Confidence from R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        confidence = max(0, r_squared)

        return TrendSignal(direction, strength, confidence)

    def detect_higher_highs_lows(
        self,
        high: pd.Series,
        low: pd.Series,
        lookback: int = 20,
        swing_threshold: int = 5
    ) -> TrendSignal:
        """
        Detect trend by analyzing swing highs and lows pattern

        Uptrend: Higher highs and higher lows
        Downtrend: Lower highs and lower lows
        """
        if len(high) < lookback:
            return TrendSignal(TrendDirection.NEUTRAL, 0, 0)

        # Find swing highs and lows
        swing_highs = []
        swing_lows = []

        for i in range(swing_threshold, len(high) - swing_threshold):
            # Swing high: higher than surrounding points
            if high.iloc[i] == high.iloc[i-swing_threshold:i+swing_threshold+1].max():
                swing_highs.append((i, high.iloc[i]))
            # Swing low: lower than surrounding points
            if low.iloc[i] == low.iloc[i-swing_threshold:i+swing_threshold+1].min():
                swing_lows.append((i, low.iloc[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return TrendSignal(TrendDirection.NEUTRAL, 0, 0)

        # Analyze recent swings
        recent_highs = swing_highs[-3:]
        recent_lows = swing_lows[-3:]

        # Count higher highs/lows vs lower highs/lows
        hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i][1] > recent_highs[i-1][1])
        hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i][1] > recent_lows[i-1][1])
        lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i][1] < recent_highs[i-1][1])
        ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i][1] < recent_lows[i-1][1])

        # Determine trend
        uptrend_score = hh_count + hl_count
        downtrend_score = lh_count + ll_count

        if uptrend_score > downtrend_score + 1:
            direction = TrendDirection.STRONG_UP if uptrend_score >= 3 else TrendDirection.UP
        elif downtrend_score > uptrend_score + 1:
            direction = TrendDirection.STRONG_DOWN if downtrend_score >= 3 else TrendDirection.DOWN
        else:
            direction = TrendDirection.NEUTRAL

        total_swings = max(1, hh_count + hl_count + lh_count + ll_count)
        strength = abs(uptrend_score - downtrend_score) / total_swings
        confidence = min(1.0, len(swing_highs + swing_lows) / 10)

        return TrendSignal(direction, strength, confidence)

    def get_composite_trend(
        self,
        ohlcv: pd.DataFrame,
        weights: Optional[dict] = None
    ) -> TrendSignal:
        """
        Combine multiple trend detection methods for robust signal

        Args:
            ohlcv: DataFrame with open, high, low, close, volume columns
            weights: Custom weights for each method
        """
        default_weights = {
            "ma_crossover": 0.3,
            "adx": 0.25,
            "linear_reg": 0.25,
            "swing": 0.2
        }
        weights = weights or default_weights

        signals = {}

        # MA Crossover
        signals["ma_crossover"] = self.detect_ma_crossover(ohlcv["close"])

        # ADX
        signals["adx"] = self.detect_trend_adx(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )

        # Linear Regression
        signals["linear_reg"] = self.detect_linear_regression_trend(ohlcv["close"])

        # Swing Analysis
        signals["swing"] = self.detect_higher_highs_lows(ohlcv["high"], ohlcv["low"])

        # Combine signals
        weighted_direction = sum(
            signals[k].direction.value * weights[k] * signals[k].confidence
            for k in weights
        )
        weighted_strength = sum(
            signals[k].strength * weights[k]
            for k in weights
        )
        weighted_confidence = sum(
            signals[k].confidence * weights[k]
            for k in weights
        )

        # Map back to direction
        if weighted_direction > 1.0:
            direction = TrendDirection.STRONG_UP
        elif weighted_direction > 0.3:
            direction = TrendDirection.UP
        elif weighted_direction < -1.0:
            direction = TrendDirection.STRONG_DOWN
        elif weighted_direction < -0.3:
            direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.NEUTRAL

        return TrendSignal(
            direction=direction,
            strength=weighted_strength,
            confidence=weighted_confidence
        )
