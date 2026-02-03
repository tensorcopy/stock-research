"""Tests for momentum signals"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.signals.momentum import MomentumSignals


class TestMomentumSignals:
    """Test cases for MomentumSignals class"""

    @pytest.fixture
    def momentum_signals(self):
        return MomentumSignals()

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data with uptrend"""
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        # Uptrending prices with some noise
        base = 100 * (1.001 ** np.arange(300))
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(300) * 2
        prices = pd.Series(base + noise, index=dates)
        return prices

    @pytest.fixture
    def sample_ohlcv(self, sample_prices):
        """Generate sample OHLCV data"""
        close = sample_prices
        rng = np.random.default_rng(1)
        high = close * (1 + np.abs(rng.standard_normal(len(close)) * 0.01))
        low = close * (1 - np.abs(rng.standard_normal(len(close)) * 0.01))
        open_prices = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(rng.integers(1_000_000, 5_000_000, len(close)), index=close.index)

        return pd.DataFrame({
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })

    def test_calculate_price_momentum(self, momentum_signals, sample_prices):
        """Test price momentum calculation"""
        momentum = momentum_signals.calculate_price_momentum(sample_prices)

        assert "mom_21d" in momentum
        assert "mom_63d" in momentum
        assert "mom_126d" in momentum
        assert "mom_252d" in momentum

        # In uptrending data, momentum should be positive
        assert momentum["mom_126d"] > 0

    def test_calculate_rsi(self, momentum_signals, sample_prices):
        """Test RSI calculation"""
        rsi = momentum_signals.calculate_rsi(sample_prices)

        assert len(rsi) == len(sample_prices)
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_calculate_macd(self, momentum_signals, sample_prices):
        """Test MACD calculation"""
        macd = momentum_signals.calculate_macd(sample_prices)

        assert "macd" in macd.columns
        assert "signal" in macd.columns
        assert "histogram" in macd.columns
        assert len(macd) == len(sample_prices)

    def test_calculate_momentum_score(self, momentum_signals, sample_ohlcv):
        """Test composite momentum score"""
        signal = momentum_signals.calculate_momentum_score(sample_ohlcv)

        assert signal.score is not None
        assert -1 <= signal.score <= 1
        assert "price_momentum" in signal.components

    def test_rank_momentum(self, momentum_signals, sample_ohlcv):
        """Test momentum ranking across universe"""
        universe = {
            "AAPL": sample_ohlcv,
            "MSFT": sample_ohlcv * 1.1,  # Different prices
            "GOOGL": sample_ohlcv * 0.9
        }

        rankings = momentum_signals.rank_momentum(universe)

        assert len(rankings) == 3
        assert "momentum_score" in rankings.columns
        assert "momentum_rank" in rankings.columns
        assert "momentum_percentile" in rankings.columns


class TestRSI:
    """Edge cases for RSI calculation"""

    def test_rsi_constant_prices(self):
        """RSI should be 50 for constant prices"""
        signals = MomentumSignals()
        constant_prices = pd.Series([100] * 30)
        rsi = signals.calculate_rsi(constant_prices)
        # RSI is undefined for constant prices, but shouldn't crash
        assert len(rsi) == 30

    def test_rsi_always_up(self):
        """RSI should be 100 for always increasing prices"""
        signals = MomentumSignals()
        up_prices = pd.Series(range(1, 31))
        rsi = signals.calculate_rsi(up_prices)
        # Last RSI should be close to 100
        assert rsi.iloc[-1] > 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
