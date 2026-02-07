"""Tests for signal classification and position sizing logic."""

import pytest
import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from analyze_market import classify_signal, calculate_position_size, SignalRating


class TestClassifySignal:
    """Test the multi-factor signal classifier."""

    def test_strong_buy_all_aligned(self):
        """Strong trend + strong momentum + good alpha + beating SPY = STRONG_BUY."""
        rating, conf, reason = classify_signal(
            trend_value=2,       # STRONG_UP
            momentum_score=0.6,  # strong
            alpha_percentile=80, # strong
            rsi=55,              # healthy
            relative_strength=0.10,  # beating SPY
            sector_rank=1,
            sector_size=10,
        )
        assert rating == SignalRating.STRONG_BUY
        assert conf >= 0.7
        assert "All signals aligned" in reason

    def test_buy_positive_trend_with_alpha(self):
        """Positive trend + good alpha (but modest momentum) = BUY."""
        rating, conf, reason = classify_signal(
            trend_value=1,       # UP
            momentum_score=0.1,  # not strong
            alpha_percentile=75, # good
            rsi=55,
            relative_strength=0.05,
            sector_rank=2,
            sector_size=10,
        )
        assert rating == SignalRating.BUY
        assert "alpha" in reason.lower()

    def test_buy_positive_trend_with_momentum(self):
        """Positive trend + good momentum (but weak alpha) = BUY."""
        rating, conf, reason = classify_signal(
            trend_value=1,
            momentum_score=0.5,
            alpha_percentile=40,  # not good
            rsi=60,
            relative_strength=-0.02,
            sector_rank=5,
            sector_size=10,
        )
        assert rating == SignalRating.BUY
        assert "momentum" in reason.lower()

    def test_hold_overbought_blocks_buy(self):
        """Overbought RSI should prevent BUY/STRONG_BUY -> HOLD_OB."""
        rating, conf, reason = classify_signal(
            trend_value=2,
            momentum_score=0.8,
            alpha_percentile=90,
            rsi=88,              # severely overbought
            relative_strength=0.20,
            sector_rank=1,
            sector_size=10,
        )
        assert rating == SignalRating.HOLD_OVERBOUGHT
        assert "overbought" in reason.lower()

    def test_hold_overbought_reduces_confidence(self):
        """Overbought stocks should have halved confidence."""
        _, conf_ob, _ = classify_signal(
            trend_value=2, momentum_score=0.6, alpha_percentile=80,
            rsi=80, relative_strength=0.10, sector_rank=1, sector_size=10,
        )
        _, conf_normal, _ = classify_signal(
            trend_value=2, momentum_score=0.6, alpha_percentile=80,
            rsi=60, relative_strength=0.10, sector_rank=1, sector_size=10,
        )
        assert conf_ob < conf_normal

    def test_avoid_weak_trend_low_alpha(self):
        """No trend + low alpha = AVOID."""
        rating, conf, reason = classify_signal(
            trend_value=0,       # NEUTRAL
            momentum_score=-0.1,
            alpha_percentile=20, # bottom quartile
            rsi=45,
            relative_strength=-0.15,
            sector_rank=8,
            sector_size=10,
        )
        assert rating == SignalRating.AVOID
        assert "weak" in reason.lower()

    def test_hold_mixed_signals(self):
        """Neutral trend + decent alpha = HOLD."""
        rating, conf, reason = classify_signal(
            trend_value=0,
            momentum_score=0.1,
            alpha_percentile=55,  # above 40, so not AVOID
            rsi=50,
            relative_strength=0.0,
            sector_rank=5,
            sector_size=10,
        )
        assert rating == SignalRating.HOLD
        assert "mixed" in reason.lower()

    def test_downtrend_with_good_alpha_is_not_buy(self):
        """Even with good alpha, downtrend should not produce BUY."""
        rating, _, _ = classify_signal(
            trend_value=-1,      # DOWN
            momentum_score=-0.2,
            alpha_percentile=80,
            rsi=40,
            relative_strength=-0.10,
            sector_rank=1,
            sector_size=10,
        )
        assert rating not in [SignalRating.BUY, SignalRating.STRONG_BUY]

    def test_rsi_at_boundary(self):
        """RSI exactly at 75 should be overbought."""
        rating, _, _ = classify_signal(
            trend_value=1, momentum_score=0.5, alpha_percentile=70,
            rsi=75, relative_strength=0.05, sector_rank=2, sector_size=10,
        )
        assert rating == SignalRating.HOLD_OVERBOUGHT

    def test_rsi_just_below_boundary(self):
        """RSI at 74 should allow BUY."""
        rating, _, _ = classify_signal(
            trend_value=1, momentum_score=0.5, alpha_percentile=70,
            rsi=74, relative_strength=0.05, sector_rank=2, sector_size=10,
        )
        assert rating in [SignalRating.BUY, SignalRating.STRONG_BUY]


class TestPositionSizing:
    """Test inverse-volatility position sizing."""

    def test_equal_volatility_equal_weight(self):
        """Same volatility as median -> approximately base weight."""
        size = calculate_position_size(
            stock_volatility=0.20,
            median_volatility=0.20,
            confidence=1.0,
            n_positions=10,
        )
        assert abs(size - 0.10) < 0.01  # 1/10 * 1.0 * 1.0

    def test_low_vol_gets_larger_size(self):
        """Low vol stock should get a larger allocation."""
        low_vol = calculate_position_size(
            stock_volatility=0.10,
            median_volatility=0.20,
            confidence=0.8,
            n_positions=10,
        )
        high_vol = calculate_position_size(
            stock_volatility=0.40,
            median_volatility=0.20,
            confidence=0.8,
            n_positions=10,
        )
        assert low_vol > high_vol

    def test_high_confidence_gets_larger_size(self):
        """Higher confidence should increase allocation."""
        high_conf = calculate_position_size(
            stock_volatility=0.20, median_volatility=0.20,
            confidence=1.0, n_positions=10,
        )
        low_conf = calculate_position_size(
            stock_volatility=0.20, median_volatility=0.20,
            confidence=0.0, n_positions=10,
        )
        assert high_conf > low_conf

    def test_max_size_cap(self):
        """Position size should never exceed max_size."""
        size = calculate_position_size(
            stock_volatility=0.01,  # very low vol -> high weight
            median_volatility=0.50,
            confidence=1.0,
            n_positions=2,
            max_size=0.10,
        )
        assert size <= 0.10

    def test_zero_volatility_handled(self):
        """Zero volatility should not crash."""
        size = calculate_position_size(
            stock_volatility=0.0,
            median_volatility=0.20,
            confidence=0.8,
            n_positions=10,
        )
        assert size > 0

    def test_single_position(self):
        """With 1 position, base size is 100%."""
        size = calculate_position_size(
            stock_volatility=0.20,
            median_volatility=0.20,
            confidence=1.0,
            n_positions=1,
            max_size=0.10,
        )
        assert size == 0.10  # capped at max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
