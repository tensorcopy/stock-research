"""Tests for alpha factors"""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.alpha.factors import AlphaFactors


class TestAlphaFactors:
    """Test cases for AlphaFactors class"""

    @pytest.fixture
    def alpha_factors(self):
        return AlphaFactors()

    @pytest.fixture
    def sample_prices(self):
        """Generate 1 year of sample prices"""
        dates = pd.date_range(start="2023-01-01", periods=260, freq="D")
        # Random walk with slight upward drift
        returns = np.random.randn(260) * 0.02 + 0.0003
        prices = 100 * (1 + returns).cumprod()
        return pd.Series(prices, index=dates)

    def test_momentum_12_1(self, alpha_factors, sample_prices):
        """Test 12-1 month momentum factor"""
        factor = alpha_factors.momentum_12_1(sample_prices)

        assert factor.name == "momentum_12_1"
        assert not np.isnan(factor.value)

    def test_momentum_insufficient_data(self, alpha_factors):
        """Test momentum with insufficient data"""
        short_prices = pd.Series([100, 101, 102])
        factor = alpha_factors.momentum_12_1(short_prices)

        assert np.isnan(factor.value)

    def test_book_to_market(self, alpha_factors):
        """Test book-to-market factor"""
        factor = alpha_factors.book_to_market(
            book_value=50_000_000,
            market_cap=100_000_000
        )

        assert factor.name == "book_to_market"
        assert factor.value == 0.5

    def test_book_to_market_zero_market_cap(self, alpha_factors):
        """Test B/M with zero market cap"""
        factor = alpha_factors.book_to_market(
            book_value=50_000_000,
            market_cap=0
        )

        assert np.isnan(factor.value)

    def test_earnings_yield(self, alpha_factors):
        """Test earnings yield factor"""
        factor = alpha_factors.earnings_yield(
            earnings=10_000_000,
            market_cap=200_000_000
        )

        assert factor.name == "earnings_yield"
        assert factor.value == 0.05  # 5% earnings yield

    def test_return_on_equity(self, alpha_factors):
        """Test ROE factor"""
        factor = alpha_factors.return_on_equity(
            net_income=20_000_000,
            equity=100_000_000
        )

        assert factor.name == "roe"
        assert factor.value == 0.2  # 20% ROE

    def test_gross_profitability(self, alpha_factors):
        """Test gross profitability (Novy-Marx) factor"""
        factor = alpha_factors.gross_profitability(
            gross_profit=30_000_000,
            total_assets=100_000_000
        )

        assert factor.name == "gross_profitability"
        assert factor.value == 0.3  # 30% GP/A

    def test_market_cap_factor(self, alpha_factors):
        """Test size factor"""
        small_cap = alpha_factors.market_cap_factor(1_000_000_000)
        large_cap = alpha_factors.market_cap_factor(100_000_000_000)

        # Small cap should have higher (less negative) score
        assert small_cap.value > large_cap.value

    def test_calculate_all_factors(self, alpha_factors, sample_prices):
        """Test calculating all available factors"""
        fundamentals = {
            "book_value": 50_000_000,
            "market_cap": 100_000_000,
            "earnings": 10_000_000,
            "net_income": 10_000_000,
            "equity": 50_000_000,
            "gross_profit": 30_000_000,
            "total_assets": 100_000_000
        }

        factors = alpha_factors.calculate_all_factors(sample_prices, fundamentals)

        assert "momentum_12_1" in factors
        assert "book_to_market" in factors
        assert "earnings_yield" in factors
        assert "roe" in factors
        assert "gross_profitability" in factors
        assert "size" in factors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
