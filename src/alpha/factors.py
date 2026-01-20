"""
Alpha Factor Library
A collection of proven alpha factors for stock selection
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class AlphaFactor:
    """Container for an alpha factor"""
    name: str
    value: float
    z_score: Optional[float] = None  # Standardized score
    percentile: Optional[float] = None


class AlphaFactors:
    """
    Library of alpha factors for stock selection

    Categories of alpha factors:
    1. Value - cheap stocks outperform expensive
    2. Momentum - winners keep winning
    3. Quality - profitable, stable companies outperform
    4. Size - small caps have higher returns
    5. Volatility - low vol stocks often outperform
    6. Growth - high growth companies (when not overpriced)

    Each factor should be:
    - Persistent across time
    - Pervasive across markets
    - Robust to different definitions
    - Investable at scale
    - Intuitive (has economic rationale)
    """

    def __init__(self):
        pass

    # ==================== MOMENTUM FACTORS ====================

    def momentum_12_1(self, prices: pd.Series) -> AlphaFactor:
        """
        Classic momentum: 12-month return skipping last month

        The most documented momentum factor. Skips last month to
        avoid short-term reversal effect.
        """
        if len(prices) < 252:
            return AlphaFactor("momentum_12_1", np.nan)

        ret = (prices.iloc[-21] / prices.iloc[-252]) - 1
        return AlphaFactor("momentum_12_1", ret)

    def momentum_6_1(self, prices: pd.Series) -> AlphaFactor:
        """6-month momentum skipping last month"""
        if len(prices) < 126:
            return AlphaFactor("momentum_6_1", np.nan)

        ret = (prices.iloc[-21] / prices.iloc[-126]) - 1
        return AlphaFactor("momentum_6_1", ret)

    def momentum_1m(self, prices: pd.Series) -> AlphaFactor:
        """
        Short-term reversal: 1-month return

        Usually negative predictor (mean reversion)
        """
        if len(prices) < 21:
            return AlphaFactor("momentum_1m", np.nan)

        ret = (prices.iloc[-1] / prices.iloc[-21]) - 1
        return AlphaFactor("momentum_1m", ret)

    def earnings_momentum(self, earnings_surprise: float) -> AlphaFactor:
        """
        Post-earnings announcement drift

        Stocks with positive surprises continue to outperform
        """
        return AlphaFactor("earnings_momentum", earnings_surprise)

    # ==================== VALUE FACTORS ====================

    def book_to_market(self, book_value: float, market_cap: float) -> AlphaFactor:
        """
        Book-to-Market ratio (inverse of P/B)

        Higher B/M = cheaper stock (value)
        """
        if market_cap <= 0:
            return AlphaFactor("book_to_market", np.nan)

        btm = book_value / market_cap
        return AlphaFactor("book_to_market", btm)

    def earnings_yield(self, earnings: float, market_cap: float) -> AlphaFactor:
        """
        Earnings Yield (E/P, inverse of P/E)

        Higher = cheaper stock
        """
        if market_cap <= 0:
            return AlphaFactor("earnings_yield", np.nan)

        ey = earnings / market_cap
        return AlphaFactor("earnings_yield", ey)

    def fcf_yield(self, fcf: float, enterprise_value: float) -> AlphaFactor:
        """
        Free Cash Flow Yield

        Often better than earnings yield as FCF is harder to manipulate
        """
        if enterprise_value <= 0:
            return AlphaFactor("fcf_yield", np.nan)

        fcf_y = fcf / enterprise_value
        return AlphaFactor("fcf_yield", fcf_y)

    def ev_ebitda(self, ev: float, ebitda: float) -> AlphaFactor:
        """
        EV/EBITDA ratio (lower = cheaper)

        Returns negative for use in ranking (lower is better)
        """
        if ebitda <= 0:
            return AlphaFactor("ev_ebitda", np.nan)

        ratio = ev / ebitda
        return AlphaFactor("ev_ebitda", -ratio)  # Negative for ranking

    # ==================== QUALITY FACTORS ====================

    def return_on_equity(self, net_income: float, equity: float) -> AlphaFactor:
        """
        Return on Equity

        Higher ROE = more efficient use of shareholder capital
        """
        if equity <= 0:
            return AlphaFactor("roe", np.nan)

        roe = net_income / equity
        return AlphaFactor("roe", roe)

    def return_on_assets(self, net_income: float, total_assets: float) -> AlphaFactor:
        """Return on Assets"""
        if total_assets <= 0:
            return AlphaFactor("roa", np.nan)

        roa = net_income / total_assets
        return AlphaFactor("roa", roa)

    def gross_profitability(self, gross_profit: float, total_assets: float) -> AlphaFactor:
        """
        Gross Profitability (Novy-Marx factor)

        Gross profits / Assets - robust quality factor
        """
        if total_assets <= 0:
            return AlphaFactor("gross_profitability", np.nan)

        gp = gross_profit / total_assets
        return AlphaFactor("gross_profitability", gp)

    def accruals(
        self,
        net_income: float,
        operating_cash_flow: float,
        total_assets: float
    ) -> AlphaFactor:
        """
        Accruals (earnings quality)

        Low accruals = higher earnings quality
        Accruals = (Net Income - OCF) / Assets
        """
        if total_assets <= 0:
            return AlphaFactor("accruals", np.nan)

        accrual = (net_income - operating_cash_flow) / total_assets
        return AlphaFactor("accruals", -accrual)  # Negative = low accruals is good

    def debt_to_equity(self, total_debt: float, equity: float) -> AlphaFactor:
        """
        Debt to Equity (leverage)

        Lower leverage often associated with better risk-adjusted returns
        """
        if equity <= 0:
            return AlphaFactor("debt_to_equity", np.nan)

        de = total_debt / equity
        return AlphaFactor("debt_to_equity", -de)  # Negative = low debt is good

    # ==================== SIZE FACTORS ====================

    def market_cap_factor(self, market_cap: float) -> AlphaFactor:
        """
        Size factor (log market cap)

        Smaller stocks have historically outperformed (size premium)
        Returns negative so smaller = higher score
        """
        if market_cap <= 0:
            return AlphaFactor("size", np.nan)

        log_cap = np.log(market_cap)
        return AlphaFactor("size", -log_cap)  # Negative = small is better

    # ==================== VOLATILITY FACTORS ====================

    def idiosyncratic_volatility(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        period: int = 60
    ) -> AlphaFactor:
        """
        Idiosyncratic volatility (IVOL)

        Low IVOL stocks tend to outperform
        """
        if len(returns) < period or len(market_returns) < period:
            return AlphaFactor("ivol", np.nan)

        # Simple regression to get residuals
        returns_recent = returns.tail(period)
        market_recent = market_returns.tail(period)

        # Calculate beta
        cov = returns_recent.cov(market_recent)
        var = market_recent.var()
        beta = cov / var if var != 0 else 1

        # Calculate residuals
        residuals = returns_recent - beta * market_recent

        ivol = residuals.std()
        return AlphaFactor("ivol", -ivol)  # Negative = low vol is good

    def beta(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        period: int = 252
    ) -> AlphaFactor:
        """
        Market beta

        Low beta stocks often have higher risk-adjusted returns
        """
        if len(returns) < period or len(market_returns) < period:
            return AlphaFactor("beta", np.nan)

        cov = returns.tail(period).cov(market_returns.tail(period))
        var = market_returns.tail(period).var()

        beta_val = cov / var if var != 0 else 1
        return AlphaFactor("beta", -beta_val)  # Negative = low beta preferred

    # ==================== GROWTH FACTORS ====================

    def revenue_growth(
        self,
        current_revenue: float,
        prior_revenue: float
    ) -> AlphaFactor:
        """Year-over-year revenue growth"""
        if prior_revenue <= 0:
            return AlphaFactor("revenue_growth", np.nan)

        growth = (current_revenue / prior_revenue) - 1
        return AlphaFactor("revenue_growth", growth)

    def earnings_growth(
        self,
        current_earnings: float,
        prior_earnings: float
    ) -> AlphaFactor:
        """Year-over-year earnings growth"""
        if prior_earnings <= 0:
            return AlphaFactor("earnings_growth", np.nan)

        growth = (current_earnings / prior_earnings) - 1
        return AlphaFactor("earnings_growth", growth)

    def asset_growth(
        self,
        current_assets: float,
        prior_assets: float
    ) -> AlphaFactor:
        """
        Asset growth (negative predictor)

        High asset growth often followed by lower returns
        """
        if prior_assets <= 0:
            return AlphaFactor("asset_growth", np.nan)

        growth = (current_assets / prior_assets) - 1
        return AlphaFactor("asset_growth", -growth)  # Negative = low growth better

    # ==================== COMBINED FACTORS ====================

    def calculate_all_factors(
        self,
        prices: pd.Series,
        fundamentals: dict,
        market_data: Optional[dict] = None
    ) -> dict[str, AlphaFactor]:
        """
        Calculate all available factors for a stock

        Args:
            prices: Price series
            fundamentals: Dict with fundamental data
            market_data: Optional market benchmark data

        Returns:
            Dictionary of factor name to AlphaFactor
        """
        factors = {}

        # Momentum factors
        factors["momentum_12_1"] = self.momentum_12_1(prices)
        factors["momentum_6_1"] = self.momentum_6_1(prices)
        factors["momentum_1m"] = self.momentum_1m(prices)

        # Value factors
        if "book_value" in fundamentals and "market_cap" in fundamentals:
            factors["book_to_market"] = self.book_to_market(
                fundamentals["book_value"],
                fundamentals["market_cap"]
            )

        if "earnings" in fundamentals and "market_cap" in fundamentals:
            factors["earnings_yield"] = self.earnings_yield(
                fundamentals["earnings"],
                fundamentals["market_cap"]
            )

        if "fcf" in fundamentals and "enterprise_value" in fundamentals:
            factors["fcf_yield"] = self.fcf_yield(
                fundamentals["fcf"],
                fundamentals["enterprise_value"]
            )

        # Quality factors
        if "net_income" in fundamentals and "equity" in fundamentals:
            factors["roe"] = self.return_on_equity(
                fundamentals["net_income"],
                fundamentals["equity"]
            )

        if "gross_profit" in fundamentals and "total_assets" in fundamentals:
            factors["gross_profitability"] = self.gross_profitability(
                fundamentals["gross_profit"],
                fundamentals["total_assets"]
            )

        # Size factor
        if "market_cap" in fundamentals:
            factors["size"] = self.market_cap_factor(fundamentals["market_cap"])

        return factors
