"""
Fundamentals Data Fetcher
Collects financial statements, ratios, and fundamental data
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class FundamentalsConfig:
    """Configuration for fundamentals data"""
    source: str = "yfinance"
    cache_dir: str = "./cache/fundamentals"


class FundamentalsFetcher:
    """
    Fetch fundamental data for stock analysis

    Key metrics for trend/alpha research:
    - Revenue/earnings growth (momentum in fundamentals)
    - Margin expansion/contraction
    - ROE, ROIC trends
    - Earnings surprise history
    - Analyst estimate revisions
    """

    def __init__(self, config: Optional[FundamentalsConfig] = None):
        self.config = config or FundamentalsConfig()
        self._cache: dict = {}

    def get_financial_statements(
        self,
        symbol: str,
        statement_type: str = "all"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch financial statements

        Args:
            symbol: Ticker symbol
            statement_type: 'income', 'balance', 'cashflow', or 'all'

        Returns:
            Dictionary with statement DataFrames
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            statements = {}

            if statement_type in ["all", "income"]:
                statements["income"] = ticker.financials.T
                statements["income_quarterly"] = ticker.quarterly_financials.T

            if statement_type in ["all", "balance"]:
                statements["balance"] = ticker.balance_sheet.T
                statements["balance_quarterly"] = ticker.quarterly_balance_sheet.T

            if statement_type in ["all", "cashflow"]:
                statements["cashflow"] = ticker.cashflow.T
                statements["cashflow_quarterly"] = ticker.quarterly_cashflow.T

            return statements
        except Exception as e:
            print(f"Error fetching financials for {symbol}: {e}")
            return {}

    def get_key_metrics(self, symbols: list[str]) -> pd.DataFrame:
        """
        Get key fundamental metrics for multiple symbols

        Metrics important for alpha:
        - P/E, P/S, P/B ratios (value factors)
        - Revenue growth, EPS growth (growth factors)
        - Profit margins
        - ROE, ROIC
        - Debt ratios
        """
        metrics_list = []

        try:
            import yfinance as yf

            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                metrics = {
                    "symbol": symbol,
                    # Classification
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    # Valuation
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "ps_ratio": info.get("priceToSalesTrailing12Months"),
                    "pb_ratio": info.get("priceToBook"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                    # Growth
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                    # Profitability
                    "profit_margin": info.get("profitMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "gross_margin": info.get("grossMargins"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                    # Financial Health
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "quick_ratio": info.get("quickRatio"),
                    # Other
                    "beta": info.get("beta"),
                    "dividend_yield": info.get("dividendYield"),
                    "payout_ratio": info.get("payoutRatio"),
                    "market_cap": info.get("marketCap"),
                }
                metrics_list.append(metrics)
        except Exception as e:
            print(f"Error fetching metrics: {e}")

        return pd.DataFrame(metrics_list)

    def get_earnings_history(self, symbol: str) -> pd.DataFrame:
        """
        Get earnings history and surprise data

        Earnings surprises are strong alpha signals:
        - Positive surprises often lead to momentum
        - Earnings revision patterns predict returns
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            # Get earnings dates and estimates
            earnings = ticker.earnings_dates
            if earnings is not None and not earnings.empty:
                return earnings

            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching earnings history for {symbol}: {e}")
            return pd.DataFrame()

    def get_analyst_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Get analyst recommendations history

        Analyst upgrades/downgrades can signal trends
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            recommendations = ticker.recommendations
            if recommendations is not None and not recommendations.empty:
                return recommendations

            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching recommendations for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_growth_metrics(
        self,
        symbol: str,
        periods: int = 4
    ) -> dict[str, float]:
        """
        Calculate growth metrics from financial statements

        Args:
            symbol: Ticker symbol
            periods: Number of periods for growth calculation

        Returns:
            Dictionary of growth metrics
        """
        statements = self.get_financial_statements(symbol)

        if not statements:
            return {}

        growth_metrics = {}

        try:
            income = statements.get("income_quarterly")
            if income is not None and len(income) >= periods:
                # Revenue growth (YoY)
                if "Total Revenue" in income.columns:
                    recent = income["Total Revenue"].iloc[0]
                    year_ago = income["Total Revenue"].iloc[min(4, len(income)-1)]
                    if year_ago and year_ago != 0:
                        growth_metrics["revenue_growth_yoy"] = (recent - year_ago) / abs(year_ago)

                # Net income growth
                if "Net Income" in income.columns:
                    recent = income["Net Income"].iloc[0]
                    year_ago = income["Net Income"].iloc[min(4, len(income)-1)]
                    if year_ago and year_ago != 0:
                        growth_metrics["net_income_growth_yoy"] = (recent - year_ago) / abs(year_ago)
        except Exception as e:
            print(f"Error calculating growth metrics: {e}")

        return growth_metrics
