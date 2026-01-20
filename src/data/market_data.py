"""
Market Data Fetcher
Collects price, volume, and market data from various sources
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class MarketDataConfig:
    """Configuration for market data fetching"""
    source: str = "yfinance"  # yfinance, polygon, alpaca
    cache_dir: str = "./cache/market_data"
    rate_limit: int = 5  # requests per second


class MarketDataFetcher:
    """
    Fetch and manage market data for stock research

    Supports multiple data sources:
    - Yahoo Finance (free, good for historical data)
    - Polygon.io (real-time, requires API key)
    - Alpaca (free tier available, good for US equities)
    """

    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or MarketDataConfig()
        self._cache: dict[str, pd.DataFrame] = {}

    def get_ohlcv(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for given symbols

        Args:
            symbols: List of ticker symbols
            start_date: Start date for data
            end_date: End date (defaults to today)
            interval: Data interval (1d, 1h, 5m, etc.)

        Returns:
            Dictionary mapping symbols to DataFrames with columns:
            [open, high, low, close, volume, adjusted_close]
        """
        end_date = end_date or datetime.now()
        results = {}

        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"

            if cache_key in self._cache:
                results[symbol] = self._cache[cache_key]
                continue

            df = self._fetch_from_source(symbol, start_date, end_date, interval)
            if df is not None and not df.empty:
                self._cache[cache_key] = df
                results[symbol] = df

        return results

    def _fetch_from_source(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from configured source"""
        if self.config.source == "yfinance":
            return self._fetch_yfinance(symbol, start_date, end_date, interval)
        elif self.config.source == "polygon":
            return self._fetch_polygon(symbol, start_date, end_date, interval)
        else:
            raise ValueError(f"Unknown data source: {self.config.source}")

    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                return None

            # Standardize column names
            df.columns = df.columns.str.lower()
            df = df.rename(columns={
                "stock splits": "splits",
                "adj close": "adjusted_close"
            })

            return df
        except Exception as e:
            print(f"Error fetching {symbol} from yfinance: {e}")
            return None

    def _fetch_polygon(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io (requires API key)"""
        # Placeholder for Polygon integration
        raise NotImplementedError("Polygon.io integration not yet implemented")

    def get_market_cap(self, symbols: list[str]) -> dict[str, float]:
        """Get current market capitalization for symbols"""
        market_caps = {}

        try:
            import yfinance as yf
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_caps[symbol] = info.get("marketCap", 0)
        except Exception as e:
            print(f"Error fetching market cap: {e}")

        return market_caps

    def get_sector_data(self, symbols: list[str]) -> dict[str, dict]:
        """Get sector and industry classification"""
        sector_data = {}

        try:
            import yfinance as yf
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                sector_data[symbol] = {
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "country": info.get("country", "Unknown")
                }
        except Exception as e:
            print(f"Error fetching sector data: {e}")

        return sector_data


def get_sp500_symbols() -> list[str]:
    """Fetch current S&P 500 constituent symbols"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        return sp500_table["Symbol"].str.replace(".", "-").tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        # Return a subset of known symbols as fallback
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "BRK-B", "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD"
        ]


def get_nasdaq100_symbols() -> list[str]:
    """Fetch current NASDAQ 100 constituent symbols"""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        # Find the table with tickers
        for table in tables:
            if "Ticker" in table.columns:
                return table["Ticker"].tolist()
        return []
    except Exception as e:
        print(f"Error fetching NASDAQ 100 symbols: {e}")
        return []
