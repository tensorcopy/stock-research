"""
Utility Functions
Common helper functions for stock research
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np


def resample_ohlcv(
    df: pd.DataFrame,
    target_freq: str = "W"
) -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency

    Args:
        df: DataFrame with OHLCV columns
        target_freq: Target frequency ('W', 'M', etc.)

    Returns:
        Resampled OHLCV DataFrame
    """
    return df.resample(target_freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()


def calculate_returns(
    prices: pd.Series,
    periods: int = 1,
    log_returns: bool = False
) -> pd.Series:
    """
    Calculate returns from price series

    Args:
        prices: Price series
        periods: Number of periods for return calculation
        log_returns: If True, calculate log returns

    Returns:
        Return series
    """
    if log_returns:
        return np.log(prices / prices.shift(periods))
    return prices.pct_change(periods)


def rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 60
) -> pd.Series:
    """Calculate rolling correlation between two series"""
    return series1.rolling(window).corr(series2)


def exponential_moving_average(
    series: pd.Series,
    span: int
) -> pd.Series:
    """Calculate exponential moving average"""
    return series.ewm(span=span, adjust=False).mean()


def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands

    Returns:
        DataFrame with columns: middle, upper, lower, bandwidth
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = (upper - lower) / middle

    return pd.DataFrame({
        "middle": middle,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth
    })


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """
    Calculate maximum drawdown and its duration

    Returns:
        Tuple of (max_drawdown, duration_in_periods)
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    max_dd = drawdown.min()

    # Calculate duration
    in_drawdown = drawdown < 0
    durations = []
    current = 0

    for is_dd in in_drawdown:
        if is_dd:
            current += 1
        else:
            if current > 0:
                durations.append(current)
            current = 0

    max_duration = max(durations) if durations else 0

    return max_dd, max_duration


def format_currency(value: float, currency: str = "$") -> str:
    """Format number as currency"""
    if abs(value) >= 1e9:
        return f"{currency}{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{currency}{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{currency}{value/1e3:.2f}K"
    return f"{currency}{value:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format number as percentage"""
    return f"{value * 100:.{decimals}f}%"


def get_trading_days(
    start_date: datetime,
    end_date: datetime,
    exclude_holidays: bool = True
) -> pd.DatetimeIndex:
    """
    Get list of trading days between dates

    Args:
        start_date: Start date
        end_date: End date
        exclude_holidays: If True, exclude US market holidays

    Returns:
        DatetimeIndex of trading days
    """
    all_days = pd.date_range(start=start_date, end=end_date, freq="B")

    if exclude_holidays:
        # Common US holidays (simplified)
        holidays = []
        for year in range(start_date.year, end_date.year + 1):
            # New Year's Day
            holidays.append(datetime(year, 1, 1))
            # MLK Day (3rd Monday of January)
            jan1 = datetime(year, 1, 1)
            mlk = jan1 + timedelta(days=(7 - jan1.weekday()) % 7 + 14)
            holidays.append(mlk)
            # Presidents Day (3rd Monday of February)
            feb1 = datetime(year, 2, 1)
            pres = feb1 + timedelta(days=(7 - feb1.weekday()) % 7 + 14)
            holidays.append(pres)
            # Memorial Day (last Monday of May)
            may31 = datetime(year, 5, 31)
            mem = may31 - timedelta(days=(may31.weekday() + 1) % 7)
            holidays.append(mem)
            # Independence Day
            holidays.append(datetime(year, 7, 4))
            # Labor Day (1st Monday of September)
            sep1 = datetime(year, 9, 1)
            labor = sep1 + timedelta(days=(7 - sep1.weekday()) % 7)
            holidays.append(labor)
            # Thanksgiving (4th Thursday of November)
            nov1 = datetime(year, 11, 1)
            thanks = nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21)
            holidays.append(thanks)
            # Christmas
            holidays.append(datetime(year, 12, 25))

        holidays = pd.DatetimeIndex(holidays)
        all_days = all_days.difference(holidays)

    return all_days


def normalize_symbol(symbol: str) -> str:
    """Normalize ticker symbol format"""
    return symbol.upper().replace(".", "-")


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Validate OHLCV DataFrame has required columns and format"""
    required_columns = ["open", "high", "low", "close", "volume"]
    has_columns = all(col in df.columns.str.lower() for col in required_columns)

    if not has_columns:
        return False

    # Check for valid data
    if df.empty:
        return False

    # Check high >= low
    if not (df["high"] >= df["low"]).all():
        return False

    return True


def align_series(*series: pd.Series) -> list[pd.Series]:
    """
    Align multiple series to common dates

    Returns:
        List of aligned series with same index
    """
    combined = pd.concat(series, axis=1)
    combined = combined.dropna()
    return [combined.iloc[:, i] for i in range(combined.shape[1])]
