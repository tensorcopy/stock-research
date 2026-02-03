"""
Sentiment Analysis Module
Analyze market sentiment from news, social media, and other sources
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import re


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    news_sources: list[str] = None
    lookback_days: int = 30

    def __post_init__(self):
        if self.news_sources is None:
            self.news_sources = ["yahoo", "google"]


class SentimentAnalyzer:
    """
    Analyze sentiment for stock selection

    Sentiment signals for alpha:
    - News sentiment momentum
    - Social media buzz/volume
    - Insider trading signals
    - Short interest changes
    - Options flow sentiment
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()

    def get_news_sentiment(
        self,
        symbol: str,
        days: int = 7
    ) -> dict:
        """
        Get aggregated news sentiment for a symbol

        Returns:
            Dictionary with sentiment scores and news count
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            days = days or self.config.lookback_days
            news = ticker.news

            if not news:
                return {"sentiment_score": 0, "news_count": 0, "articles": []}

            cutoff = datetime.now() - timedelta(days=days)

            # Simple sentiment based on title keywords
            positive_words = {
                "surge", "soar", "jump", "rally", "gain", "rise", "up",
                "beat", "exceed", "strong", "growth", "profit", "upgrade",
                "buy", "bullish", "outperform", "record", "breakthrough"
            }
            negative_words = {
                "fall", "drop", "decline", "plunge", "crash", "loss",
                "miss", "weak", "cut", "downgrade", "sell", "bearish",
                "underperform", "warning", "concern", "risk", "lawsuit"
            }

            sentiment_scores = []
            articles = []

            filtered = []
            for article in news:
                publish_time = article.get("providerPublishTime")
                if publish_time is None:
                    filtered.append(article)
                    continue

                published_at = datetime.fromtimestamp(publish_time)
                if published_at >= cutoff:
                    filtered.append(article)

            for article in filtered[:20]:  # Limit to recent 20 articles
                title = article.get("title", "").lower()

                pos_count = sum(1 for word in positive_words if word in title)
                neg_count = sum(1 for word in negative_words if word in title)

                if pos_count + neg_count > 0:
                    score = (pos_count - neg_count) / (pos_count + neg_count)
                else:
                    score = 0

                sentiment_scores.append(score)
                articles.append({
                    "title": article.get("title"),
                    "publisher": article.get("publisher"),
                    "link": article.get("link"),
                    "sentiment": score
                })

            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

            return {
                "sentiment_score": avg_sentiment,
                "news_count": len(filtered),
                "positive_ratio": sum(1 for s in sentiment_scores if s > 0) / len(sentiment_scores) if sentiment_scores else 0,
                "articles": articles
            }
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {e}")
            return {"sentiment_score": 0, "news_count": 0, "articles": []}

    def get_insider_activity(self, symbol: str) -> dict:
        """
        Get insider trading activity

        Insider buying is often a bullish signal
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            insider_transactions = ticker.insider_transactions
            insider_holders = ticker.insider_holders

            if insider_transactions is None or insider_transactions.empty:
                return {"net_insider_activity": 0, "transactions": []}

            # Calculate net insider activity
            recent = insider_transactions.head(20)

            buys = 0
            sells = 0

            for _, row in recent.iterrows():
                transaction = str(row.get("Transaction", "")).lower()
                shares = row.get("Shares", 0) or 0

                if "buy" in transaction or "purchase" in transaction:
                    buys += shares
                elif "sell" in transaction or "sale" in transaction:
                    sells += shares

            net_activity = buys - sells

            return {
                "net_insider_activity": net_activity,
                "total_buys": buys,
                "total_sells": sells,
                "buy_sell_ratio": buys / sells if sells > 0 else float('inf') if buys > 0 else 0,
                "transaction_count": len(recent)
            }
        except Exception as e:
            print(f"Error getting insider activity for {symbol}: {e}")
            return {"net_insider_activity": 0, "transactions": []}

    def get_short_interest(self, symbol: str) -> dict:
        """
        Get short interest data

        High short interest + positive catalyst = potential squeeze
        Increasing short interest may signal trouble ahead
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "short_ratio": info.get("shortRatio"),
                "short_percent_float": info.get("shortPercentOfFloat"),
                "shares_short": info.get("sharesShort"),
                "shares_short_prior": info.get("sharesShortPriorMonth"),
            }
        except Exception as e:
            print(f"Error getting short interest for {symbol}: {e}")
            return {}

    def get_institutional_holdings(self, symbol: str) -> dict:
        """
        Get institutional ownership data

        Institutional buying/selling patterns can signal trends
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            holders = ticker.institutional_holders
            info = ticker.info

            return {
                "institutional_ownership": info.get("heldPercentInstitutions"),
                "insider_ownership": info.get("heldPercentInsiders"),
                "top_holders": holders.to_dict("records") if holders is not None else []
            }
        except Exception as e:
            print(f"Error getting institutional holdings for {symbol}: {e}")
            return {}

    def calculate_composite_sentiment(
        self,
        symbol: str,
        weights: Optional[dict] = None
    ) -> dict:
        """
        Calculate composite sentiment score combining multiple signals

        Args:
            symbol: Ticker symbol
            weights: Custom weights for each sentiment component

        Returns:
            Composite sentiment score and breakdown
        """
        default_weights = {
            "news": 0.3,
            "insider": 0.3,
            "short_interest": 0.2,
            "institutional": 0.2
        }
        weights = weights or default_weights

        # Gather all sentiment data
        news = self.get_news_sentiment(symbol)
        insider = self.get_insider_activity(symbol)
        short = self.get_short_interest(symbol)
        institutional = self.get_institutional_holdings(symbol)

        # Normalize scores to [-1, 1] range
        scores = {}

        # News sentiment (already in range)
        scores["news"] = news.get("sentiment_score", 0)

        # Insider activity (normalize based on buy/sell ratio)
        bsr = insider.get("buy_sell_ratio", 1)
        if bsr == float('inf'):
            scores["insider"] = 1
        elif bsr == 0:
            scores["insider"] = -1
        else:
            scores["insider"] = min(1, max(-1, (bsr - 1) / 2))

        # Short interest (high = bearish)
        short_pct = short.get("short_percent_float") or 0
        scores["short_interest"] = -min(1, short_pct * 10)  # Scale: 10% short = -1

        # Institutional ownership (high = bullish)
        inst_pct = institutional.get("institutional_ownership") or 0
        scores["institutional"] = min(1, inst_pct * 2 - 1)  # Scale: 50% inst = 0

        # Calculate weighted composite
        composite = sum(
            scores[key] * weights[key]
            for key in weights
            if key in scores
        )

        return {
            "composite_score": composite,
            "component_scores": scores,
            "weights": weights
        }
