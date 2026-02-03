"""
Portfolio Management Module
Track positions, cash, and portfolio value
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class Position:
    """Represents a single position in a stock"""
    symbol: str
    quantity: float
    avg_cost: float
    entry_date: datetime = field(default_factory=datetime.now)

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return self.market_value(current_price) - self.cost_basis

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.cost_basis == 0:
            return 0
        return self.unrealized_pnl(current_price) / self.cost_basis


class Portfolio:
    """
    Manages a portfolio of positions

    Tracks:
    - Current positions and their cost basis
    - Cash balance
    - Transaction history
    - P&L calculations
    """

    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.transaction_history: list[dict] = []

    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        date: Optional[datetime] = None
    ):
        """Add to existing position or create new one"""
        date = date or datetime.now()

        if symbol in self.positions:
            # Update average cost
            existing = self.positions[symbol]
            total_cost = existing.cost_basis + (quantity * price)
            total_quantity = existing.quantity + quantity
            new_avg_cost = total_cost / total_quantity if total_quantity > 0 else 0

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                avg_cost=new_avg_cost,
                entry_date=existing.entry_date
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                entry_date=date
            )

        self.transaction_history.append({
            "date": date,
            "symbol": symbol,
            "action": "buy",
            "quantity": quantity,
            "price": price
        })

    def remove_position(
        self,
        symbol: str,
        quantity: float,
        date: Optional[datetime] = None,
        price: Optional[float] = None
    ):
        """Remove from position (sell)"""
        date = date or datetime.now()

        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        remaining = position.quantity - quantity

        if remaining <= 0.001:  # Effectively zero
            del self.positions[symbol]
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining,
                avg_cost=position.avg_cost,
                entry_date=position.entry_date
            )

        self.transaction_history.append({
            "date": date,
            "symbol": symbol,
            "action": "sell",
            "quantity": quantity,
            "price": price
        })

    def calculate_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.quantity * prices.get(pos.symbol, pos.avg_cost)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_positions_summary(self) -> list[dict]:
        """Get summary of all positions"""
        return [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "cost_basis": pos.cost_basis
            }
            for pos in self.positions.values()
        ]

    def get_holdings_df(self, prices: dict[str, float]) -> pd.DataFrame:
        """Get holdings as DataFrame with current values"""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for pos in self.positions.values():
            current_price = prices.get(pos.symbol, pos.avg_cost)
            data.append({
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": current_price,
                "market_value": pos.market_value(current_price),
                "cost_basis": pos.cost_basis,
                "unrealized_pnl": pos.unrealized_pnl(current_price),
                "unrealized_pnl_pct": pos.unrealized_pnl_pct(current_price) * 100,
                "entry_date": pos.entry_date
            })

        df = pd.DataFrame(data)
        df["weight"] = df["market_value"] / df["market_value"].sum()
        return df

    def get_sector_allocation(
        self,
        prices: dict[str, float],
        sector_map: dict[str, str]
    ) -> dict[str, float]:
        """Calculate allocation by sector"""
        holdings = self.get_holdings_df(prices)

        if holdings.empty:
            return {}

        holdings["sector"] = holdings["symbol"].map(sector_map).fillna("Unknown")
        sector_allocation = holdings.groupby("sector")["weight"].sum().to_dict()

        return sector_allocation

    def get_concentration_metrics(self, prices: dict[str, float]) -> dict:
        """Calculate portfolio concentration metrics"""
        holdings = self.get_holdings_df(prices)

        if holdings.empty:
            return {"n_positions": 0}

        weights = holdings["weight"].sort_values(ascending=False)

        return {
            "n_positions": len(holdings),
            "top_1_weight": weights.iloc[0] if len(weights) > 0 else 0,
            "top_5_weight": weights.head(5).sum(),
            "top_10_weight": weights.head(10).sum(),
            "herfindahl_index": (weights ** 2).sum(),  # Concentration measure
            "effective_n": 1 / (weights ** 2).sum() if (weights ** 2).sum() > 0 else 0
        }

    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_cash
        self.positions = {}
        self.transaction_history = []
