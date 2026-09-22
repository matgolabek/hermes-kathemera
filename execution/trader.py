"""Live trading orchestration logic."""

from __future__ import annotations

from execution.exchange_client import ExchangeClient, OrderRequest
from execution.risk import should_trigger_stop_loss


class Trader:
    """Coordinates signal handling, risk checks, and order execution."""

    def __init__(self, client: ExchangeClient, stop_loss_pct: float, order_size: float) -> None:
        self.client = client
        self.stop_loss_pct = stop_loss_pct
        self.order_size = order_size

    def execute_signal(self, symbol: str, signal: str, entry_price: float, current_price: float) -> None:
        """Execute a buy/sell signal while applying simple stop-loss logic."""
        if should_trigger_stop_loss(entry_price, current_price, self.stop_loss_pct):
            stop_order = OrderRequest(symbol=symbol, side="sell", amount=self.order_size)
            self.client.place_market_order(stop_order)
            return

        if signal not in {"buy", "sell"}:
            return

        order = OrderRequest(symbol=symbol, side=signal, amount=self.order_size)
        self.client.place_market_order(order)
