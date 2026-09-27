"""Exchange client abstractions for order and market operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrderRequest:
    """Represents a standardized market order request."""

    symbol: str
    side: str
    amount: float


class ExchangeClient:
    """Minimal exchange interface, intended for CCXT-backed implementation."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.api_key = api_key
        self.api_secret = api_secret

    def place_market_order(self, order: OrderRequest) -> dict[str, Any]:
        """Place a market order through the connected exchange (not implemented)."""
        raise NotImplementedError("Live order placement is intentionally not implemented; see the README.")
