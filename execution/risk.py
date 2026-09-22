"""Risk management helpers for live execution."""

from __future__ import annotations


def should_trigger_stop_loss(entry_price: float, current_price: float, stop_loss_pct: float) -> bool:
    """Return True when stop-loss threshold is breached."""
    stop_loss_price = entry_price * (1.0 - stop_loss_pct)
    return current_price <= stop_loss_price
