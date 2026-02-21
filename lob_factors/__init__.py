"""
LOB factor library: one interface for all limit order book factors.

All functions expect bids/asks as list of [price, size], best first.

Usage:
  from lob_factors import compute_all, microprice_deviation, depth_slope_factor, entropy_factor, obi_weighted

  row = compute_all(bids, asks, n_levels=10)
  # row = { "microprice_dev", "depth_slope_imb", "entropy_imb", "obi", "bid_liq_5", "ask_liq_5", ... }
"""

from typing import List, Dict, Any
import numpy as np

# Import existing factor modules (same repo)
from factor2_microprice_deviation import calculate_microprice
from factor3_depth_slope import depth_slope_factor as _depth_slope_factor
from factorl3_shannon_entropy import entropy_factor as _entropy_factor


def microprice_deviation(bids: list, asks: list) -> float:
    """Microprice minus mid (L1). Positive => bid side heavier."""
    if not bids or not asks:
        return 0.0
    bp, bq = float(bids[0][0]), float(bids[0][1])
    ap, aq = float(asks[0][0]), float(asks[0][1])
    mid = (bp + ap) / 2
    micro = calculate_microprice(bp, bq, ap, aq)
    return micro - mid


def obi_weighted(bids: list, asks: list, n_levels: int = 20, decay: float = 50.0) -> float:
    """
    Weighted order book imbalance: (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask).
    Weights = exp(-decay * distance_pct_from_mid). Closer levels matter more.
    """
    if not bids or not asks:
        return 0.0
    bids = np.array(bids[:n_levels], dtype=float)
    asks = np.array(asks[:n_levels], dtype=float)
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    mid = (bids[0, 0] + asks[0, 0]) / 2
    if mid <= 0:
        return 0.0
    bid_dist = (mid - bids[:, 0]) / mid
    ask_dist = (asks[:, 0] - mid) / mid
    bid_w = np.exp(-decay * bid_dist)
    ask_w = np.exp(-decay * ask_dist)
    bid_pressure = np.sum(bids[:, 1] * bid_w)
    ask_pressure = np.sum(asks[:, 1] * ask_w)
    total = bid_pressure + ask_pressure
    if total <= 0:
        return 0.0
    return float((bid_pressure - ask_pressure) / total)


def liquidity_top_n(bids: list, asks: list, n: int = 5) -> tuple:
    """Total size on bid and ask over top n levels. For depletion-style factors."""
    bid_liq = sum(float(b[1]) for b in bids[:n])
    ask_liq = sum(float(a[1]) for a in asks[:n])
    return bid_liq, ask_liq


def compute_all(
    bids: List[List[float]],
    asks: List[List[float]],
    n_levels: int = 10,
) -> Dict[str, Any]:
    """
    Compute all LOB factors from one snapshot. Returns a flat dict for one row.

    bids, asks: list of [price, size], best first.
    n_levels: used for depth slope, entropy, and OBI (OBI uses min(n_levels, 20) internally).
    """
    row = {}
    # L1 microprice
    row["microprice_dev"] = microprice_deviation(bids, asks)
    # Depth slope
    ds = _depth_slope_factor(bids, asks, n_levels=n_levels)
    row["depth_slope_imb"] = ds["slope_imbalance"]
    row["bid_slope"] = ds["bid_slope"]
    row["ask_slope"] = ds["ask_slope"]
    # Entropy
    ent = _entropy_factor(bids, asks, n_levels=n_levels)
    row["entropy_imb"] = ent["entropy_imbalance"]
    row["combined_entropy"] = ent["combined_entropy"]
    # Weighted OBI
    row["obi"] = obi_weighted(bids, asks, n_levels=n_levels)
    # Liquidity (top 5)
    bid_liq, ask_liq = liquidity_top_n(bids, asks, n=5)
    row["bid_liq_5"] = bid_liq
    row["ask_liq_5"] = ask_liq
    return row


# Re-export for direct use
__all__ = [
    "compute_all",
    "microprice_deviation",
    "depth_slope_factor",
    "entropy_factor",
    "obi_weighted",
    "liquidity_top_n",
    "calculate_microprice",
]

# Alias so callers can use depth_slope_factor / entropy_factor from here
depth_slope_factor = _depth_slope_factor
entropy_factor = _entropy_factor
