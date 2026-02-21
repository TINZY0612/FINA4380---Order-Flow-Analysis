"""
Depth slope as a limit order book factor.

Depth slope = how quickly depth changes across levels (e.g. top 5 or 10).
- Steep (negative) slope: liquidity concentrated at the touch, thin book.
- Shallow slope: depth persists across levels, thicker book.
- Asymmetry (bid slope vs ask slope) can signal support/resistance.

Input: bids/asks as list of [price, size], e.g. [[p1,q1],[p2,q2],...] (best first).
"""
import numpy as np


def depth_slope_one_side(levels: list, n_levels: int = 10) -> float:
    """
    Slope of size across the first n_levels (1 to n_levels).
    Uses linear regression: size ~ level_index (1, 2, ..., n).
    Returns slope (positive = depth increases with level, negative = decreases).
    """
    if not levels or n_levels < 2:
        return 0.0
    sizes = [float(levels[i][1]) for i in range(min(n_levels, len(levels)))]
    if len(sizes) < 2:
        return 0.0
    x = np.arange(1, len(sizes) + 1, dtype=float)
    y = np.asarray(sizes)
    if np.var(x) == 0:
        return 0.0
    slope = np.cov(x, y)[0, 1] / np.var(x)
    return float(slope)


def depth_slope_factor(bids: list, asks: list, n_levels: int = 10) -> dict:
    """
    Compute depth slope for bid and ask side and an asymmetry.

    Returns
    -------
    dict with:
        bid_slope: slope of bid sizes across levels
        ask_slope: slope of ask sizes across levels
        slope_imbalance: (bid_slope - ask_slope) / (|bid_slope| + |ask_slope| + 1e-12)
                        > 0: bid side deeper (more size away from touch)
                        < 0: ask side deeper
    """
    bid_s = depth_slope_one_side(bids, n_levels)
    ask_s = depth_slope_one_side(asks, n_levels)
    denom = abs(bid_s) + abs(ask_s) + 1e-12
    slope_imbalance = (bid_s - ask_s) / denom
    return {
        "bid_slope": bid_s,
        "ask_slope": ask_s,
        "slope_imbalance": slope_imbalance,
    }


# Example: use with L2 snapshot
# bids = [[96000, 1.5], [95900, 2.1], [95800, 0.8], ...]  # [price, size]
# asks = [[96100, 1.2], [96200, 1.9], ...]
# out = depth_slope_factor(bids, asks, n_levels=5)
# if out["slope_imbalance"] > 0.2:
#     print("Bid side deeper book (more depth slope) -> potential support")
