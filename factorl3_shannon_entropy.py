"""
Shannon entropy as a limit order book factor.

Entropy H = -Σ p_i log2(p_i) measures how spread out a distribution is.
- High LOB entropy: depth spread across levels (balanced book).
- Low LOB entropy: depth concentrated at few levels (e.g. all at touch).

Input: bids/asks as list of [price, size], e.g. [[p1,q1],[p2,q2],...] (best first).
"""
import numpy as np


def shannon_entropy(probs: np.ndarray) -> float:
    """
    Shannon entropy in bits: H = -Σ p_i log2(p_i).
    probs: 1D array of probabilities (should sum to 1; zeros excluded from sum).
    """
    p = np.asarray(probs, dtype=float).ravel()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def depth_entropy_one_side(levels: list, n_levels: int = 10) -> float:
    """
    Entropy of the depth distribution over the first n_levels.
    p_i = size at level i / total size (over those levels).
    """
    if not levels or n_levels < 1:
        return 0.0
    sizes = np.array([float(levels[i][1]) for i in range(min(n_levels, len(levels)))])
    total = sizes.sum()
    if total <= 0:
        return 0.0
    probs = sizes / total
    return shannon_entropy(probs)


def entropy_factor(bids: list, asks: list, n_levels: int = 10) -> dict:
    """
    Compute Shannon entropy of depth on each side and an imbalance.

    Returns
    -------
    dict with:
        bid_entropy: entropy of bid depth distribution (bits)
        ask_entropy: entropy of ask depth distribution (bits)
        entropy_imbalance: (bid_entropy - ask_entropy) / (bid_entropy + ask_entropy + 1e-12)
                          > 0: bid side more spread out
                          < 0: ask side more spread out
        combined_entropy: entropy of combined bid+ask depth (all 2*n_levels levels)
    """
    bid_h = depth_entropy_one_side(bids, n_levels)
    ask_h = depth_entropy_one_side(asks, n_levels)
    denom = bid_h + ask_h + 1e-12
    entropy_imbalance = (bid_h - ask_h) / denom

    # Combined: treat all levels as one distribution
    bid_sizes = [float(b[1]) for b in bids[:n_levels]]
    ask_sizes = [float(a[1]) for a in asks[:n_levels]]
    all_sizes = np.array(bid_sizes + ask_sizes)
    total = all_sizes.sum()
    combined_h = shannon_entropy(all_sizes / total) if total > 0 else 0.0

    return {
        "bid_entropy": bid_h,
        "ask_entropy": ask_h,
        "entropy_imbalance": entropy_imbalance,
        "combined_entropy": combined_h,
    }


# Example: use with L2 snapshot
# bids = [[96000, 1.5], [95900, 2.1], [95800, 0.8], ...]
# asks = [[96100, 1.2], [96200, 1.9], ...]
# out = entropy_factor(bids, asks, n_levels=10)
# if out["entropy_imbalance"] > 0.2:
#     print("Bid side more spread -> ask side more concentrated near touch")
