"""
Pipeline: L2 data (CCXT or QuestDB) → all factors → Rank IC report.

Usage:
  # Live: collect 2 min from CCXT at 100ms, then compute factors and rank IC
  python pipeline_rank_ic.py --source ccxt --duration 120 --tick 100

  # From QuestDB (run ccxt_l2_to_questdb.py first, or use existing DB)
  python pipeline_rank_ic.py --source questdb --limit 3000

  # Default: CCXT 60s, 1 symbol, 200ms tick (fewer API calls)
  python pipeline_rank_ic.py
"""

import argparse
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
from lob_factors import compute_all

# Defaults
DEFAULT_SYMBOLS = ["BTC/USDT"]
LOOKAHEAD = 2          # T+2s forward return
N_LEVELS = 10
MIN_SAMPLES = 30


def build_factor_df(snapshots: list, n_levels: int = N_LEVELS) -> pd.DataFrame:
    """Convert list of L2 snapshots (common format) to DataFrame with mid_price and all factors."""
    rows = []
    for s in snapshots:
        bids = s["bids"]
        asks = s["asks"]
        if not bids or not asks:
            continue
        mid = (bids[0][0] + asks[0][0]) / 2
        factors = compute_all(bids, asks, n_levels=n_levels)
        rows.append({
            "timestamp": s["timestamp"],
            "symbol": s["symbol"],
            "mid_price": mid,
            **factors,
        })
    return pd.DataFrame(rows)


def add_forward_return(df: pd.DataFrame, lookahead: int = LOOKAHEAD) -> pd.DataFrame:
    """Add ret_future = mid_price shifted by -lookahead (per symbol). Expects 1 row per tick."""
    df = df.sort_values(["symbol", "timestamp"]).copy()
    df["ret_future"] = df.groupby("symbol")["mid_price"].transform(
        lambda x: x.shift(-lookahead) / x - 1
    )
    return df


def resample_to_seconds(df: pd.DataFrame, factor_cols: list) -> pd.DataFrame:
    """Aggregate to 1-second bars (last mid_price, mean of factors) per symbol."""
    df = df.copy()
    df["time_sec"] = pd.to_datetime(df["timestamp"], unit="s").dt.floor("1s")
    agg = {"mid_price": "last"}
    for c in factor_cols:
        if c in df.columns:
            agg[c] = "mean"
    return df.groupby(["symbol", "time_sec"]).agg(agg).reset_index()


def rank_ic_report(
    df: pd.DataFrame,
    factor_columns: list,
    ret_column: str = "ret_future",
    min_samples: int = MIN_SAMPLES,
) -> pd.DataFrame:
    """
    Compute Spearman rank IC (and p-value) per factor, per symbol and pooled.
    Returns a summary DataFrame.
    """
    df_clean = df.dropna(subset=[ret_column] + factor_columns)
    if len(df_clean) < min_samples:
        return pd.DataFrame()

    results = []
    symbols = df_clean["symbol"].unique()

    for fac in factor_columns:
        if fac not in df_clean.columns:
            continue
        # Pooled IC
        ic_pool, p_pool = stats.spearmanr(df_clean[fac], df_clean[ret_column])
        n_pool = len(df_clean)
        results.append({
            "factor": fac,
            "symbol": "POOLED",
            "rank_ic": ic_pool if not np.isnan(ic_pool) else None,
            "p_value": p_pool,
            "n": n_pool,
        })
        # Per-symbol IC
        for sym in symbols:
            sub = df_clean[df_clean["symbol"] == sym]
            if len(sub) < min_samples:
                continue
            ic, p = stats.spearmanr(sub[fac], sub[ret_column])
            if np.isnan(ic):
                continue
            results.append({
                "factor": fac,
                "symbol": sym,
                "rank_ic": ic,
                "p_value": p,
                "n": len(sub),
            })

    return pd.DataFrame(results)


def run_ccxt(duration_seconds: int, tick_ms: int, symbols: list, n_levels: int):
    from data_sources import collect_l2_ccxt
    snapshots = list(collect_l2_ccxt(symbols, duration_seconds=duration_seconds, tick_ms=tick_ms, n_levels=n_levels))
    return snapshots


def run_questdb(limit: int, symbol, n_levels: int):
    from data_sources import get_l2_from_questdb
    snapshots = get_l2_from_questdb(limit=limit, symbol=symbol)
    return snapshots


def main():
    ap = argparse.ArgumentParser(description="L2 → factors → Rank IC")
    ap.add_argument("--source", choices=["ccxt", "questdb"], default="ccxt", help="L2 data source")
    ap.add_argument("--duration", type=int, default=60, help="CCXT: collect duration (seconds)")
    ap.add_argument("--tick", type=int, default=200, help="CCXT: tick interval (ms)")
    ap.add_argument("--limit", type=int, default=3000, help="QuestDB: max rows")
    ap.add_argument("--symbol", type=str, default=None, help="Single symbol (e.g. BTCUSDT for QuestDB)")
    ap.add_argument("--lookahead", type=int, default=LOOKAHEAD, help="Forward return horizon (steps)")
    ap.add_argument("--resample", action="store_true", default=True, help="Resample to 1s before IC")
    ap.add_argument("--no-resample", action="store_false", dest="resample")
    args = ap.parse_args()

    symbols = [args.symbol.replace("USDT", "/USDT")] if args.symbol and "USDT" in args.symbol else DEFAULT_SYMBOLS
    if args.symbol and "USDT" not in args.symbol and args.source == "questdb":
        symbols = [args.symbol]

    print("Loading L2 snapshots...", end=" ")
    if args.source == "ccxt":
        snapshots = run_ccxt(args.duration, args.tick, symbols, N_LEVELS)
    else:
        snapshots = run_questdb(args.limit, args.symbol, N_LEVELS)
    print(f"got {len(snapshots)}.")

    if not snapshots:
        print("No data. Check source (CCXT rate limit, QuestDB table l2_orderbook).")
        sys.exit(1)

    df = build_factor_df(snapshots, n_levels=N_LEVELS)
    factor_cols = [c for c in df.columns if c not in ("timestamp", "symbol", "mid_price")]

    if args.resample:
        # Resample to 1s first, then add forward return (lookahead = number of seconds)
        df = resample_to_seconds(df, factor_cols)
        df["timestamp"] = df["time_sec"].astype("int64") // 10**9
        df["ret_future"] = df.groupby("symbol")["mid_price"].transform(
            lambda x: x.shift(-args.lookahead) / x - 1
        )
    else:
        df = add_forward_return(df, lookahead=args.lookahead)

    report = rank_ic_report(df, factor_cols, ret_column="ret_future", min_samples=MIN_SAMPLES)
    if report.empty:
        print("Not enough samples for rank IC.")
        sys.exit(0)

    print("\nRank IC report (Spearman factor vs forward return):")
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    main()
