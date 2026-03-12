"""
Pipeline: L2 data (CCXT or QuestDB) → LOB factors → Rank IC backtest.

LOB factors (inlined from lob_factors): microprice_dev, depth_slope_imb, entropy_imb,
obi, bid_liq_5, ask_liq_5, etc. Expects L2 snapshots as list of
{ "timestamp": float (Unix s), "symbol": str, "bids": [[p,q],...], "asks": [[p,q],...] }
(e.g. from data_sources.get_l2_from_questdb).

Usage:
  # From QuestDB: one symbol
  python pipeline_rank_ic.py --source questdb --symbol BTCUSDT --limit 3000

  # From QuestDB: all symbols in one run
  python pipeline_rank_ic.py --source questdb --all-symbols --limit 50000

  # Live CCXT (2 min, 100ms tick)
  python pipeline_rank_ic.py --source ccxt --duration 120 --tick 100

  # Default: CCXT 60s, 1 symbol, 200ms tick
  python pipeline_rank_ic.py
"""

import argparse
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from factor2_microprice_deviation import calculate_microprice
from factor3_depth_slope import depth_slope_factor as _depth_slope_factor
from factorl3_shannon_entropy import entropy_factor as _entropy_factor

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEFAULT_SYMBOLS = ["BTC/USDT"]
LOOKAHEAD = 2
N_LEVELS = 10
MIN_SAMPLES = 30
TICK_SEC = 0.1  # 100ms grid for timestamp alignment


# -----------------------------------------------------------------------------
# LOB factors (inlined for single-file backtest; inputs: bids/asks = list of [price, size])
# -----------------------------------------------------------------------------

def _microprice_deviation(bids: list, asks: list) -> float:
    """Microprice minus mid (L1). Positive => bid side heavier."""
    if not bids or not asks:
        return 0.0
    bp, bq = float(bids[0][0]), float(bids[0][1])
    ap, aq = float(asks[0][0]), float(asks[0][1])
    mid = (bp + ap) / 2
    micro = calculate_microprice(bp, bq, ap, aq)
    return micro - mid


def _obi_weighted(bids: list, asks: list, n_levels: int = 20, decay: float = 50.0) -> float:
    """Weighted order book imbalance. Weights = exp(-decay * distance_pct_from_mid)."""
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


def _liquidity_top_n(bids: list, asks: list, n: int = 5) -> tuple:
    """Total size on bid and ask over top n levels."""
    bid_liq = sum(float(b[1]) for b in bids[:n])
    ask_liq = sum(float(a[1]) for a in asks[:n])
    return bid_liq, ask_liq


def compute_all_factors(
    bids: List[List[float]],
    asks: List[List[float]],
    n_levels: int = N_LEVELS,
) -> Dict[str, Any]:
    """
    Compute all LOB factors from one L2 snapshot (bids/asks = list of [price, size], best first).
    Used for QuestDB/CCXT snapshot rows.
    """
    row = {}
    row["microprice_dev"] = _microprice_deviation(bids, asks)
    ds = _depth_slope_factor(bids, asks, n_levels=n_levels)
    row["depth_slope_imb"] = ds["slope_imbalance"]
    row["bid_slope"] = ds["bid_slope"]
    row["ask_slope"] = ds["ask_slope"]
    ent = _entropy_factor(bids, asks, n_levels=n_levels)
    row["entropy_imb"] = ent["entropy_imbalance"]
    row["combined_entropy"] = ent["combined_entropy"]
    row["obi"] = _obi_weighted(bids, asks, n_levels=n_levels)
    bid_liq, ask_liq = _liquidity_top_n(bids, asks, n=5)
    row["bid_liq_5"] = bid_liq
    row["ask_liq_5"] = ask_liq
    return row


# -----------------------------------------------------------------------------
# Pipeline: snapshots → factor DataFrame → forward return → Rank IC
# -----------------------------------------------------------------------------

def align_snapshots_to_grid(
    snapshots: List[Dict[str, Any]],
    tick_sec: float = TICK_SEC,
) -> List[Dict[str, Any]]:
    """
    Preprocess L2 snapshots so timestamps lie on a regular grid (multiples of tick_sec).
    For each (symbol, grid_time) keeps one snapshot: the one with original timestamp
    closest to that grid time. Output timestamps are set to the grid time.
    """
    if not snapshots or tick_sec <= 0:
        return snapshots
    # key (symbol, grid_ts) -> (snapshot, original_ts) for "closest" selection
    best: Dict[tuple, tuple] = {}
    for s in snapshots:
        ts = s.get("timestamp")
        if ts is None:
            continue
        if hasattr(ts, "timestamp"):
            ts = ts.timestamp()
        ts = float(ts)
        sym = str(s.get("symbol", ""))
        grid_ts = np.floor(ts / tick_sec) * tick_sec
        key = (sym, grid_ts)
        if key not in best or abs(ts - grid_ts) < abs(best[key][1] - grid_ts):
            out = dict(s)
            out["timestamp"] = float(grid_ts)
            best[key] = (out, ts)
    result = [t[0] for t in best.values()]
    result.sort(key=lambda x: (x["symbol"], x["timestamp"]))
    return result


def fill_snapshot_timestamp_gaps(
    snapshots: List[Dict[str, Any]],
    tick_sec: float = TICK_SEC,
) -> List[Dict[str, Any]]:
    """
    For each symbol, add snapshots so consecutive timestamps are exactly tick_sec apart.
    Missing timestamps get the previous snapshot's bids/asks (forward-filled), with timestamp set to the grid time.
    """
    if not snapshots or tick_sec <= 0:
        return snapshots
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for s in snapshots:
        sym = str(s.get("symbol", ""))
        ts = s.get("timestamp")
        if hasattr(ts, "timestamp"):
            ts = ts.timestamp()
        ts = float(ts)
        by_symbol.setdefault(sym, []).append((ts, s))
    out = []
    for sym, list_ts_s in by_symbol.items():
        list_ts_s.sort(key=lambda x: x[0])
        t_min, t_max = list_ts_s[0][0], list_ts_s[-1][0]
        full_ts = np.round(np.arange(t_min, t_max + tick_sec * 0.5, tick_sec), decimals=10)
        idx = 0
        for t in full_ts:
            t = float(t)
            while idx + 1 < len(list_ts_s) and list_ts_s[idx + 1][0] <= t:
                idx += 1
            _, s = list_ts_s[idx]
            out.append({
                "timestamp": t,
                "symbol": sym,
                "bids": list(s["bids"]),
                "asks": list(s["asks"]),
            })
    out.sort(key=lambda x: (x["symbol"], x["timestamp"]))
    return out


def align_snapshots_to_master_timestamps(
    snapshots: List[Dict[str, Any]],
    master_symbol: str = None,
) -> List[Dict[str, Any]]:
    """
    Force all symbols to share the same timestamp grid. By default uses the symbol with
    the smallest number of timestamps as master so all symbols can align. Each symbol
    gets exactly len(master_ts) rows; missing times are forward-filled from previous snapshot.
    """
    if not snapshots:
        return snapshots
    by_symbol: Dict[str, List[tuple]] = {}
    for s in snapshots:
        sym = str(s.get("symbol", ""))
        ts = s.get("timestamp")
        if hasattr(ts, "timestamp"):
            ts = ts.timestamp()
        by_symbol.setdefault(sym, []).append((float(ts), s))
    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda x: x[0])

    if not master_symbol or master_symbol not in by_symbol:
        master_symbol = min(by_symbol.keys(), key=lambda k: len(by_symbol[k]))
    master_ts = sorted(set(t for t, _ in by_symbol[master_symbol]))

    out = []
    for sym, list_ts_s in by_symbol.items():
        if not list_ts_s:
            continue
        idx = 0
        for t in master_ts:
            while idx + 1 < len(list_ts_s) and list_ts_s[idx + 1][0] <= t:
                idx += 1
            _, s = list_ts_s[idx]
            out.append({
                "timestamp": t,
                "symbol": sym,
                "bids": list(s["bids"]),
                "asks": list(s["asks"]),
            })
    out.sort(key=lambda x: (x["symbol"], x["timestamp"]))
    return out


def snapshots_to_raw_df(snapshots: List[Dict[str, Any]], n_levels: int = 20) -> pd.DataFrame:
    """Convert snapshots to QuestDB-style columns: timestamp, symbol, bid_1_p, bid_1_s, ..., ask_N_p, ask_N_s."""
    if not snapshots:
        return pd.DataFrame()
    n = n_levels
    rows = []
    for s in snapshots:
        bids = (s.get("bids") or [])[:n]
        asks = (s.get("asks") or [])[:n]
        ts = s.get("timestamp")
        if hasattr(ts, "timestamp"):
            ts = ts.timestamp()
        row = {"timestamp": float(ts), "symbol": str(s.get("symbol", ""))}
        for i in range(n):
            row[f"bid_{i+1}_p"] = float(bids[i][0]) if i < len(bids) else 0.0
            row[f"bid_{i+1}_s"] = float(bids[i][1]) if i < len(bids) else 0.0
        for i in range(n):
            row[f"ask_{i+1}_p"] = float(asks[i][0]) if i < len(asks) else 0.0
            row[f"ask_{i+1}_s"] = float(asks[i][1]) if i < len(asks) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_factor_df(snapshots: list, n_levels: int = N_LEVELS) -> pd.DataFrame:
    """
    Convert L2 snapshots (QuestDB/CCXT common format) to DataFrame with mid_price and factors.
    Each snapshot: { timestamp (float s), symbol (str), bids [[p,q],...], asks [[p,q],...] }.
    """
    rows = []
    for s in snapshots:
        bids = s.get("bids") or []
        asks = s.get("asks") or []
        if not bids or not asks:
            continue
        mid = (float(bids[0][0]) + float(asks[0][0])) / 2
        factors = compute_all_factors(bids, asks, n_levels=n_levels)
        ts = s.get("timestamp")
        if hasattr(ts, "timestamp"):
            ts = ts.timestamp()
        rows.append({
            "timestamp": float(ts),
            "symbol": str(s.get("symbol", "")),
            "mid_price": mid,
            **factors,
        })
    return pd.DataFrame(rows)


def fill_timestamp_gaps(df: pd.DataFrame, tick_sec: float = TICK_SEC) -> pd.DataFrame:
    """
    For each symbol, add rows so consecutive timestamps are exactly tick_sec apart.
    Missing timestamps get forward-filled values (same mid_price and factors as previous row).
    """
    if df.empty or tick_sec <= 0:
        return df
    dfs = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        t_min = g["timestamp"].min()
        t_max = g["timestamp"].max()
        full_ts = np.arange(t_min, t_max + tick_sec * 0.5, tick_sec)
        full_ts = np.round(full_ts, decimals=10)
        full_df = pd.DataFrame({"timestamp": full_ts})
        merged = full_df.merge(g, on="timestamp", how="left")
        merged["symbol"] = merged["symbol"].fillna(sym)
        merged = merged.ffill()
        dfs.append(merged)
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def add_forward_return(df: pd.DataFrame, lookahead: int = LOOKAHEAD) -> pd.DataFrame:
    """Add ret_future = mid_price shifted by -lookahead (per symbol)."""
    df = df.sort_values(["symbol", "timestamp"]).copy()
    df["ret_future"] = df.groupby("symbol")["mid_price"].transform(
        lambda x: x.shift(-lookahead) / x - 1
    )
    return df


def resample_to_seconds(df: pd.DataFrame, factor_cols: list) -> pd.DataFrame:
    """Aggregate to 1-second bars (last mid_price, mean of factors) per symbol."""
    df = df.copy()
    df["time_sec"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.floor("1s")
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
    """Spearman rank IC (and p-value) per factor, per symbol and pooled."""
    df_clean = df.dropna(subset=[ret_column] + factor_columns)
    if len(df_clean) < min_samples:
        return pd.DataFrame()

    results = []
    symbols = df_clean["symbol"].unique()

    for fac in factor_columns:
        if fac not in df_clean.columns:
            continue
        ic_pool, p_pool = stats.spearmanr(df_clean[fac], df_clean[ret_column])
        results.append({
            "factor": fac,
            "symbol": "POOLED",
            "rank_ic": ic_pool if not np.isnan(ic_pool) else None,
            "p_value": p_pool,
            "n": len(df_clean),
        })
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
    return list(collect_l2_ccxt(symbols, duration_seconds=duration_seconds, tick_ms=tick_ms, n_levels=n_levels))


def run_questdb(limit: int, symbol, n_levels: int):
    from data_sources import get_l2_from_questdb
    return get_l2_from_questdb(limit=limit, symbol=symbol)


def main():
    ap = argparse.ArgumentParser(description="L2 (QuestDB/CCXT) → LOB factors → Rank IC backtest")
    ap.add_argument("--source", choices=["ccxt", "questdb"], default="questdb", help="L2 data source (default: questdb)")
    ap.add_argument("--duration", type=int, default=60, help="CCXT: collect duration (seconds)")
    ap.add_argument("--tick", type=int, default=200, help="CCXT: tick interval (ms)")
    ap.add_argument("--limit", type=int, default=3000, help="QuestDB: max rows to load")
    ap.add_argument("--symbol", type=str, default=None, help="Single symbol (e.g. BTCUSDT). Omit with questdb for all symbols.")
    ap.add_argument("--all-symbols", action="store_true", help="QuestDB: load all symbols (ignores --symbol)")
    ap.add_argument("--lookahead", type=int, default=LOOKAHEAD, help="Forward return horizon (seconds when resampled)")
    ap.add_argument("--resample", action="store_true", default=True, help="Resample to 1s before IC")
    ap.add_argument("--no-resample", action="store_false", dest="resample")
    ap.add_argument("--levels", type=int, default=N_LEVELS, help="LOB depth levels for factors")
    ap.add_argument("--tick-sec", type=float, default=TICK_SEC, help="Timestamp grid interval in seconds (default 0.1 = 100ms)")
    ap.add_argument("--out-csv", type=str, default=None, metavar="FILE", help="Write preprocessed factor DataFrame to CSV (grid-aligned)")
    args = ap.parse_args()

    symbols = [args.symbol.replace("USDT", "/USDT")] if args.symbol and "USDT" in args.symbol else DEFAULT_SYMBOLS
    if args.symbol and "USDT" not in args.symbol and args.source == "questdb":
        symbols = [args.symbol]

    questdb_symbol = None if (args.source == "questdb" and (args.all_symbols or not args.symbol)) else args.symbol

    if args.source == "ccxt":
        print("Loading L2 snapshots...", end=" ")
        snapshots = run_ccxt(args.duration, args.tick, symbols, args.levels)
    else:
        scope = "all symbols" if questdb_symbol is None else f"symbol {questdb_symbol}"
        print(f"Loading L2 snapshots ({scope})...", end=" ")
        snapshots = run_questdb(args.limit, questdb_symbol, args.levels)
    print(f"got {len(snapshots)}.")

    if not snapshots:
        print("No data. Check QuestDB table l2_orderbook or CCXT.")
        sys.exit(1)

    print("Aligning timestamps to {:.0f}ms grid...".format(args.tick_sec * 1000), end=" ")
    snapshots = align_snapshots_to_grid(snapshots, tick_sec=args.tick_sec)
    print(f"{len(snapshots)} snapshots.")

    print("Filling timestamp gaps (100ms grid per symbol)...", end=" ")
    snapshots = fill_snapshot_timestamp_gaps(snapshots, tick_sec=args.tick_sec)
    print(f"{len(snapshots)} snapshots.")

    print("Aligning all symbols to master (smallest timestamp count)...", end=" ")
    snapshots = align_snapshots_to_master_timestamps(snapshots)
    print(f"{len(snapshots)} snapshots (same row count per symbol).")

    if args.out_csv:
        raw_df = snapshots_to_raw_df(snapshots, n_levels=args.levels)
        raw_df.to_csv(args.out_csv, index=False)
        print(f"Raw L2 data (QuestDB-style columns) written to {args.out_csv} ({len(raw_df)} rows).")

    df = build_factor_df(snapshots, n_levels=args.levels)
    factor_cols = [c for c in df.columns if c not in ("timestamp", "symbol", "mid_price")]

    if args.resample:
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
