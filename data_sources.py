"""
Data adapters: produce L2 snapshots in a common format.

Format: list of dicts, each:
  { "timestamp": float (Unix s), "symbol": str, "bids": [[p,q],...], "asks": [[p,q],...] }
"""

import time
from typing import List, Dict, Any, Optional

# Optional: for QuestDB path
try:
    from QuestDB.questdb_fetch import fetch_l2_orderbook
except Exception:
    fetch_l2_orderbook = None


def get_l2_snapshot_ccxt(symbol: str = "BTC/USDT", exchange=None, n_levels: int = 20) -> Optional[Dict[str, Any]]:
    """Fetch one L2 snapshot via CCXT. Returns one row in common format or None."""
    if exchange is None:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    try:
        ob = exchange.fetch_order_book(symbol, limit=n_levels)
        bids = [[float(p), float(q)] for p, q in ob.get("bids", [])[:n_levels]]
        asks = [[float(p), float(q)] for p, q in ob.get("asks", [])[:n_levels]]
        if not bids or not asks:
            return None
        ts = ob.get("timestamp")
        if ts is None:
            ts = time.time() * 1000
        return {
            "timestamp": ts / 1000.0,
            "symbol": symbol.replace("/", ""),
            "bids": bids,
            "asks": asks,
        }
    except Exception:
        return None


def collect_l2_ccxt(
    symbols: List[str],
    duration_seconds: float = 120,
    tick_ms: int = 100,
    n_levels: int = 20,
):
    """
    Poll CCXT at tick_ms for duration_seconds; yield snapshots in common format.
    Yields dicts: { timestamp, symbol, bids, asks }.
    """
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    tick_sec = tick_ms / 1000.0
    end = time.time() + duration_seconds
    while time.time() < end:
        t0 = time.time()
        for sym in symbols:
            row = get_l2_snapshot_ccxt(sym, exchange=exchange, n_levels=n_levels)
            if row:
                yield row
        elapsed = time.time() - t0
        time.sleep(max(0.0, tick_sec - elapsed))


def get_l2_from_questdb(limit: int = 5000, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load L2 data from QuestDB and convert to common snapshot format.
    Requires QuestDB running and table l2_orderbook (from ccxt_l2_to_questdb.py).
    """
    if fetch_l2_orderbook is None:
        raise ImportError("QuestDB fetch not available; run from repo root or install requests/pandas.")
    df = fetch_l2_orderbook(symbol=symbol, limit=limit)
    if df is None or df.empty:
        return []
    # Table has timestamp, symbol, bid_1_p, bid_1_s, ..., ask_N_p, ask_N_s
    n = 0
    while f"bid_{n+1}_p" in df.columns:
        n += 1
    if n == 0:
        return []
    rows = []
    for _, r in df.iterrows():
        bids = [[float(r[f"bid_{i+1}_p"]), float(r[f"bid_{i+1}_s"])] for i in range(n)]
        asks = [[float(r[f"ask_{i+1}_p"]), float(r[f"ask_{i+1}_s"])] for i in range(n)]
        ts = r["timestamp"]
        if hasattr(ts, "timestamp"):
            ts = ts.timestamp()
        rows.append({
            "timestamp": ts,
            "symbol": str(r["symbol"]),
            "bids": bids,
            "asks": asks,
        })
    return rows
