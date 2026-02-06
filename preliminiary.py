import time
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from scipy.stats import spearmanr

# ----------------------------
# CONFIG
# ----------------------------
BASE_URL = "https://api.data.coindesk.com"   # <-- replace with your CoinDesk data base if different
ENDPOINT = "/spot/v2/historical/orderbook/l2/snapshots/minute"  # per docs page name  [1](https://developers.coindesk.com/documentation/data-api/spot_v2_historical_orderbook_l2_snapshots_minute)
API_KEY = "YOUR_COINDESK_API_KEY"

MARKET = "binance"  # exchange/market  [1](https://developers.coindesk.com/documentation/data-api/spot_v2_historical_orderbook_l2_snapshots_minute)
INSTRUMENTS = [
    "BTC-USDT","ETH-USDT","BNB-USDT","SOL-USDT","XRP-USDT",
    "ADA-USDT","DOGE-USDT","AVAX-USDT","MATIC-USDT","LTC-USDT"
]
LIMIT = 10       # 10 snapshots per instrument
DEPTH = 50       # request deeper book, we will use top N=20 for factor  [1](https://developers.coindesk.com/documentation/data-api/spot_v2_historical_orderbook_l2_snapshots_minute)
TOP_N = 20       # top N levels for OI
TIMEOUT = 20

# ----------------------------
# HELPERS
# ----------------------------
def auth_headers():
    return {
        "Accept": "application/json",
        "Authorization": f"Apikey {API_KEY}"  # adjust if your plan uses a different header scheme
    }

def now_ts_seconds():
    return int(datetime.now(timezone.utc).timestamp())

def fetch_snapshots_for_instrument(instrument: str, limit: int = LIMIT, to_ts: int = None, depth: int = DEPTH):
    """
    Calls CoinDesk Spot L2 snapshots minute endpoint for one instrument.
    API parameters per docs: market (required), instrument (required), limit, to_ts, depth, apply_mapping, response_format...
    [1](https://developers.coindesk.com/documentation/data-api/spot_v2_historical_orderbook_l2_snapshots_minute)
    """
    if to_ts is None:
        to_ts = now_ts_seconds()

    params = {
        "market": MARKET,
        "instrument": instrument,
        "limit": limit,
        "to_ts": to_ts,
        "depth": depth,
        "apply_mapping": True,            # get mapped instrument fields if available  [1](https://developers.coindesk.com/documentation/data-api/spot_v2_historical_orderbook_l2_snapshots_minute)
        "response_format": "JSON"
    }
    url = f"{BASE_URL}{ENDPOINT}"
    r = requests.get(url, headers=auth_headers(), params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    # Expect a top-level payload with Data array (see docs schema). [1](https://developers.coindesk.com/documentation/data-api/spot_v2_historical_orderbook_l2_snapshots_minute)
    # Each element corresponds to a snapshot at a minute boundary for that instrument.
    return data.get("Data", []) or data.get("data", [])  # handle case-insensitive variations

def extract_best_and_oi(snapshot: dict, top_n: int = TOP_N):
    """
    From a single snapshot record, pull best bid/ask and compute OI_N.
    CoinDesk snapshots include bids/asks arrays with price/quantity, at the minute boundary. [1](https://developers.coindesk.com/documentation/data-api/spot_v2_historical_orderbook_l2_snapshots_minute)
    This function assumes a record structure like:
      {..., "ORDERBOOK": {"BIDS":[{"P": price, "Q": qty}, ...], "ASKS":[...]}, ...}
    Adjust keys to match your plan's exact response schema.
    """
    # Try to find orderbook section (schema examples in docs show mapped fields; exact keys may vary by plan).
    ob = snapshot.get("ORDERBOOK") or snapshot.get("orderbook") or {}
    bids = ob.get("BIDS") or ob.get("bids") or []
    asks = ob.get("ASKS") or ob.get("asks") or []

    if not bids or not asks:
        return None

    # Sort just in case (highest bid first, lowest ask first)
    bids_sorted = sorted(bids, key=lambda x: float(x["P"]), reverse=True)
    asks_sorted = sorted(asks, key=lambda x: float(x["P"]))

    # Best of book:
    best_bid_p, best_bid_q = float(bids_sorted[0]["P"]), float(bids_sorted[0]["Q"])
    best_ask_p, best_ask_q = float(asks_sorted[0]["P"]), float(asks_sorted[0]["Q"])
    mid = (best_bid_p + best_ask_p) / 2.0

    # OI over top N
    bn = bids_sorted[:top_n]
    an = asks_sorted[:top_n]
    sum_bid = sum(float(lv["Q"]) for lv in bn)
    sum_ask = sum(float(lv["Q"]) for lv in an)
    denom = sum_bid + sum_ask
    oi = (sum_bid - sum_ask) / denom if denom > 0 else 0.0

    return {
        "best_bid_p": best_bid_p,
        "best_bid_q": best_bid_q,
        "best_ask_p": best_ask_p,
        "best_ask_q": best_ask_q,
        "mid": mid,
        "oi_topN": oi
    }

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    all_rows = []
    to_ts = now_ts_seconds()

    for inst in INSTRUMENTS:
        snaps = fetch_snapshots_for_instrument(inst, limit=LIMIT, to_ts=to_ts, depth=DEPTH)
        # Each snapshot should contain a minute timestamp field TIMESTAMP (seconds) per docs. [1](https://developers.coindesk.com/documentation/data-api/spot_v2_historical_orderbook_l2_snapshots_minute)
        for s in snaps:
            ts = s.get("TIMESTAMP") or s.get("timestamp")  # seconds
            features = extract_best_and_oi(s, top_n=TOP_N)
            if features and ts is not None:
                row = {
                    "instrument": inst,
                    "ts": int(ts),
                    **features
                }
                all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No data rows parsed. Check API key, base URL, endpoint path, and response schema.")

    df = pd.DataFrame(all_rows)

    # Make sure we have 10 snapshots per instrument and are sorted by time
    df.sort_values(["instrument", "ts"], inplace=True)

    # Compute forward 1-minute return per instrument using mid(t+1) - mid(t)
    df["mid_next"] = df.groupby("instrument")["mid"].shift(-1)
    df["ts_next"] = df.groupby("instrument")["ts"].shift(-1)
    df["dt"] = df["ts_next"] - df["ts"]  # seconds
    # keep only true consecutive minutes: dt == 60
    df = df[df["dt"] == 60].copy()
    df["ret_1m"] = (df["mid_next"] - df["mid"]) / df["mid"]

    # Cross-sectional RankIC per minute t (factor at t vs return t->t+1)
    ic_rows = []
    for ts, g in df.groupby("ts"):
        if g["instrument"].nunique() < 3:
            continue
        # ranks
        f_rank = g["oi_topN"].rank(method="average")
        r_rank = g["ret_1m"].rank(method="average")
        # Spearman = Pearson of ranks
        ic = f_rank.corr(r_rank)
        ic_rows.append({"ts": ts, "n": len(g), "rankic": ic})

    ic_df = pd.DataFrame(ic_rows).sort_values("ts")
    mean_ic = ic_df["rankic"].mean() if not ic_df.empty else float("nan")

    print("Per-minute RankIC (first few rows):")
    print(ic_df.head())
    print("\nMean RankIC over window:", mean_ic)

if __name__ == "__main__":
    main()
