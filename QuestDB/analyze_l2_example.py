"""
Example: pull L2 orderbook data from QuestDB and run simple analysis.

Run after QuestDB has data (from ccxt_l2_to_questdb.py).
  pip install requests pandas
  python analyze_l2_example.py
"""
from questdb_fetch import fetch_l2_orderbook, query_questdb
import pandas as pd

# 1. Load data (one symbol, last 10k rows)
df = fetch_l2_orderbook(symbol="BTCUSDT", limit=10_000)
print(f"Loaded {len(df)} rows")
print(df[["timestamp", "symbol", "bid_1_p", "ask_1_p"]].head())

# 2. Mid price and spread
df["mid"] = (df["bid_1_p"] + df["ask_1_p"]) / 2
df["spread"] = df["ask_1_p"] - df["bid_1_p"]
df["spread_bps"] = 10_000 * df["spread"] / df["mid"]

# 3. Simple stats
print("\nMid price (last):", df["mid"].iloc[-1])
print("Spread (bps) – mean:", df["spread_bps"].mean(), "std:", df["spread_bps"].std())

# 4. Or use raw SQL for aggregation in the DB
summary = query_questdb("""
    SELECT symbol, count(*) AS ticks, min(timestamp) AS first_ts, max(timestamp) AS last_ts
    FROM l2_orderbook
    GROUP BY symbol
""")
print("\nTicks per symbol:")
print(summary)
