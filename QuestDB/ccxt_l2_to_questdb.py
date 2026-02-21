"""
Collect 5 minutes of L2 order book data at 100ms tick from CCXT (REST poll)
and ingest into QuestDB.

Requirements:
  pip install ccxt questdb

QuestDB with Docker (persistent data):
  See QUESTDB_DOCKER.md. Quick start:
  docker compose -f docker-compose.questdb.yml up -d
  Then run this script. Data is stored in volume questdb_data.

Table 'l2_orderbook' is created automatically. Columns:
  timestamp (designated), symbol (SYMBOL), bid_1_p, bid_1_s, ..., ask_5_p, ask_5_s.

Example query (after run):
  SELECT timestamp, symbol, bid_1_p, ask_1_p FROM l2_orderbook WHERE symbol = 'BTCUSDT' LIMIT 10;
"""
import time
import sys
from datetime import datetime, timezone

import ccxt
from questdb.ingress import Sender, IngressError

# ==========================================
# CONFIG
# ==========================================
TICK_MS = 100
DURATION_SECONDS = 5 * 60  # 5 minutes
LEVELS = 20  # top N bid/ask levels to store per side

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
]  # add more if desired; more symbols = more API calls per tick

# QuestDB ILP HTTP (no auth by default)
QUESTDB_CONF = "http::addr=localhost:9000;"
TABLE_NAME = "l2_orderbook"

# ==========================================
# CCXT + QuestDB
# ==========================================
def build_ob_columns(bids, asks, levels=LEVELS):
    """Build flat columns for top `levels` bids and asks. Fill with 0 if fewer levels."""
    cols = {}
    for i in range(levels):
        if i < len(bids):
            cols[f"bid_{i+1}_p"] = float(bids[i][0])
            cols[f"bid_{i+1}_s"] = float(bids[i][1])
        else:
            cols[f"bid_{i+1}_p"] = 0.0
            cols[f"bid_{i+1}_s"] = 0.0
    for i in range(levels):
        if i < len(asks):
            cols[f"ask_{i+1}_p"] = float(asks[i][0])
            cols[f"ask_{i+1}_s"] = float(asks[i][1])
        else:
            cols[f"ask_{i+1}_p"] = 0.0
            cols[f"ask_{i+1}_s"] = 0.0
    return cols


def main():
    exchange = ccxt.binance({"enableRateLimit": True})
    tick_sec = TICK_MS / 1000.0
    num_ticks = int(DURATION_SECONDS / tick_sec)
    total_rows = 0

    print(f"Collecting L2 order book: {len(SYMBOLS)} symbol(s), {LEVELS} levels, "
          f"every {TICK_MS}ms for {DURATION_SECONDS}s ({num_ticks} ticks).")
    print(f"QuestDB: {QUESTDB_CONF}  table: {TABLE_NAME}")
    print()

    try:
        with Sender.from_conf(QUESTDB_CONF) as sender:
            start_wall = time.perf_counter()
            for tick in range(num_ticks):
                tick_start = time.perf_counter()
                ts_ns = int(time.time() * 1_000_000_000)
                ts_dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)

                for symbol in SYMBOLS:
                    try:
                        ob = exchange.fetch_order_book(symbol, limit=LEVELS)
                        bids = ob.get("bids", [])[:LEVELS]
                        asks = ob.get("asks", [])[:LEVELS]
                        cols = build_ob_columns(bids, asks)
                        # Symbol: normalize to no slash for QuestDB symbol type
                        sym = symbol.replace("/", "")
                        sender.row(
                            TABLE_NAME,
                            symbols={"symbol": sym},
                            columns=cols,
                            at=ts_dt,
                        )
                        total_rows += 1
                    except Exception as e:
                        print(f"  [{symbol}] fetch error: {e}", file=sys.stderr)

                elapsed = time.perf_counter() - tick_start
                sleep_sec = max(0.0, tick_sec - elapsed)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

                if (tick + 1) % 100 == 0 or tick == 0:
                    sys.stdout.write(f"\r  Tick {tick + 1}/{num_ticks}  rows={total_rows}")
                    sys.stdout.flush()

            sender.flush()

        wall = time.perf_counter() - start_wall
        print(f"\nDone. Total rows: {total_rows}  wall time: {wall:.1f}s")
    except IngressError as e:
        print(f"QuestDB error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
