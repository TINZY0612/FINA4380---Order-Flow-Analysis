"""
Collect L2 order book data via Binance WebSocket streams and ingest into QuestDB.

- Uses depth20@100ms streams (no REST polling), so no rate-limit or IP ban risk.
- Same table schema as before. Requires QuestDB on localhost:9000.

Usage:
  pip install websocket-client requests
  python ccxt_l2_to_questdb.py [--minutes 5]

QuestDB with Docker:
  See QUESTDB_DOCKER.md. Quick start: docker compose -f docker-compose.questdb.yml up -d

Table 'l2_orderbook' is created automatically. Columns:
  timestamp (designated), symbol (SYMBOL), bid_1_p, bid_1_s, ..., ask_20_p, ask_20_s.

Example query:
  SELECT timestamp, symbol, bid_1_p, ask_1_p FROM l2_orderbook WHERE symbol = 'BTCUSDT' LIMIT 10;
"""
import time
import sys
import json
import argparse
import threading
import ssl

import websocket
import requests

# ==========================================
# CONFIG
# ==========================================
L2_LEVELS = 20
L2_DURATION_MINUTES = 5
TABLE_NAME = "l2_orderbook"
L2_ILP_BATCH_SIZE = 500

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "MATIC/USDT",
]

# Binance WebSocket (spot)
WS_BASE = "wss://stream.binance.com:9443"

QUESTDB_HOST = "http://localhost:9000"
QUESTDB_WRITE_URL = f"{QUESTDB_HOST}/write"
QUESTDB_WRITE_URL_V2 = f"{QUESTDB_HOST}/api/v2/write"


def build_ob_columns(bids, asks, levels=L2_LEVELS):
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


def write_l2_ilp(rows: list, url: str = QUESTDB_WRITE_URL, verbose: bool = False) -> None:
    """Send L2 snapshot rows to QuestDB via HTTP ILP. rows = list of (ts_ns, sym, cols_dict)."""
    if not rows:
        return
    lines = []
    for ts_ns, sym, cols in rows:
        parts = [f"{k}={v}" for k, v in cols.items()]
        line = f"{TABLE_NAME},symbol={sym} " + ",".join(parts) + f" {ts_ns}"
        lines.append(line)
    body = "\n".join(lines) + "\n"
    r = requests.post(url, data=body, headers={"Content-Type": "text/plain"}, timeout=60)
    if r.status_code == 404 and url == QUESTDB_WRITE_URL:
        r = requests.post(
            QUESTDB_WRITE_URL_V2, data=body, headers={"Content-Type": "text/plain"}, timeout=60
        )
        if r.status_code == 200 and verbose:
            print("    QuestDB L2 write: /api/v2/write HTTP 200")
    if verbose or r.status_code != 200:
        print(f"    QuestDB L2 write: HTTP {r.status_code} -> {r.text[:200] if r.text else '(empty)'}")
    r.raise_for_status()


def run_ws_collection(duration_seconds: int):
    """Run WebSocket collection for duration_seconds. Returns (total_rows, final_batch)."""
    symbols_lower = [s.replace("/", "").lower() for s in SYMBOLS]
    streams = [f"{s}@depth20@100ms" for s in symbols_lower]
    url = f"{WS_BASE}/stream?streams={'/'.join(streams)}"

    state = {
        "batch": [],
        "total": 0,
        "first_write_done": False,
        "start": time.perf_counter(),
        "last_progress": 0,
    }

    def on_message(ws, message):
        try:
            msg = json.loads(message)
            payload = msg.get("data", msg)
            stream_name = msg.get("stream", "")
            symbol = stream_name.split("@")[0].upper()
            bids = payload.get("bids", [])[:L2_LEVELS]
            asks = payload.get("asks", [])[:L2_LEVELS]
            if not bids and not asks:
                return
            if "E" in payload:
                ts_ns = int(payload["E"] * 1_000_000)
            else:
                ts_ns = int(time.time() * 1_000_000_000)
            cols = build_ob_columns(bids, asks)
            state["batch"].append((ts_ns, symbol, cols))
            state["total"] += 1
            if len(state["batch"]) >= L2_ILP_BATCH_SIZE:
                write_l2_ilp(state["batch"], verbose=not state["first_write_done"])
                state["first_write_done"] = True
                state["batch"] = []
            now = time.perf_counter()
            if now - state["last_progress"] >= 5.0:
                state["last_progress"] = now
                sys.stdout.write(f"\r  Received {state['total']} L2 snapshots...")
                sys.stdout.flush()
        except Exception as e:
            print(f"  on_message error: {e}", file=sys.stderr)

    def on_error(ws, error):
        print(f"  WebSocket error: {error}", file=sys.stderr)

    def on_close(ws, close_status_code, close_msg):
        pass

    def on_open(ws):
        print("  WebSocket connected. Streaming depth20@100ms.\n")

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    wst = threading.Thread(
        target=ws.run_forever,
        kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
    )
    wst.daemon = True
    wst.start()

    # Wait for duration
    for _ in range(duration_seconds):
        time.sleep(1)
    ws.close()
    time.sleep(0.5)
    return state["total"], state["batch"]


def main():
    p = argparse.ArgumentParser(description="CCXT → QuestDB: L2 order book (WebSocket)")
    p.add_argument("--minutes", type=int, default=L2_DURATION_MINUTES, help=f"Collection duration in minutes (default {L2_DURATION_MINUTES})")
    args = p.parse_args()

    duration_seconds = args.minutes * 60

    print(f"Collecting L2 order book via WebSocket: {len(SYMBOLS)} symbol(s), depth20@100ms, "
          f"for {args.minutes} min.")
    print(f"QuestDB: {QUESTDB_HOST}  table: {TABLE_NAME}\n")

    try:
        total_rows, final_batch = run_ws_collection(duration_seconds)
        if final_batch:
            write_l2_ilp(final_batch, verbose=True)

        print(f"\nDone. Total rows: {total_rows}")

        r = requests.get(
            f"{QUESTDB_HOST}/exec",
            params={"query": f"SELECT count(*) AS n FROM {TABLE_NAME}"},
            timeout=10,
        )
        if r.status_code == 200:
            d = r.json()
            if "error" not in d and d.get("dataset"):
                n = d["dataset"][0][0]
                print(f"QuestDB {TABLE_NAME}: {n} rows total")
        print(f"\nWeb console: http://localhost:9000  →  SELECT * FROM {TABLE_NAME} LIMIT 10;")
    except requests.RequestException as e:
        print(f"QuestDB HTTP error: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.status_code} {e.response.text[:300]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
