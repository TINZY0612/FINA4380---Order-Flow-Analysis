import websocket
import json
import pandas as pd
import numpy as np
import time
import threading
import scipy.stats as stats
import sys

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOLS = [
    'btcusdt', 'ethusdt', 'bnbusdt', 'solusdt', 'xrpusdt', 
    'adausdt', 'dogeusdt', 'suiusdt', 'asterusdt', 'avaxusdt'
]

DURATION_SECONDS = 120  # Collect for 2 minutes
LOOKAHEAD = 5           # Predict T+5 seconds
DEPTH_LEVELS = 20       # Use Top 20 Bids/Asks

data_buffer = []

# ==========================================
# 1. WEBSOCKET LOGIC (L2 DEPTH)
# ==========================================
def on_message(ws, message):
    try:
        json_msg = json.loads(message)

        # 1. UNWRAP THE DATA
        # If the message comes from a multiplex stream, the real data is inside 'data'
        if 'data' in json_msg:
            payload = json_msg['data']
            # We also need to grab the symbol from the stream name if it's not in the data
            # Format: btcusdt@depth20@100ms -> btcusdt
            stream_name = json_msg.get('stream', '')
            symbol = stream_name.split('@')[0].upper() 
        else:
            payload = json_msg
            # Fallback: try to find symbol in payload, or hardcode if single stream
            symbol = payload.get('s', 'UNKNOWN')

        # DEBUG: Print first valid payload to confirm structure
        if len(data_buffer) == 0:
            print(f"\nExample Payload: {str(payload)[:100]}...")

        # 2. CHECK FOR BIDS/ASKS IN THE UNWRAPPED PAYLOAD
        if 'bids' in payload and 'asks' in payload:
            bids = np.array(payload['bids'], dtype=float)
            asks = np.array(payload['asks'], dtype=float)
            
            # 3. Calculate Mid Price
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            
            # 4. Calculate Weighted OBI
            w_obi = calculate_weighted_obi(bids, asks, mid_price)
            
            record = {
                'timestamp': time.time(),
                'symbol': symbol,
                'mid_price': mid_price,
                'l2_obi': w_obi
            }
            data_buffer.append(record)
            
    except Exception as e:
        # print(f"Error: {e}") 
        pass

def calculate_weighted_obi(bids, asks, mid_price):
    """
    Calculates Imbalance weighted by 'Distance to Price'.
    Closer orders = Higher Weight.
    """
    # 1. Calculate Distances (Percentage away from Mid)
    # Add epsilon 1e-9 to avoid division by zero
    bid_dist = (mid_price - bids[:, 0]) / mid_price
    ask_dist = (asks[:, 0] - mid_price) / mid_price
    
    # 2. Calculate Inverse Weights (Closer = Higher Value)
    # We use exp(-k * distance) to decay weight rapidly
    # Decay factor k=50 means an order 1% away loses ~40% influence
    decay = 50 
    bid_weights = np.exp(-decay * bid_dist)
    ask_weights = np.exp(-decay * ask_dist)
    
    # 3. Sum Weighted Volumes
    total_bid_pressure = np.sum(bids[:, 1] * bid_weights)
    total_ask_pressure = np.sum(asks[:, 1] * ask_weights)
    
    # 4. Final Formula
    if (total_bid_pressure + total_ask_pressure) == 0:
        return 0
        
    return (total_bid_pressure - total_ask_pressure) / (total_bid_pressure + total_ask_pressure)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("\n### Stream Closed ###")

def on_open(ws):
    print(f"✅ Connected to L2 Streams! Subscribing to {len(SYMBOLS)} pairs...")
    
    # FORMAT FOR MULTIPLEX STREAM: <symbol>@depth20@100ms
    # We combine them into one URL connection string usually, 
    # but here we use the subscribe method.
    params = [f"{s}@depth20@100ms" for s in SYMBOLS]
    subscribe_msg = {
        "method": "SUBSCRIBE",
        "params": params,
        "id": 1
    }
    ws.send(json.dumps(subscribe_msg))

# ==========================================
# 2. DATA PROCESSING ENGINE
# ==========================================
def calculate_rank_ic():
    print(f"\nProcessing {len(data_buffer)} L2 snapshots...", end="")
    
    if not data_buffer:
        print("❌ No data received.")
        return

    df = pd.DataFrame(data_buffer)
    df['time_sec'] = pd.to_datetime(df['timestamp'], unit='s').dt.round('1s')
    
    # Resample to 1 second
    df_clean = df.groupby(['symbol', 'time_sec']).agg({
        'l2_obi': 'mean',
        'mid_price': 'last'
    }).reset_index()

    print(" Done!\n")
    print(f"{'SYMBOL':<10} | {'L2 RANK IC':<12} | {'P-VALUE':<10} | {'SAMPLES':<8}")
    print("-" * 55)

    ic_values = []

    for symbol in [s.upper() for s in SYMBOLS]:
        df_sym = df_clean[df_clean['symbol'] == symbol].copy().sort_values('time_sec')

        if len(df_sym) < 30:
            print(f"{symbol:<10} | {'Not enough':<12} | {'-':<10} | {len(df_sym)}")
            continue

        # Target: T+5 Returns
        df_sym['ret_5s'] = df_sym['mid_price'].shift(-LOOKAHEAD) / df_sym['mid_price'] - 1
        df_final = df_sym.dropna()

        # Rank IC
        ic, p_val = stats.spearmanr(df_final['l2_obi'], df_final['ret_5s'])
        
        if np.isnan(ic): continue

        ic_values.append(ic)
        status = "✅" if (ic > 0.05 and p_val < 0.05) else ""
        
        print(f"{symbol:<10} | {ic:.4f} {status:<4} | {p_val:.4f}     | {len(df_final)}")

    if ic_values:
        print("-" * 55)
        print(f"AVERAGE L2 IC: {sum(ic_values)/len(ic_values):.4f}")

# ==========================================
# 3. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    # NOTE: For multiple streams, we typically use the combined stream URL 
    # OR we use the subscribe method on the base URL.
    # The safest way is to use the base URL and send subscribe.
    socket = "wss://stream.binance.com:9443/stream" # Changed to /stream for multiplexing
    
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    try:
        for i in range(DURATION_SECONDS):
            sys.stdout.write(f"\r⏳ Collecting L2 Data: {i+1}/{DURATION_SECONDS}s")
            sys.stdout.flush()
            time.sleep(1)
        print("\n⏳ Waiting for T+5...")
        time.sleep(LOOKAHEAD + 2)
    except KeyboardInterrupt:
        pass
    
    ws.close()
    calculate_rank_ic()