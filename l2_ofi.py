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
LOOKAHEAD = 2           # T+2s Prediction
DEPTH_LEVELS = 5        # Look at top 5 levels of the book

data_buffer = []

# GLOBAL MEMORY
# Format: {'BTCUSDT': {'bids': [[p,q], [p,q]...], 'asks': [[p,q]...]}, ...}
previous_state = {}

# ==========================================
# 1. WEBSOCKET LOGIC (L2 OFI CAPTURE)
# ==========================================
def on_message(ws, message):
    global previous_state
    
    try:
        json_msg = json.loads(message)

        if 'data' in json_msg:
            payload = json_msg['data']
            stream_name = json_msg.get('stream', '')
            symbol = stream_name.split('@')[0].upper()
        else:
            payload = json_msg
            symbol = payload.get('s', 'UNKNOWN')

        if 'bids' in payload and 'asks' in payload:
            # Parse Top 5 Levels
            # payload['bids'] is a list of lists: [[price, qty], [price, qty], ...]
            cur_bids = [[float(p), float(q)] for p, q in payload['bids'][:DEPTH_LEVELS]]
            cur_asks = [[float(p), float(q)] for p, q in payload['asks'][:DEPTH_LEVELS]]
            
            # Mid Price based on Level 1
            mid_price = (cur_bids[0][0] + cur_asks[0][0]) / 2
            
            # Initialize state
            if symbol not in previous_state:
                previous_state[symbol] = {'bids': cur_bids, 'asks': cur_asks}
                return 

            # Calculate Integrated OFI (Multi-Level)
            prev_bids = previous_state[symbol]['bids']
            prev_asks = previous_state[symbol]['asks']
            
            ofi_value = calculate_integrated_ofi(prev_bids, cur_bids, prev_asks, cur_asks)
            
            # Update Memory
            previous_state[symbol] = {'bids': cur_bids, 'asks': cur_asks}
            
            # Store Data
            data_buffer.append({
                'timestamp': time.time(),
                'symbol': symbol,
                'mid_price': mid_price,
                'ofi': ofi_value
            })
            
    except Exception as e:
        # print(f"Error: {e}")
        pass

def calculate_integrated_ofi(prev_bids, cur_bids, prev_asks, cur_asks):
    """
    Calculates Multi-Level OFI (Integrated OFI).
    Sums the OFI of top 5 levels with decaying weights.
    Weight Formula: w = 1 / (level + 1)
    """
    total_net_flow = 0
    total_abs_flow = 0
    
    # Loop through each level (0 to 4)
    for i in range(DEPTH_LEVELS):
        # Safety check if book is thin
        if i >= len(prev_bids) or i >= len(cur_bids): break

        pb_price, pb_qty = prev_bids[i]
        cb_price, cb_qty = cur_bids[i]
        pa_price, pa_qty = prev_asks[i]
        ca_price, ca_qty = cur_asks[i]
        
        # 1. Bid Flow (Buying Pressure)
        if cb_price > pb_price:   b_flow = cb_qty
        elif cb_price < pb_price: b_flow = -pb_qty
        else:                     b_flow = cb_qty - pb_qty
            
        # 2. Ask Flow (Selling Pressure)
        if ca_price < pa_price:   a_flow = ca_qty
        elif ca_price > pa_price: a_flow = -pa_qty
        else:                     a_flow = ca_qty - pa_qty
        
        # 3. Apply Weighting (Decay)
        # Level 1: 1.0, Level 2: 0.5, Level 3: 0.33...
        weight = 1 / (i + 1)
        
        net_flow = (b_flow - a_flow) * weight
        abs_flow = (abs(b_flow) + abs(a_flow)) * weight
        
        total_net_flow += net_flow
        total_abs_flow += abs_flow

    # 4. Normalize (Ratio between -1 and 1)
    return total_net_flow / (total_abs_flow + 1e-9)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("\n### Stream Closed ###")

def on_open(ws):
    print(f"✅ Connected! Collecting L2 (Depth 5) OFI Data...")
    params = [f"{s}@depth5@100ms" for s in SYMBOLS]
    subscribe_msg = { "method": "SUBSCRIBE", "params": params, "id": 1 }
    ws.send(json.dumps(subscribe_msg))

# ==========================================
# 2. STATISTICAL BACKTEST ENGINE
# ==========================================
def analyze_performance():
    print(f"\nProcessing {len(data_buffer)} L2 OFI snapshots...", end="")
    
    if not data_buffer:
        print("❌ No data received.")
        return

    df = pd.DataFrame(data_buffer)
    df['time_sec'] = pd.to_datetime(df['timestamp'], unit='s').dt.round('1s')
    
    # Resample: Sum the flow over each second
    df_clean = df.groupby(['symbol', 'time_sec']).agg({
        'ofi': 'sum',
        'mid_price': 'last'
    }).reset_index()

    print(" Done!\n")
    print(f"{'SYMBOL':<10} | {'RANK IC':<10} | {'P-VALUE':<10} | {'IR':<8} | {'STATUS'}")
    print("-" * 65)

    ic_values = []
    ir_values = []

    for symbol in [s.upper() for s in SYMBOLS]:
        df_sym = df_clean[df_clean['symbol'] == symbol].copy().sort_values('time_sec')

        if len(df_sym) < 30: continue

        # Target: T+2 Returns
        df_sym['ret_future'] = df_sym['mid_price'].shift(-LOOKAHEAD) / df_sym['mid_price'] - 1
        df_final = df_sym.dropna()

        # Metrics
        ic, p_val = stats.spearmanr(df_final['ofi'], df_final['ret_future'])
        
        strategy_rets = np.sign(df_final['ofi']) * df_final['ret_future']
        ann_factor = np.sqrt(365 * 24 * 60 * 60 / LOOKAHEAD)
        
        if strategy_rets.std() == 0: ir = 0
        else: ir = (strategy_rets.mean() / strategy_rets.std()) * ann_factor

        if np.isnan(ic): continue

        ic_values.append(ic)
        ir_values.append(ir)
        
        status = "✅" if (ic > 0.03 and p_val < 0.05) else "❌"
        print(f"{symbol:<10} | {ic:.4f}     | {p_val:.4f}     | {ir:.2f}     | {status}")

    if ic_values:
        print("-" * 65)
        print(f"AVG RANK IC:          {sum(ic_values)/len(ic_values):.4f}")
        print(f"AVG INFO RATIO:       {sum(ir_values)/len(ir_values):.2f}")

# ==========================================
# 3. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    socket = "wss://stream.binance.com:9443/stream"
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    try:
        for i in range(DURATION_SECONDS):
            sys.stdout.write(f"\r⏳ Collecting L2 Data: {i+1}/{DURATION_SECONDS}s")
            sys.stdout.flush()
            time.sleep(1)
        print("\n⏳ Waiting for Lookahead Data...")
        time.sleep(LOOKAHEAD + 2)
    except KeyboardInterrupt:
        pass
    
    ws.close()
    analyze_performance()