import websocket
import json
import pandas as pd
import numpy as np
import time
import threading
import scipy.stats as stats
import sys
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOLS = [
    'btcusdt', 'ethusdt', 'bnbusdt', 'solusdt', 'xrpusdt', 
    'adausdt', 'dogeusdt', 'suiusdt', 'asterusdt', 'avaxusdt'
]

DURATION_SECONDS = 120  # Collect for 2 minutes
LOOKAHEAD = 2           # T+2s Prediction (Based on your Lead-Lag Analysis)

data_buffer = []

# GLOBAL MEMORY (Dictionary to handle multiple symbols)
# Format: {'BTCUSDT': {'bid': [price, qty], 'ask': [price, qty]}, ...}
previous_state = {}

# ==========================================
# 1. WEBSOCKET LOGIC (OFI CAPTURE)
# ==========================================
def on_message(ws, message):
    global previous_state
    
    try:
        json_msg = json.loads(message)

        # 1. Unwrap Data
        if 'data' in json_msg:
            payload = json_msg['data']
            stream_name = json_msg.get('stream', '')
            symbol = stream_name.split('@')[0].upper()
        else:
            payload = json_msg
            symbol = payload.get('s', 'UNKNOWN')

        # 2. Parse Bids/Asks (L1 Data is sufficient for OFI)
        if 'bids' in payload and 'asks' in payload:
            cur_bid = [float(payload['bids'][0][0]), float(payload['bids'][0][1])]
            cur_ask = [float(payload['asks'][0][0]), float(payload['asks'][0][1])]
            mid_price = (cur_bid[0] + cur_ask[0]) / 2
            
            # Initialize state for this symbol if new
            if symbol not in previous_state:
                previous_state[symbol] = {'bid': cur_bid, 'ask': cur_ask}
                return # Skip first tick (cannot calc flow)

            # 3. Calculate OFI
            prev_bid = previous_state[symbol]['bid']
            prev_ask = previous_state[symbol]['ask']
            
            ofi_value = calculate_ofi(prev_bid, cur_bid, prev_ask, cur_ask)
            
            # 4. Update Memory
            previous_state[symbol] = {'bid': cur_bid, 'ask': cur_ask}
            
            # 5. Store Data
            data_buffer.append({
                'timestamp': time.time(),
                'symbol': symbol,
                'mid_price': mid_price,
                'ofi': ofi_value
            })
            
    except Exception as e:
        # print(f"Error: {e}")
        pass

def calculate_ofi(prev_bid, cur_bid, prev_ask, cur_ask):
    """
    Calculates Net Order Flow Imbalance (OFI).
    OFI = Bid Flow - Ask Flow
    """
    # Bid Flow (Buying Pressure)
    if cur_bid[0] > prev_bid[0]:   bid_flow = cur_bid[1]
    elif cur_bid[0] < prev_bid[0]: bid_flow = -prev_bid[1]
    else:                          bid_flow = cur_bid[1] - prev_bid[1]

    # Ask Flow (Selling Pressure)
    if cur_ask[0] < prev_ask[0]:   ask_flow = cur_ask[1]
    elif cur_ask[0] > prev_ask[0]: ask_flow = -prev_ask[1]
    else:                          ask_flow = cur_ask[1] - prev_ask[1]

    return bid_flow - ask_flow

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("\n### Stream Closed ###")

def on_open(ws):
    print(f"✅ Connected! Collecting OFI Data for {len(SYMBOLS)} pairs...")
    # Subscribe to Top 5 Depth for all symbols (Depth 5 is standard for L1/OFI)
    params = [f"{s}@depth5@100ms" for s in SYMBOLS]
    subscribe_msg = {
        "method": "SUBSCRIBE",
        "params": params,
        "id": 1
    }
    ws.send(json.dumps(subscribe_msg))

# ==========================================
# 2. STATISTICAL BACKTEST ENGINE
# ==========================================
def analyze_performance():
    print(f"\nProcessing {len(data_buffer)} OFI snapshots...", end="")
    
    if not data_buffer:
        print("❌ No data received.")
        return

    df = pd.DataFrame(data_buffer)
    df['time_sec'] = pd.to_datetime(df['timestamp'], unit='s').dt.round('1s')
    
    # RESAMPLE Logic:
    # OBI is a State -> We take the MEAN.
    # OFI is a Flow -> We take the SUM (Accumulate flow over the second).
    df_clean = df.groupby(['symbol', 'time_sec']).agg({
        'ofi': 'sum',        # <--- KEY CHANGE: Summing the flow
        'mid_price': 'last'
    }).reset_index()

    print(" Done!\n")
    print(f"{'SYMBOL':<10} | {'RANK IC':<10} | {'P-VALUE':<10} | {'IR':<8} | {'STATUS'}")
    print("-" * 65)

    ic_values = []
    ir_values = []

    for symbol in [s.upper() for s in SYMBOLS]:
        df_sym = df_clean[df_clean['symbol'] == symbol].copy().sort_values('time_sec')

        if len(df_sym) < 30:
            continue

        # Target: T+2 Returns (Based on your analysis)
        df_sym['ret_future'] = df_sym['mid_price'].shift(-LOOKAHEAD) / df_sym['mid_price'] - 1
        df_final = df_sym.dropna()

        # 1. Rank IC (Predictive Power)
        ic, p_val = stats.spearmanr(df_final['ofi'], df_final['ret_future'])
        
        # 2. Information Ratio (Risk-Adjusted Return)
        # Strategy: Trade proportional to Signal (OFI)
        # Note: We normalize OFI to act as a weight
        strategy_rets = np.sign(df_final['ofi']) * df_final['ret_future']
        ann_factor = np.sqrt(365 * 24 * 60 * 60 / LOOKAHEAD) # Annualize based on trading freq
        if strategy_rets.std() == 0:
            ir = 0
        else:
            ir = (strategy_rets.mean() / strategy_rets.std()) * ann_factor

        if np.isnan(ic): continue

        ic_values.append(ic)
        ir_values.append(ir)
        
        status = "✅" if (ic > 0.03 and p_val < 0.05) else "❌"
        print(f"{symbol:<10} | {ic:.4f}     | {p_val:.4f}     | {ir:.2f}     | {status}")

    if ic_values:
        print("-" * 65)
        print(f"AVERAGE RANK IC:      {sum(ic_values)/len(ic_values):.4f}")
        print(f"AVERAGE INFO RATIO:   {sum(ir_values)/len(ir_values):.2f}")

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
            sys.stdout.write(f"\r⏳ Collecting OFI Data: {i+1}/{DURATION_SECONDS}s")
            sys.stdout.flush()
            time.sleep(1)
        print("\n⏳ Waiting for Lookahead Data...")
        time.sleep(LOOKAHEAD + 2)
    except KeyboardInterrupt:
        pass
    
    ws.close()
    analyze_performance()