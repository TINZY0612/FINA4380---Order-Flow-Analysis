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

DURATION_SECONDS = 120   # Collect for 2 minutes
LOOKAHEAD = 1            # T+1s Prediction

data_buffer = []

# GLOBAL MEMORY
# We need to accumulate volume for each second
current_second_vol = {} 

def init_vol_memory():
    global current_second_vol
    for s in SYMBOLS:
        current_second_vol[s.upper()] = {'buy_vol': 0.0, 'sell_vol': 0.0}

init_vol_memory()

# ==========================================
# 1. WEBSOCKET LOGIC (TRADE CAPTURE)
# ==========================================
def on_message(ws, message):
    global current_second_vol
    
    try:
        json_msg = json.loads(message)

        # Handle AggTrade Stream
        if 'data' in json_msg:
            payload = json_msg['data']
            stream_name = json_msg.get('stream', '')
            symbol = stream_name.split('@')[0].upper()
        else:
            payload = json_msg
            symbol = payload.get('s', 'UNKNOWN')

        # Parse Trade Data
        if 'p' in payload and 'q' in payload:
            price = float(payload['p'])
            qty = float(payload['q'])
            is_buyer_maker = payload['m'] # True = Sell, False = Buy
            
            # Update Volume Buckets
            if symbol in current_second_vol:
                if is_buyer_maker:
                    current_second_vol[symbol]['sell_vol'] += qty
                else:
                    current_second_vol[symbol]['buy_vol'] += qty
            
            # Store Snapshot (We store every trade, but will resample later)
            # Note: TFI is a flow, so we don't need "mid_price" from the book here.
            # But we NEED price to calculate returns. 
            # We use the Trade Price as the proxy for Price.
            
            data_buffer.append({
                'timestamp': time.time(),
                'symbol': symbol,
                'price': price,
                'buy_vol': qty if not is_buyer_maker else 0,
                'sell_vol': qty if is_buyer_maker else 0
            })
            
    except Exception as e:
        # print(f"Error: {e}")
        pass

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("\n### Stream Closed ###")

def on_open(ws):
    print(f"✅ Connected! Collecting Trade Flow (TFI) Data...")
    # Subscribe to AggTrade (Real-time executions)
    params = [f"{s}@aggTrade" for s in SYMBOLS]
    subscribe_msg = { "method": "SUBSCRIBE", "params": params, "id": 1 }
    ws.send(json.dumps(subscribe_msg))

def analyze_tfi_lead_lag(df):
    """
    Analyzes how far ahead Trade Flow (TFI) predicts price changes.
    """
    print("\n--- RUNNING TFI LEAD-LAG ANALYSIS ---")
    
    lags = []
    correlations = []
    # We test lags from 1s to 10s (Since data is 1s buckets, we step by 1)
    max_lag_seconds = 10
    
    for i in range(1, max_lag_seconds + 1):
        # Target: Future Return (Price at T+Lag / Price at T - 1)
        future_returns = df['price'].shift(-i) / df['price'] - 1
        
        # Clean Data
        temp_df = pd.DataFrame({
            'signal': df['tfi'], 
            'target': future_returns
        }).dropna()
        
        if len(temp_df) < 30: continue

        # Calculate Rank IC (Spearman)
        corr, _ = stats.spearmanr(temp_df['signal'], temp_df['target'])
        
        lags.append(i)
        correlations.append(corr)
        print(f"Lag {i}s | Correlation: {corr:.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(lags, correlations, marker='o', color='orange', label='TFI Predictive Power')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.title('TFI Lead-Lag Analysis')
    plt.xlabel('Seconds into Future')
    plt.ylabel('Rank IC')
    plt.grid(True, alpha=0.3)
    plt.savefig('tfi_lead_lag.png')
    print("📊 Chart saved as 'tfi_lead_lag.png'")

# ==========================================
# 2. STATISTICAL BACKTEST ENGINE
# ==========================================
def analyze_performance():
    print(f"\nProcessing {len(data_buffer)} Trade snapshots...", end="")
    
    if not data_buffer:
        print("❌ No data received.")
        return

    df = pd.DataFrame(data_buffer)
    df['time_sec'] = pd.to_datetime(df['timestamp'], unit='s').dt.round('1s')
    
    # Resample: Sum Volumes per Second
    df_clean = df.groupby(['symbol', 'time_sec']).agg({
        'buy_vol': 'sum',
        'sell_vol': 'sum',
        'price': 'last'  # Use last trade price of the second
    }).reset_index()
    
    # CALCULATE TFI (Normalized)
    # TFI = (Buy - Sell) / (Buy + Sell)
    df_clean['total_vol'] = df_clean['buy_vol'] + df_clean['sell_vol']
    df_clean['tfi'] = (df_clean['buy_vol'] - df_clean['sell_vol']) / (df_clean['total_vol'] + 1e-9)

    print(" Done!\n")
    print(f"{'SYMBOL':<10} | {'RANK IC':<10} | {'P-VALUE':<10} | {'IR':<8} | {'STATUS'}")
    print("-" * 65)

    ic_values = []
    ir_values = []

    for symbol in [s.upper() for s in SYMBOLS]:
        df_sym = df_clean[df_clean['symbol'] == symbol].copy().sort_values('time_sec')

        if len(df_sym) < 30: continue

        # Target: T+2 Returns
        df_sym['ret_future'] = df_sym['price'].shift(-LOOKAHEAD) / df_sym['price'] - 1
        df_final = df_sym.dropna()

        # Metrics
        ic, p_val = stats.spearmanr(df_final['tfi'], df_final['ret_future'])
        
        strategy_rets = np.sign(df_final['tfi']) * df_final['ret_future']
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

    analyze_tfi_lead_lag(df_clean)

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
            sys.stdout.write(f"\r⏳ Collecting Trades: {i+1}/{DURATION_SECONDS}s")
            sys.stdout.flush()
            time.sleep(1)
        print("\n⏳ Waiting for Lookahead Data...")
        time.sleep(LOOKAHEAD + 2)
    except KeyboardInterrupt:
        pass
    
    ws.close()
    analyze_performance()