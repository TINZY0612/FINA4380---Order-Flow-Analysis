import websocket
import json
import pandas as pd
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats 

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = 'ethusdt'      
DURATION = 60           # Collect data for 60 seconds

data_buffer = []

# GLOBAL MEMORY (Required for OFI)
# We need to remember the previous state to calculate "Flow"
prev_best_bid = None # Format: [Price, Qty]
prev_best_ask = None # Format: [Price, Qty]

# ==========================================
# 1. WEBSOCKET LOGIC (WITH MEMORY)
# ==========================================
def on_message(ws, message):
    global prev_best_bid, prev_best_ask
    
    try:
        json_msg = json.loads(message)
        
        # Unwrap Data (Binance Multiplex format)
        if 'data' in json_msg:
            payload = json_msg['data']
        else:
            payload = json_msg

        if 'bids' in payload and 'asks' in payload:
            # Parse Current L1 Data (Top of Book)
            # OFI is calculated on the Best Bid and Best Ask
            cur_best_bid = [float(payload['bids'][0][0]), float(payload['bids'][0][1])]
            cur_best_ask = [float(payload['asks'][0][0]), float(payload['asks'][0][1])]
            mid_price = (cur_best_bid[0] + cur_best_ask[0]) / 2
            
            # ----------------------------------------------
            # CALCULATE OFI (Requires Previous State)
            # ----------------------------------------------
            if prev_best_bid is None:
                # First snapshot: We can't calculate flow yet.
                # Just save state and wait for next tick.
                ofi_value = 0
            else:
                # Call the OFI math engine
                ofi_value = calculate_ofi(prev_best_bid, cur_best_bid, prev_best_ask, cur_best_ask)
            
            # Update Memory for Next Loop
            prev_best_bid = cur_best_bid
            prev_best_ask = cur_best_ask
            
            # Store Data
            record = {
                'timestamp': time.time(),
                'mid_price': mid_price,
                'ofi': ofi_value
            }
            data_buffer.append(record)
            
    except Exception as e:
        # print(e) # Uncomment for debugging
        pass

def calculate_ofi(prev_bid, cur_bid, prev_ask, cur_ask):
    """
    Calculates Order Flow Imbalance (OFI).
    Logic: Tracks changes in the Best Bid and Best Ask levels.
    Formula based on Cont, Kukanov, Stoikov (2014).
    """
    # 1. Calculate Bid Flow (Buying Pressure)
    # If Bid Price moved UP, it's strong buying (Add full volume)
    if cur_bid[0] > prev_bid[0]:
        bid_flow = cur_bid[1]
    # If Bid Price moved DOWN, previous buyers canceled (Subtract previous volume)
    elif cur_bid[0] < prev_bid[0]:
        bid_flow = -prev_bid[1]
    # If Bid Price is SAME, look at volume change
    else:
        bid_flow = cur_bid[1] - prev_bid[1]

    # 2. Calculate Ask Flow (Selling Pressure)
    # If Ask Price moved DOWN, it's strong selling (Add full volume)
    if cur_ask[0] < prev_ask[0]:
        ask_flow = cur_ask[1]
    # If Ask Price moved UP, sellers retreated (Subtract previous volume)
    elif cur_ask[0] > prev_ask[0]:
        ask_flow = -prev_ask[1]
    # If Ask Price is SAME, look at volume change
    else:
        ask_flow = cur_ask[1] - prev_ask[1]

    # 3. Net Order Flow
    # OFI = Bid Flow - Ask Flow
    return bid_flow - ask_flow

def on_open(ws):
    print(f"✅ Connected! Collecting OFI Data for {SYMBOL.upper()}...")
    # We use depth5 because OFI only needs the top level (L1), 
    # but depth5 is the smallest standard stream.
    params = [f"{SYMBOL}@depth5@100ms"] 
    subscribe_msg = { "method": "SUBSCRIBE", "params": params, "id": 1 }
    ws.send(json.dumps(subscribe_msg))

def analyze_lead_lag(df):
    """
    Analyzes how far ahead OFI predicts price changes.
    Generates a second chart: 'lead_lag_analysis.png'
    """
    print("\n--- RUNNING LEAD-LAG ANALYSIS ---")
    
    # 1. Setup Parameters
    # We test lags from 0.0s to 5.0s (50 steps of 100ms)
    lags = []
    correlations = []
    max_lag_steps = 50 
    
    # 2. Loop through future times
    for i in range(max_lag_steps):
        lag_seconds = i / 10.0  # 0.1s, 0.2s, ... 5.0s
        
        # Target: Future Return (Price at T+Lag / Price at T - 1)
        # We shift price BACKWARDS (-i) to align Future Price with Current OFI
        future_returns = df['mid_price'].shift(-i) / df['mid_price'] - 1
        
        # Clean Data (Remove NaNs created by shift)
        temp_df = pd.DataFrame({
            'signal': df['ofi'], 
            'target': future_returns
        }).dropna()
        
        if len(temp_df) < 30: continue

        # Calculate Rank IC (Spearman)
        corr, _ = stats.spearmanr(temp_df['signal'], temp_df['target'])
        
        lags.append(lag_seconds)
        correlations.append(corr)

    # 3. Find the Winner
    if not correlations:
        print("❌ Not enough data for Lead-Lag analysis.")
        return

    peak_corr = max(correlations)
    peak_lag = lags[correlations.index(peak_corr)]
    
    print(f"✅ Analysis Complete!")
    print(f"👉 Peak Correlation: {peak_corr:.4f}")
    print(f"👉 Optimal Lead Time: {peak_lag} seconds")

    # 4. Plot the Curve (Save to a separate file)
    plt.figure(figsize=(10, 5))
    plt.plot(lags, correlations, color='purple', label='OFI Predictive Power')
    plt.axvline(x=peak_lag, color='green', linestyle='--', label=f'Best Lag: {peak_lag}s')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.title('Lead-Lag Analysis: How far ahead does OFI look?')
    plt.xlabel('Seconds into Future')
    plt.ylabel('Correlation (IC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lead_lag_analysis.png')
    print("📊 Chart saved as 'lead_lag_analysis.png'")

# ==========================================
# 2. VISUALIZATION LOGIC
# ==========================================
def plot_results():
    if not data_buffer:
        print("\n❌ No data collected. Check connection.")
        return

    print("\nGenerating Chart...", end="")
    df = pd.DataFrame(data_buffer)
    
    # Normalize time
    start_time = df['timestamp'].iloc[0]
    df['seconds'] = df['timestamp'] - start_time
    
    # CALCULATE CUMULATIVE OFI
    # Raw OFI is noisy. We sum it up to see the "Inventory Trend".
    df['cum_ofi'] = df['ofi'].cumsum()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Price (Orange)
    color = 'tab:orange'
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel(f'{SYMBOL.upper()} Price ($)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df['seconds'], df['mid_price'], color=color, linewidth=2, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Cumulative OFI (Blue)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Cumulative OFI', color=color, fontsize=12, fontweight='bold')
    ax2.plot(df['seconds'], df['cum_ofi'], color=color, linewidth=1.5, label='Cumulative OFI')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Visual Proof: Cumulative OFI vs Price ({SYMBOL.upper()})', fontsize=14)
    fig.tight_layout()
    
    # Save the chart
    filename = "ofi_chart.png"
    plt.savefig(filename)
    print(f" Done! Saved to {filename}")

    analyze_lead_lag(df)

# ==========================================
# 3. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    # Use the /stream endpoint
    socket = "wss://stream.binance.com:9443/stream"
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message)
    
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    try:
        # Countdown loop
        for i in range(DURATION):
            sys.stdout.write(f"\r⏳ Collecting Data: {i+1}/{DURATION}s")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping early...")
    
    ws.close()
    plot_results()