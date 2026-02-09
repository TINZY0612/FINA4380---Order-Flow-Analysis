import websocket
import json
import pandas as pd
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
import sys

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = 'ethusdt'      # Focused on ETH (High L2 Performance)
DURATION = 60           # Collect data for 1 minute
DECAY_FACTOR = 50       # The "k" in your formula

data_buffer = []

# ==========================================
# 1. WEBSOCKET LOGIC
# ==========================================
def on_message(ws, message):
    try:
        json_msg = json.loads(message)
        
        # Unwrapping Logic (Same as before)
        if 'data' in json_msg:
            payload = json_msg['data']
        else:
            payload = json_msg

        if 'bids' in payload and 'asks' in payload:
            bids = np.array(payload['bids'], dtype=float)
            asks = np.array(payload['asks'], dtype=float)
            
            # Calculate Mid Price
            mid_price = (bids[0][0] + asks[0][0]) / 2
            
            # Calculate Weighted OBI
            w_obi = calculate_weighted_obi(bids, asks, mid_price)
            
            record = {
                'timestamp': time.time(),
                'mid_price': mid_price,
                'l2_obi': w_obi
            }
            data_buffer.append(record)
            
    except Exception:
        pass

#Most Important Part: Calculating OBI
def calculate_weighted_obi(bids, asks, mid_price):
    # Distance Calculation
    bid_dist = (mid_price - bids[:, 0]) / mid_price
    ask_dist = (asks[:, 0] - mid_price) / mid_price
    
    # Weighting (Exponential Decay)
    bid_weights = np.exp(-DECAY_FACTOR * bid_dist)
    ask_weights = np.exp(-DECAY_FACTOR * ask_dist)
    
    # Weighted Sums
    total_bid_pressure = np.sum(bids[:, 1] * bid_weights)
    total_ask_pressure = np.sum(asks[:, 1] * ask_weights)
    
    if (total_bid_pressure + total_ask_pressure) == 0: return 0
    return (total_bid_pressure - total_ask_pressure) / (total_bid_pressure + total_ask_pressure)

def on_open(ws):
    print(f"✅ Connected! Collecting L2 Data for {SYMBOL.upper()}...")
    params = [f"{SYMBOL}@depth20@100ms"]
    subscribe_msg = { "method": "SUBSCRIBE", "params": params, "id": 1 }
    ws.send(json.dumps(subscribe_msg))

def analyze_lead_lag(df):
    print("\nCalculating Optimal Lead-Lag...")
    
    # We will test lags from 0 to 10 seconds (in 100ms steps)
    # Since we collect at roughly 100ms intervals, 1 sec = 10 steps
    lags = range(0, 100) 
    correlations = []
    
    # Calculate % Return for Price (Stationary data correlates better than raw price)
    df['price_return'] = df['mid_price'].pct_change()
    
    for lag in lags:
        # Shift price BACKWARDS by 'lag' steps to see if past OBI predicts future Price
        shifted_return = df['price_return'].shift(-lag)
        
        # Calculate correlation
        corr = df['l2_obi'].corr(shifted_return)
        correlations.append(corr)
        
    # Find the lag with the highest correlation
    max_corr = max(correlations)
    optimal_lag_index = correlations.index(max_corr)
    
    # Estimate time in seconds (assuming ~0.1s per step)
    # Note: For accuracy, we should use actual timestamps, but this is a good estimate.
    estimated_time = optimal_lag_index * 0.1 
    
    print(f"✅ Peak Correlation: {max_corr:.4f}")
    print(f"✅ Optimal Lead Time: ~{estimated_time:.1f} seconds")
    
    # Plot the "Lead Curve"
    plt.figure(figsize=(10, 4))
    plt.plot([x * 0.1 for x in lags], correlations)
    plt.axvline(estimated_time, color='r', linestyle='--', label=f'Peak: {estimated_time}s')
    plt.title("Lead-Lag Analysis: How far ahead does OBI look?")
    plt.xlabel("Lag (Seconds)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.savefig("lead_lag_analysis.png")
    print("Graph saved to lead_lag_analysis.png")


# ==========================================
# 2. VISUALIZATION LOGIC
# ==========================================
def plot_results():
    if not data_buffer:
        print("❌ No data to plot.")
        return

    print("\nGeneratng Chart...", end="")
    df = pd.DataFrame(data_buffer)
    
    # Normalize time to start at 0 seconds
    start_time = df['timestamp'].iloc[0]
    df['seconds'] = df['timestamp'] - start_time
    
    # Create the Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Price (Orange)
    color = 'tab:orange'
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('ETH Price ($)', color=color)
    ax1.plot(df['seconds'], df['mid_price'], color=color, linewidth=2, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for OBI (Blue)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Weighted L2 OBI', color=color)
    ax2.plot(df['seconds'], df['l2_obi'], color=color, linewidth=1.5, alpha=0.7, label='L2 Imbalance')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add a horizontal line at 0 (Neutral)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)

    plt.title(f'Visual Proof: Order Book Imbalance vs Price ({SYMBOL.upper()})')
    fig.tight_layout()

    analyze_lead_lag(df)
    
    # Save to a file instead of showing
    filename = "obi_chart.png"
    plt.savefig(filename)
    print(f" Done! Chart saved to {filename}")

# ==========================================
# 3. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    socket = "wss://stream.binance.com:9443/stream"
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message)
    
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    try:
        for i in range(DURATION):
            sys.stdout.write(f"\r⏳ Collecting: {i+1}/{DURATION}s")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    ws.close()
    plot_results()