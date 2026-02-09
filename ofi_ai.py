import json
import websocket
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import threading
import time

# ==========================================
# 1. THE BRAIN: DeepLOB Architecture
# (Must match your training code exactly)
# ==========================================
class DeepLOB(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len
        
        # Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.lstm = nn.LSTM(input_size=320, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc1(x)

# ==========================================
# 2. THE TRADER: Live AI Logic
# ==========================================
class LiveAITrader:
    def __init__(self, model_path='deeplob_weights.pth'):
        self.device = torch.device("cpu") # CPU is fast enough for inference
        print(f"🧠 Initializing AI Model...")
        
        self.model = DeepLOB(y_len=3).to(self.device)
        self.buffer = deque(maxlen=100) # Memory of last 100 ticks
        
        try:
            # Load Weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval() # Vital: Set to evaluation mode
            print("✅ Brain Loaded! AI is ready.")
        except FileNotFoundError:
            print("❌ ERROR: 'deeplob_weights.pth' not found in this folder.")
            print("   Please download it from your training notebook.")
            exit()

    def process_book(self, bids, asks):
        """
        Input: bids, asks (Lists of [price, size] strings from Binance)
        """
        # 1. Feature Engineering (Top 10 Levels)
        features = []
        for i in range(10):
            # Safe parsing (Binance sends strings)
            ap = float(asks[i][0]) if i < len(asks) else 0.0
            av = float(asks[i][1]) if i < len(asks) else 0.0
            bp = float(bids[i][0]) if i < len(bids) else 0.0
            bv = float(bids[i][1]) if i < len(bids) else 0.0
            features.extend([ap, av, bp, bv])
            
        # 2. Add to Rolling Memory
        self.buffer.append(features)

        # 3. Predict (only if memory is full)
        if len(self.buffer) == 100:
            return self._predict()
        else:
            return None # Warming up

    def _predict(self):
        # Convert to Numpy
        data = np.array(self.buffer)
        
        # --- Z-SCORE NORMALIZATION (Crucial) ---
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data_norm = (data - mean) / std
        
        # To Tensor
        tensor = torch.FloatTensor(data_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
        # Extract Signal
        # Col 0: Down, Col 1: Stationary, Col 2: Up
        p_down = probs[0][0].item()
        p_up = probs[0][2].item()
        
        return p_up - p_down # Alpha Signal

# ==========================================
# 3. WEBSOCKET CLIENT
# ==========================================
SYMBOL = "btcusdt"
SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth20@100ms"

# Initialize Global AI
trader = LiveAITrader()

def on_message(ws, message):
    json_msg = json.loads(message)
    
    # Binance Partial Depth Payload
    bids = json_msg['bids']
    asks = json_msg['asks']
    
    # Run AI
    signal = trader.process_book(bids, asks)
    
    # Print Result
    if signal is None:
        print(f"⏳ Filling Memory... ({len(trader.buffer)}/100)", end='\r')
    else:
        # Visual Bar for Terminal
        bar_len = 20
        # Map signal (-1 to 1) to bar index (0 to 20)
        pos = int((signal + 1) / 2 * bar_len)
        pos = max(0, min(bar_len, pos))
        
        bar = ['-'] * (bar_len + 1)
        bar[bar_len // 2] = '|' # Center line
        if signal > 0:
            for i in range(bar_len // 2, pos): bar[i] = '🟩'
            bar[pos] = '🚀'
        else:
            for i in range(pos, bar_len // 2): bar[i] = '🟥'
            bar[pos] = '🔻'
            
        bar_str = "".join(bar).replace('-', ' ')
        
        print(f"AI Signal: {signal:+.4f} [{bar_str}]")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### Connection Closed ###")

def on_open(ws):
    print("### Connected to Binance WebSocket ###")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    print(f"Listening to {SYMBOL}...")
    ws.run_forever()