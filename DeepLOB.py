import torch
import torch.nn as nn
import numpy as np
from collections import deque

# ==========================================
# 1. THE BRAIN (Model Architecture)
# ==========================================
class DeepLOB(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len
        
        # Convolutional Block (Extracts Spatial Features)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        
        # Inception Module (Extracts Multi-Scale Features)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        
        # LSTM (Extracts Temporal Features)
        # input_size=320 because the Conv output is 32 channels * 10 features
        self.lstm = nn.LSTM(input_size=320, hidden_size=64, num_layers=1, batch_first=True)
        
        # Output Layer (3 Classes: Down, Stationary, Up)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        # x shape: (Batch, 1, 100, 40)
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Reshape for LSTM: (Batch, Channels, Time, Features) -> (Batch, Time, Flat_Features)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        
        x, _ = self.lstm(x)
        x = x[:, -1, :] # Take the last time step
        forecast = self.fc1(x)
        return forecast

# ==========================================
# 2. THE TRADER (Live Interface)
# ==========================================
class LiveAITrader:
    def __init__(self, model_path='deeplob_weights.pth', device='cpu'):
        self.device = device
        print(f"🧠 Initializing AI Trader on {device}...")
        
        # Initialize Model
        self.model = DeepLOB(y_len=3).to(device)
        
        # Load Weights (Safely)
        try:
            # map_location ensures it loads on CPU even if trained on GPU
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"✅ AI Weights Loaded from {model_path}")
        except FileNotFoundError:
            print(f"❌ ERROR: '{model_path}' not found. Did you download it?")
            print("⚠️ Running with RANDOM WEIGHTS (Useless predictions)")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")

        self.model.eval() # Set to evaluation mode (Important!)
        self.buffer = deque(maxlen=100) # Rolling memory of last 100 ticks

    def process_tick(self, bids, asks):
        """
        Ingests a live Order Book snapshot from Binance.
        bids: List of [price, size]
        asks: List of [price, size]
        Returns: Float Signal (-1.0 to 1.0) or None
        """
        # 1. Feature Engineering (Construct the 40-feature vector)
        # We need top 10 levels: [AskP, AskV, BidP, BidV]...
        features = []
        for i in range(10):
            # Safety check: pad with 0 if book is thin
            ap = float(asks[i][0]) if i < len(asks) else 0.0
            av = float(asks[i][1]) if i < len(asks) else 0.0
            bp = float(bids[i][0]) if i < len(bids) else 0.0
            bv = float(bids[i][1]) if i < len(bids) else 0.0
            features.extend([ap, av, bp, bv])
            
        # Add to memory buffer
        self.buffer.append(features)

        # 2. Predict (Only if we have a full window of history)
        if len(self.buffer) == 100:
            return self._predict()
        else:
            # Warming up... (Need 100 ticks)
            return None 

    def _predict(self):
        # Convert buffer to Numpy Array
        data = np.array(self.buffer) # Shape (100, 40)
        
        # --- CRITICAL: REAL-TIME NORMALIZATION ---
        # The model was trained on stock prices ~10.00
        # Crypto prices are ~90,000.00. We MUST Z-Score normalize.
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8 # Avoid divide by zero
        data_norm = (data - mean) / std
        
        # Prepare Tensor
        tensor = torch.FloatTensor(data_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        # Shape: (1, 1, 100, 40)
        
        # Run Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
        # Extract Signal
        # 0: Down, 1: Stationary, 2: Up
        p_down = probs[0][0].item()
        p_up = probs[0][2].item()
        
        # Alpha Signal: Confidence of Up - Confidence of Down
        signal = p_up - p_down 
        return signal

# ==========================================
# 3. TEST BLOCK (Run this file to verify)
# ==========================================
if __name__ == "__main__":
    # Create dummy file to test loading
    dummy_model = DeepLOB(y_len=3)
    torch.save(dummy_model.state_dict(), 'deeplob_weights.pth')
    
    # Test the Trader
    trader = LiveAITrader('deeplob_weights.pth')
    
    print("⏳ Warming up with dummy data...")
    # Feed 105 dummy ticks
    for _ in range(105):
        # Fake Bids/Asks
        dummy_bids = [[100, 1]] * 10
        dummy_asks = [[101, 1]] * 10
        
        sig = trader.process_tick(dummy_bids, dummy_asks)
        if sig is not None:
            print(f"🤖 Output Signal: {sig:.4f}")
            break