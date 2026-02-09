# FINA4380---Order-Flow-Analysis
CUHK 2025/26 FINA4380 Group Project 3 - Order Flow Analysis

Order Flow Analysis & Deep Learning for HFT
This repository contains a comprehensive exploration of Market Microstructure and High-Frequency Trading (HFT) strategies. The project transitions from traditional statistical factors to state-of-the-art Deep Learning models for price trend prediction.

📌 Project Overview
The core objective of this project is to analyze Limit Order Book (LOB) dynamics to predict short-term price movements. It is divided into two primary phases:

Factor Research (Phase 1): Manual engineering and testing of order flow factors (OFI, OBI, TFI, and Microprice Deviation).

Deep Learning (Phase 2): Implementation of the DeepLOB architecture (Zhang et al., 2018) to automate feature extraction and capture non-linear temporal dependencies.

📂 Repository Structure
🔬 Phase 1: Statistical Factor Analysis
These files contain the manual research and testing of traditional alpha factors:

l1_ofi.py / l2_ofi.py: Implementation of Order Flow Imbalance across different book depths.

l2_obi.py: Analysis of Order Book Imbalance (static liquidity snapshots).

tfi.py: Research into Trade Flow Imbalance and its impact on price impact.

FINA4380_Supplementary.pptx: Core Project Documentation. Contains the thought process, experimental results, and a deep dive into the findings of each factor.

🤖 Phase 2: Deep Learning (DeepLOB)
Implementation of a spatial-temporal neural network for HFT:

study_deeplob.ipynb: The research notebook used for model training, data preprocessing (transposing and normalization), and hyperparameter tuning.

DeepLOB.py: The blueprint file defining the CNN-LSTM hybrid architecture.

deeplob_weights.pth: The trained "Brain" of the model, containing weights optimized on LOB data.

ofi_ai.py: A deployment-ready script that connects to the Binance WebSocket to generate live AI-driven trade signals.

🛠 Methodology: The DeepLOB Architecture
The model utilizes a hierarchical structure to "see" and "remember" market states:

CNN Blocks: Automatically integrate Price and Volume data across multiple book levels to extract spatial features (e.g., bid-ask spreads and liquidity clusters).

Inception Modules: Capture market trends across multiple time scales simultaneously.

LSTM Layer: Processes the sequence of extracted features to understand the "story" behind the order flow (momentum vs. absorption).

🚀 How to Run the Live AI Signal
Ensure you have the requirements installed: pip install torch numpy websocket-client.

Place deeplob_weights.pth and DeepLOB.py in the root directory.

Execute the live tracker:

Bash
python ofi_ai.py
The terminal will display a live confidence bar indicating the model's predicted direction for BTC/USDT.

📝 Observations & Findings
Factor Decay: Traditional factors like OFI show strong short-term predictive power but decay rapidly as the prediction horizon increases.

Deep Learning Advantage: DeepLOB achieved an accuracy of 43.27% on 3-class classification (Up/Down/Stationary), significantly outperforming random chance (33%) by capturing complex non-linear interactions between levels.
