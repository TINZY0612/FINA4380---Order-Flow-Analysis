def depletion_strategy(bids, asks, historical_data):
    # 1. Measure current liquidity
    current_bid_liquidity = sum([b[1] for b in bids[:5]])
    current_ask_liquidity = sum([a[1] for a in asks[:5]])
    
    # 2. Compare to historical average
    avg_bid_liquidity = historical_data['avg_bid_liquidity_5min']
    avg_ask_liquidity = historical_data['avg_ask_liquidity_5min']
    
    bid_ratio = current_bid_liquidity / avg_bid_liquidity if avg_bid_liquidity > 0 else 0
    ask_ratio = current_ask_liquidity / avg_ask_liquidity if avg_ask_liquidity > 0 else 0
    
    # 3. Check for depletion
    if bid_ratio < 0.3 and ask_ratio < 0.3:
        # Both sides depleted - mean reversion opportunity
        mid = (bids[0][0] + asks[0][0]) / 2
        last_price = get_last_trade_price()
        
        if last_price < mid:
            return "BUY - price below mid in depleted market"
        else:
            return "SELL - price above mid in depleted market"
    
    elif bid_ratio < 0.2 and ask_ratio > 0.7:
        # Bid side depleted - short opportunity
        return "SHORT - vacuum below"
    
    elif ask_ratio < 0.2 and bid_ratio > 0.7:
        # Ask side depleted - long opportunity
        return "LONG - vacuum above"
    
    else:
        return "NO TRADE"