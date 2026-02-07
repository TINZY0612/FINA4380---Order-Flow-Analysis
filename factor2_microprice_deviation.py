def calculate_microprice(best_bid_p, best_bid_q, best_ask_p, best_ask_q):
    """
    Calculates the Volume-Weighted Microprice.
    """
    total_qty = best_bid_q + best_ask_q
    
    if total_qty == 0:
        return (best_bid_p + best_ask_p) / 2 # Fallback to MidPrice
        
    # The Cross-Multiplication Formula
    microprice = ((best_bid_q * best_ask_p) + (best_ask_q * best_bid_p)) / total_qty
    
    return microprice

# Example Usage in your Strategy Logic:
mid_price = (best_bid_p + best_ask_p) / 2
deviation = microprice - mid_price

if deviation > 0.5: # Threshold (needs optimization)
    print("Bullish Microprice Divergence! Preparing to Buy.")
