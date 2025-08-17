from datetime import datetime, timedelta
import ccxt
import pandas as pd
import json
import os
import time

def get_high_volume_pairs():
    exchange = ccxt.binance()
    
    try:
        # Get all USDT spot pairs
        markets = exchange.load_markets()
        usdt_pairs = [symbol for symbol in markets.keys() 
                     if symbol.endswith('/USDT') and markets[symbol]['spot']]
        
        print(f"Found {len(usdt_pairs)} USDT spot pairs")
        
        # Get tickers in batches of 100
        all_tickers = {}
        batch_size = 100
        
        for i in range(0, len(usdt_pairs), batch_size):
            batch = usdt_pairs[i:i + batch_size]
            print(f"Fetching batch {i//batch_size + 1}/{len(usdt_pairs)//batch_size + 1}")
            tickers = exchange.fetch_tickers(batch)
            all_tickers.update(tickers)
            time.sleep(1)  # Rate limit compliance
        
        # Convert to DataFrame and sort by volume
        df = pd.DataFrame.from_dict(all_tickers, orient='index')
        pairs = df.sort_values('quoteVolume', ascending=False)
        
        # Format pairs for whitelist (top 200)
        whitelist = [pair for pair in pairs.index[:200]]
        
        # Save just the whitelist array to a file
        save_path = os.path.join(os.getcwd(), 'user_data/pairlist-volume-top-200.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(whitelist, f, indent=4)
        
        print(f"Saved {len(whitelist)} pairs to: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_high_volume_pairs() 