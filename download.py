import requests
import pandas as pd
import time

def download_btc_data():
    print("Starting download...")
    
    all_candles = []
    start_time = 1640995200000
    end_time = 1704067200000
    
    while start_time < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": start_time,
            "limit": 1000
        }
        
        response = requests.get(url, params=params)
        candles = response.json()
        
        if len(candles) == 0:
            break
            
        all_candles.extend(candles)
        start_time = candles[-1][0] + 3600000
        print(f"Downloaded {len(all_candles)} candles so far...")
        time.sleep(0.5)

    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    
    numeric_cols = ["open", "high", "low", "close", "volume", 
                    "taker_buy_volume", "taker_buy_quote_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    df.to_csv("btc_data.csv")
    print(f"Done! Total candles: {len(df)}")
    return df

if __name__ == "__main__":
    download_btc_data()