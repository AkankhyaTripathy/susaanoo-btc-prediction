import pandas as pd
import numpy as np
class FeatureEngineer:
    def compute(self, df):
        df=df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility']=df['log_return'].rolling(window=24).std()
        df['rsi']=self.compute_rsi(df['close'])
        df['bb_deviation'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(24).mean()) / df['volume'].rolling(24).std()
        df['taker_buy_ratio'] = df['taker_buy_volume'] / df['volume']
        return df
    
    def compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
if __name__ == "__main__": # type: ignore
    df = pd.read_csv("btc_data.csv", index_col="timestamp")
    fe = FeatureEngineer()
    df = fe.compute(df)
    print(df.tail(10))
    print(df.shape)