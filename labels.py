import pandas as pd
import numpy as np

from features import FeatureEngineer 
def triple_barrier(df, tp_mult=2, sl_mult=1, max_bars=24):
    labels =[]
    close = df['close']. values
    volatility = df['volatility'].values

    for i in range(len(df)):
        if np. isnan(volatility[i]):
            labels.append(0)
            continue

        price = close[i]
        vol = volatility[i]
        tp = price + tp_mult * vol
        sl = price - sl_mult * vol
        label = 0
    
        for j in range(i + 1, min(i + max_bars, len(df))):
            if close[j] >= tp:
                label = 1
                break
            elif close[j] <= sl:
                label = -1
                break

        labels.append(label)

    df = df.copy()
    df['label'] = labels
    return df

if __name__ == "__main__":
    df = pd.read_csv("btc_data.csv", index_col="timestamp")
    from features import FeatureEngineer
    fe = FeatureEngineer()
    df = fe.compute(df)
    df = triple_barrier(df)
    print(df['label'].value_counts())
    print(df.shape)

