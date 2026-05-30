import pandas as pd

df = pd.read_csv("btc_data.csv", index_col="timestamp")
print(df.shape)
print(df.head())
print(df.columns.tolist())