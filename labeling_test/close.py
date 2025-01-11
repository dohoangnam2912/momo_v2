import pandas as pd

df = pd.read_csv('/home/yosakoi/Work/momo_v2/labeling_test/klines.csv')
df = df["close"]

df.to_csv('/home/yosakoi/Work/momo_v2/labeling_test/close.csv', index=False)