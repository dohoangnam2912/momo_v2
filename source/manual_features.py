import pandas as pd
import talib

# Load your klines data
df = pd.read_csv('/home/yosakoi/Work/momo_v2/data/BTCUSDT/features.csv')

# Ensure numeric columns are correctly formatted
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# Candlestick Patterns
df['CDL_DOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
df['CDL_HAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
df['CDL_ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])

# Support/Resistance Breakout Detection
df['Resistance_Breakout'] = (df['close'] > df['high'].rolling(20).max())
df['Support_Breakout'] = (df['close'] < df['low'].rolling(20).min())

# Fill NaN values generated by indicators
df.fillna(0, inplace=True)

# Save the processed dataset
df.to_csv('/home/yosakoi/Work/momo_v2/data/BTCUSDT/features.csv', index=False)
print("Feature engineering complete. Data saved to /home/yosakoi/Work/momo_v2/data/BTCUSDT/features.csv")