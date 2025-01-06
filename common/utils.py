import dateparser
import pytz
from datetime import datetime, timezone, timedelta
from decimal import *
import numpy as np
import pandas as pd
from apscheduler.triggers.cron import CronTrigger
from common.gen_features import *

def klines_to_df(klines, df):
    """
    Preprocessing klines data and turn into dataframe
    """
    data = pd.DataFrame(klines, 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
                        )
    
    dtypes = {
        'open': "float64",
        'high': "float64",
        'low': "float64",
        'close': "float64",
        'volume': "float64",
        'close_time': "int64" ,
        'quote_av': "float64" ,
        'trades': "int64" ,
        'tb_base_av': "float64" ,
        'tb_quote_av': "float64" ,
        'ignore': 'float64',
    }
    
    data = data.astype(dtypes)

    if df is None or len(df) == 0:
        df = data
    else:
        df = pd.concat([df, data])
    
    df = df.drop_duplicates(subset=["timestamp"], keep='last')
    df.set_index('timestamp', inplace=True)

    return df

def binance_freq_from_pandas(freq: str) -> str:
    """
    Pandas frequency (for example: min), is not similar to Binance frequency (for example: m)
    """
    if freq.endswith("min"):
        freq = freq.replace("min", "m")
    elif freq.endswith("D"):
        freq = freq.replace("D", "d")
    elif freq.endswith("W"):
        freq = freq.replace("W", "w")
    elif freq == "BMS":
        freq = freq.replace("BMS", "m")

    if len(freq) == 1:
        freq = "1" + freq
    
    return freq

def to_decimal(value):
    """
    Converting to decimal value for more precision.
    The value can be string, float or decimal.
    """
    precision = 8  # Number of decimal places to round to
    precision_factor = Decimal(1) / (Decimal(10) ** precision)  # The smallest step for precision
    rounded_value = Decimal(str(value)).quantize(precision_factor, rounding=ROUND_DOWN)  # Round to precision
    return rounded_value

def round_str(value, precision):
    precision_factor = Decimal(1) / (Decimal(10) ** precision)  # The smallest step for precision
    rounded_value = Decimal(str(value)).quantize(precision_factor, rounding=ROUND_HALF_UP)  # Round half up to precision
    return rounded_value
    
def round_down_str(value, precision):
    precision_factor = Decimal(1) / (Decimal(10) ** precision)  # The smallest step for precision
    rounded_value = Decimal(str(value)).quantize(precision_factor, rounding=ROUND_DOWN)  # Round to precision
    return rounded_value

def find_index(df: pd.DataFrame, date_str: str, column_name: str = "timestamp"):
    d = dateparser.parse(date_str)
    try:
        res = df[df[column_name] == d]
    except TypeError:
        if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
            d = d.replace(tzinfo=pytz.utc)
        else:
            d = d.replace(tzinfo=None)
        
        res = df[df[column_name] == d]

    if res is None or len(res) == 0:
        raise ValueError(f"Cannot find date '{date_str}' i the column {column_name}")
    
    id = res.index[0]

    return id
