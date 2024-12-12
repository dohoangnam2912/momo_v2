from typing import Union, List
from decimal import *
import numpy as np
import pandas as pd
from scipy import stats

def _add_aggregations(df, column_name:str, fn, 
                      windows:Union[int, List[int]], 
                      suffix=None, rel_column_name:str=None, 
                      rel_factor: float=1.0,
                      last_rows: int = 0
                      ):
    """
    Add moving aggregations over past values.
    """

    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]

    if rel_column_name:
        rel_column = df[rel_column_name]
    
    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:
        # Aggregate 
        if not last_rows:
            feature = column.rolling(windows=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
        else:
            feature = _aggregate_last_rows(column, w, last_rows, fn)

        # Normalize
        feature_name = column_name + suffix + '_' + str(w)
        features.append(feature_name)
        if rel_column_name:
            df[feature_name] = rel_factor * (feature - rel_column) / rel_column
        else:
            df[feature_name] = rel_factor * feature
    
    return features

def _add_weighted_aggregations(df, 
                               column_name:str,
                               weight_column_name: str,
                               fn, 
                               windows: Union[int, List[int]],
                               suffix=None,
                               rel_column_name:str=None,
                               rel_factor: float=1.0,
                               last_rows: int = 0):
    """
    Weighted rolling aggregation
    """

    column = df[column_name]

    if weight_column_name:
        weight_column = df[weight_column_name]
    else:
        # If no weight_column is specified, it will be equal to 1
        weight_column = pd.Series(data=1.0, index=column.index)
    
    products_column = column * weight_column

    if isinstance(windows, int):
        windows = [windows]
    
    if rel_column_name:
        rel_column = df[rel_column_name]
    
    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:
        if not last_rows:
            # Sum of products
            feature = products_column.rolling(window=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
            # Sum of weights
            weights = weight_column.rolling(window=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
        else:
            feature = _aggregate_last_rows(products_column, w, last_rows, fn)
            weights = _aggregate_last_rows(weight_column, w, last_rows, fn)

        feature = feature / weights

        feature_name = column_name + suffix + '_' + str(w)
        features.append(feature_name)
        if rel_column_name:
            df[feature_name] = rel_factor * (feature - rel_column) / rel_column
        else:
            df[feature_name] = rel_factor * feature

    return features

def add_area_ratio(df, column_name: str, windows: Union[int, List[int]], suffix=None, last_rows: int = 0):
    """
    We take past element and compare the previous sub-series: The are under and over this element
    """
    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]
    
    if suffix is None:
        suffix = "_" + "area_ratio"

    features = []
    for w in windows:
        if not last_rows:
            ro = column.rolling(window=w, min_periods=max(1, w // 2))
            feature = ro.apply(area_fn, raw=True)
        else:
            feature = _aggregate_last_rows(column, w, last_rows, area_fn)

def area_fn(x):
    level = x[-1]
    x_diff = x - level
    a = np.nansum(x_diff)
    b = np.nansum(np.absolute(x_diff))
    pos = (b + a) / 2
    ratio = pos / b
    ratio = (ratio * 2) - 1
    return ratio

def add_linear_trends(df, column_name:str, windows: Union[int, List[int]], suffix=None, last_rows:int=0):
    """
    Computing the slope of fitted line.
    """
    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]

    if suffix is None:
        suffix = "_" + "trend"

    features = []
    for w in windows:
        if not last_rows:
            ro = column.rolling(window=w, min_period=max(1, w // 2))
            feature = ro.apply(slope_fn, raw=True)
        else:
            feature = _aggregate_last_rows(column, w, last_rows, slope_fn)
        
        feature_name = column_name + suffix + '_' + str(w)
        features.append(feature_name)
    
    return features

def slope_fn(x):
    """
    Fit a linear regression model and return its slope
    """
    X_array = np.asarray(range(len(x)))
    y_array = x
    if np.isnan(y_array).any():
        nans = ~np.isnan(y_array)
        X_array = X_array[nans]
        y_array = y_array[nans]
        
    slope, _, _, _, _ = stats.linregress(X_array, y_array)

    return slope

def to_log_diff(sr):
    return np.log(sr).diff

def to_diff(sr):
    return 100 * sr.diff() / sr

def _aggregate_last_rows(column, window, last_rows, fn, *args):
    length = len(column)
    values = [fn(column.iloc[-window - r: length - r].to_numpy(), *args) for r in range(last_rows)]
    feature = pd.Series(data=np.nan, index=column.index, dtype=float)
    feature.iloc[-last_rows:] = list(reversed(values))
    return feature