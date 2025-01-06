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
        Add aggregations to a DataFrame column.
        Args:
            df (DataFrame): The DataFrame to add aggregations to.
            column_name (str): The name of the column to aggregate.
            fn (function): The aggregation function to apply.
            windows (Union[int, List[int]]): The window size(s) for aggregation.
            suffix (str, optional): The suffix to append to the feature names. Defaults to None.
            rel_column_name (str, optional): The name of the relative column for normalization. Defaults to None.
            rel_factor (float, optional): The factor to multiply the normalized feature by. Defaults to 1.0.
            last_rows (int, optional): The number of last rows to aggregate. Defaults to 0.
        Returns:
            list: A list of feature names created by the aggregation.
    """

    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]
 
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
        df[feature_name] = rel_factor * feature

    return features

def add_area_ratio(df, column_name: str, windows: Union[int, List[int]], suffix=None, last_rows: int = 0):
    """
    Parameters:
    - df: DataFrame
        The input DataFrame.
    - column_name: str
        The name of the column to calculate the area ratio for.
    - windows: int or List[int]
        The window size(s) to use for calculating the area ratio.
    - suffix: str, optional
        The suffix to append to the feature column names. Default is "_area_ratio".
    - last_rows: int, optional
        The number of last rows to aggregate. Default is 0.
    Returns:
    - features: List
        A list of calculated area ratio features.
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
    """
    Calculate the ratio of positive values to the total absolute difference between each element and the last element in the input array.

    Parameters:
    - x (array-like): Input array.

    Returns:
    - float: The calculated ratio.

    Example:
    >>> area_fn([1, 2, 3, 4, 5])
    0.5
    """
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
    Calculates the linear trends for a given column in a DataFrame.
    Parameters:
        df (DataFrame): The input DataFrame.
        column_name (str): The name of the column to calculate linear trends for.
        windows (Union[int, List[int]]): The window size(s) to use for calculating linear trends. 
            If an integer is provided, the same window size will be used for all calculations. 
            If a list of integers is provided, multiple window sizes will be used.
        suffix (str, optional): The suffix to append to the column name in the output feature names. 
            Defaults to None.
        last_rows (int, optional): The number of last rows to aggregate when calculating linear trends. 
            If set to 0, all rows will be used. Defaults to 0.
    Returns:
        List[str]: A list of feature names representing the calculated linear trends.
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
    Calculate the slope of a linear regression line for the given data.
    Parameters:
    x (array-like): The input data.
    Returns:
    float: The slope of the linear regression line.
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
    """
    Aggregate the last rows of a column using a given function.

    Parameters:
        column (pd.Series): The column to aggregate.
        window (int): The size of the window to consider for aggregation.
        last_rows (int): The number of last rows to aggregate.
        fn (function): The function to use for aggregation.
        *args: Additional arguments to pass to the aggregation function.

    Returns:
        pd.Series: A series containing the aggregated values.

    """
    length = len(column)
    values = [fn(column.iloc[-window - r: length - r].to_numpy(), *args) for r in range(last_rows)]
    feature = pd.Series(data=np.nan, index=column.index, dtype=float)
    feature.iloc[-last_rows:] = list(reversed(values))
    return feature