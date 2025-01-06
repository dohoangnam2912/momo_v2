import sys
import importlib
import itertools
import numpy as np
import pandas as pd

import scipy.stats as stats

from common.utils import *
from common.gen_features_agg import _aggregate_last_rows

def generate_features_tsfresh(df, config:dict, last_rows: int = 0):
    """
    Generate time-series features using the tsfresh library.
    Args:
        df (pandas.DataFrame): The input dataframe.
        config (dict): Configuration dictionary containing the columns and windows for feature generation.
        last_rows (int, optional): Number of last rows to generate features for. Defaults to 0.
    Returns:
        list: List of generated feature names.
    """
    import tsfresh.feature_extraction.feature_calculators as tsf

    column_names = config.get('columns')
    if not column_names:
        raise ValueError(f"No input column for feature generator 'tsfresh': {column_names}")
    
    if isinstance(column_names, str):
        column_name = column_names
    elif isinstance(column_names, list):
        column_name = column_names[0]
    elif isinstance(column_names, dict):
        column_name = next(iter(column_names.values())) #TODO: What is this
    else:
        raise ValueError(f"Columns are provided as a string, list or dict. Wrong type: {type(column_names)}")
    
    column = df[column_name].interpolate() # Fill the NaN

    windows = config.get('windows')
    if not isinstance(windows, list):
        windows = [windows]
    
    features = []
    for w in windows:
        rolling = column.rolling(window=w, min_periods=max(1, w // 2))

        feature_name = column_name + "_skerness_" + str(w)

        if not last_rows: 
            df[feature_name] = rolling.apply(tsf.skewness, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.skewness) # Final row has to be generated differently

        features.append(feature_name)

        feature_name = column_name + "_kurtosis_" + str(w)

        if not last_rows: 
            df[feature_name] = rolling.apply(tsf.kurtosis, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.kurtosis) # Final row has to be generated differently

        features.append(feature_name)

        feature_name = column_name + "_msdc_" + str(w)

        if not last_rows: 
            df[feature_name] = rolling.apply(tsf.mean_second_derivative_central, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.mean_second_derivative_central) # Final row has to be generated differently

        features.append(feature_name)

        feature_name = column_name + "_lsbm_" + str(w)

        if not last_rows: 
            df[feature_name] = rolling.apply(tsf.longest_strike_below_mean, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.longest_strike_below_mean) # Final row has to be generated differently

        features.append(feature_name)

        feature_name = column_name + "_fmax_" + str(w)

        if not last_rows: 
            df[feature_name] = rolling.apply(tsf.first_location_of_maximum, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.first_location_of_maximum) # Final row has to be generated differently

        features.append(feature_name)

        # TODO: Add more features

    return features

def generate_features_talib(df, config:dict, last_rows: int = 0):
    """
    Generate features using talib library.
    Args:
        df (pandas.DataFrame): The input DataFrame.
        config (dict): Configuration dictionary containing parameters for feature generation.
        last_rows (int, optional): Number of last rows to consider. Defaults to 0.
    Returns:
        list: List of feature names generated.
    Raises:
        ValueError: If the columns parameter is not of type string, list, or dict.
        ValueError: If the talib function name cannot be resolved.
        ValueError: If the names parameter is not of type string or list.
    """

    relative_base = config.get('parameters', {}).get('relative_base', False)
    relative_function = config.get('parameters', {}).get('relative_function', False)
    percentage = config.get('parameters', {}).get('percentage', False)
    log = config.get('parameters', {}).get('log', False)

    # talib
    module_name = "talib" # Functions applying to a rolling series of windows
    talib_mod = sys.modules.get(module_name)
    if talib_mod is None:
        try:
            talib_mod = importlib.import_module(module_name)
        except Exception as e:
            raise ValueError(f"Can't import module {module_name}. Check if it's installed correctly")
        
    module_name = "talib.stream" # Functions applying to a rolling series of windows
    talib_mod_stream = sys.modules.get(module_name)
    if talib_mod_stream is None:
        try:
            talib_mod_stream = importlib.import_module(module_name)
        except Exception as e:
            raise ValueError(f"Can't import module {module_name}. Check if it's installed correctly")
        
    module_name = "talib.abstract" # Functions applying to a rolling series of windows
    talib_mod_abstract = sys.modules.get(module_name)
    if talib_mod_abstract is None:
        try:
            talib_mod_abstract = importlib.import_module(module_name)
        except Exception as e:
            raise ValueError(f"Can't import module {module_name}. Check if it's installed correctly")

    # Data preprocessing
    column_names = config.get('columns')
    if isinstance(column_names, str):
        column_names = {'real': column_names}
    elif isinstance(column_names, list) and len(column_names) == 1:
        column_names = {'real': column_names[0]}
    elif isinstance(column_names, list):
        column_names = {f'real{i}': col for i,col in enumerate(column_names)}
    elif isinstance(column_names, dict):
        pass
    else:
        raise ValueError(f"Columns must be string, list or dict!. Current type: {type(column_names)}")
    
    # One NaN and the library talib won't work, so we need to interpolate
    columns = {param: df[col_name].interpolate() for param, col_name in column_names.items()}

    column_all_names = "_".join(column_names.values())

    func_names = config.get('functions')
    if not isinstance(func_names, list):
        func_names = [func_names]
    
    windows = config.get('windows')
    if not isinstance(windows, list):
        windows = [windows]

    names = config.get('names')
    outputs = []
    features = []
    for func_name in func_names:
        fn_outs = []
        fn_out_names = []

        try:
            fn = getattr(talib_mod_abstract, func_name)
        except Exception as e:
            raise ValueError(f"Can't resolve talib function name '{func_name}'. Check if the function is existed.")

        for i, w in enumerate(windows):
            # Working on offline mode
            if not last_rows or not w:
                try:
                    fn = getattr(talib_mod, func_name)
                except Exception as e:
                    raise ValueError(f"Can't resolve talib function name '{func_name}'. Check if the function is existed.")
                
                args = columns.copy()
                if func_name in ['ATR']:
                    args = {
                        'high': df['high'].interpolate(),
                        'low': df['low'].interpolate(),
                        'close': df['close'].interpolate(),
                        'timeperiod': w
                    }
                elif func_name in ['STOCH']:
                    args = {
                        'high': df['high'].interpolate(),
                        'low': df['low'].interpolate(),
                        'close': df['close'].interpolate(),
                        'fastk_period': config.get("fask_period", 5),
                        'slowk_period': config.get("slowk_period", 3),
                        'slowd_period': config.get("slowd_period", 3)
                    }
                elif func_name in ['MACD']:
                    args = {
                        'close': df['close'].interpolate(),
                        'fastperiod': config.get("fast_period", 12),
                        'slowperiod': config.get("slow_period", 26),
                        'signalperiod': config.get("signal_period", 9)
                    }
                if w:
                    args['timeperiod'] = w
                if w == 1 and len(columns) == 1 :
                    out = next(iter(columns.values()))
                else:
                    out = fn(**args)

            # Name of the output column
            if not w:
                if not names:
                    out_name = f"{column_all_names}_{func_name}"
                elif isinstance(names, str):
                    out_name = names
                elif isinstance(names, list):
                    out_name = names[i]
                else:
                    raise ValueError(f"Names must be string, or at least List.")
            else:
                out_name = f"{column_all_names}_{func_name}_"
                win_name = str(w)
                if not names:
                    out_name = out_name + win_name
                elif isinstance(names, str):
                    out_name = out_name + names + "_" + win_name
                elif isinstance(names, list):
                    out_name = out_name + names[i]
                else:
                    raise ValueError(f"Names must be string, or at least List.")
                
            fn_out_names.append(out_name)

            # if isinstance(out, tuple):
            #     out = out[0]
            # out.name = out_name

            # fn_outs.append(out)

            if isinstance(out, tuple):
                for i, element in enumerate(out):
                    element.name = f"{out_name}_{i}"  # Assign a unique name for each element
                    fn_outs.append(element)
            else:
                out.name = out_name
                fn_outs.append(out)

        # Convert to relative values and percentage
        fn_outs = _convert_to_relative(fn_outs, relative_base, relative_function, percentage)
        features.extend(fn_out_names)
        outputs.extend(fn_outs)
        print(outputs)

    for output in outputs:
        df[output.name] = np.log(out) if log else out
    
    return features

def _convert_to_relative(fn_outs: list, relative_base, relative_function, percentage):
    relative_outputs = []
    size = len(fn_outs)
    for i, feature in enumerate(fn_outs):
        if not relative_base:
            relative_output = fn_outs[i] # No change
        elif (relative_base == "next" or relative_base == "last") and i == size - 1:
            relative_output = fn_outs[i] # No change, because last rows
        elif (relative_base == "prev" or relative_base == "first") and i == 0:
            relative_output = fn_outs[i] # No change, because first rows
        
        elif relative_base == "next" or relative_base == "last":
            if relative_base == "next":
                base = fn_outs[i+1]
            elif relative_base == "last":
                base = fn_outs[size - 1]
            else:
                raise ValueError(f"Unknown value of the 'relative_function' config parameter: {relative_function}")
            
            if relative_function == "rel":
                relative_output = feature / base
            elif relative_function == "diff":
                relative_output = feature - base
            elif relative_function == "rel_diff":
                relative_output = (feature - base) / base
            else:
                raise ValueError(f"Unknown value of the 'relative_function' config parameter: {relative_function}")
        
        elif relative_base == "prev" or relative_base == "first":
            if relative_base == "prev":
                base = fn_outs[i-1]
            elif relative_base == "first":
                base = fn_outs[0]
            else:
                raise ValueError(f"Unknown value of the 'rel_base' config parameter: {relative_base=}")
            
            if relative_function == "rel":
                relative_output = feature / base
            elif relative_function == "diff":
                relative_output = feature - base
            elif relative_function == "rel_diff":
                relative_output = (feature - base) / base
            else:
                raise ValueError(f"Unknown value of the 'relative_function' config parameter: {relative_function}")

        
        if percentage:
            relative_output = relative_output * 100.0
            
        relative_output.name = fn_outs[i].name
        relative_outputs.append(relative_output)
        
    return relative_outputs