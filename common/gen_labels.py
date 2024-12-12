import pandas as pd

"""
We want to label each timestamp to be top, or bottom price.
1. We do that by leverage neighbor timestamp, if the current timestamp is 
higher/lower than a threshold, it may be considered a top/bottom price.

    The threshold is price diff percentage comparing to the top/bottom to other timestamp
2. After we define the top/bottom, we would care the interval around that top/bottom. So a
scores, called tolerance will come to play.
"""

def generate_labels(df, config: dict):
    init_column_number = len(df.columns)

    column_name = config.get('columns')
    if not column_name:
        raise ValueError(f"The 'columns' parameter must not be NULL.")
    elif not isinstance(column_name, str):
        raise ValueError(f"The 'columns' parameter must be string. Current type: {type(column_name)}")
    elif column_name not in df.columns:
        raise ValueError(f"{column_name} does not exist in the input data.")
    
    function = config.get('function')
    if not isinstance(function, str):
        raise ValueError(f"The 'function' parameter must be string. Current type: {type(function)}")
    if function not in ['top', 'bot']:
        raise ValueError(f"Unknown function name {function}. Only 'top' or 'bot' are possible.")

    tolerances = config.get('tolarance') # Percentage of tolerance
    if not isinstance(tolerance, list):
        tolerance = [tolerance]

    level = config.get('level') # Threshold to be consider top/bot
    if function == 'top':
        level = abs(level)
    elif function == 'bot':
        level = -abs(level)
    else:
        raise ValueError(f"Level has only 2 options: top, bot")
    
    names = config.get('names') # ['top1_025', 'top1_01']
    if len(names) != len(tolerance):
        raise ValueError(f"Label generator has name for each tolerance value")
    
    labels = []
    for i, tolerance in enumerate(tolerances):
        df, new_labels = add_extremum_features(df, column_name=column_name, 
                                               level_fracs=[level], 
                                               tolerance_frac=abs(level)*tolerance, 
                                               out_name=names[i:i+1])
        labels.extend(new_labels)

    print(f"{len(names)} labels generated: {names}")

    labels = df.columns.to_list()[init_column_number:]

    return df, labels

def add_extremum_features(df, column_name: str, level_fracs: list, tolerance_frac: float, out_names: list):
    column = df[column_name]
    out_columns = []
    for i, level_frac in enumerate(level_fracs):
        if level_frac > 0.0:
            extremums = find_all_extremums(column, True, level_frac, tolerance_frac)
        else:
            extremums = find_all_extremums(column, False, -level_frac, tolerance_frac)

        out_name = out_names[i]
        out_column = pd.Series(data=False, index=df.index, dtype=bool, name=out_name)

        for extremum in extremums:
            out_column.loc[extremum[1]:extremum[3]] = True
        
        out_columns.append(out_column)

    df = pd.concat([df] + out_columns, axis=1)

    return df, out_names

def find_all_extremums(sr: pd.Series, is_max: bool, level_frac: float, tolerance_frac: float) -> list:
    extremums = list()

    intervals = [(sr.index[0], sr.index[-1] + 1)] # All intervals needed to be analyzed to find top/bot
    while True:
        if not intervals:
            break
        interval = intervals.pop()

        extremum = find_one_extremum(sr.loc[interval[0] : interval[1]], is_max, level_frac, tolerance_frac)
        
        if extremum[0] and extremum[-1]: # If found then store to return
            extremums.append(extremum)
        
        if extremum[0] and interval[0] < extremum[0]:
            intervals.append((interval[0], extremum[0]))
        if extremum[-1] and extremum[-1] < interval[1]:
            intervals.append((extremum[-1], interval[1]))

    return sorted(extremums, key=lambda x: x[2])

def find_one_extremum(sr: pd.Series, is_max: bool, level_frac: float, tolerance_frac: float) -> tuple:
    if is_max:
        extr_idx = sr.idxmax()
        extr_val = sr.loc[extr_idx]
        level_val = extr_val * (1 - level_frac)
        tolerance_val = extr_val * (1 - tolerance_frac)

    else:
        extr_idx = sr.idxmin()
        extr_val = sr.loc[extr_idx]
        level_val = extr_val / (1 - level_frac)
        tolerance_val = extr_val / (1 - tolerance_frac)
    
    # Split into 2 sub-intervals to find the left and right ends
    sr_left = sr.loc[:extr_idx]
    sr_right = sr.loc[extr_idx:]

    # Check threshold
    left_level_idx = _left_level_idx(sr_left, is_max, level_val)
    right_level_idx = _right_level_idx(sr_right, is_max, level_val)

    # Find tolerance interval
    left_tolerance_idx = _left_level_idx(sr_left, is_max, tolerance_val)
    right_tolerance_idx = _right_level_idx(sr_right, is_max, tolerance_val)

    return (left_level_idx, left_tolerance_idx, extr_idx, right_tolerance_idx, right_level_idx)

def _left_level_idx(sr_left: pd.Series, is_max: bool, level_val: float):
    if is_max:
        sr_left_level = sr_left[sr_left < level_val]
    else:
        sr_left_level = sr_left[sr_left > level_val]
    
    if len(sr_left_level) > 0:
        left_idx = sr_left_level.index[-1]
    else:
        left_idx = None

    return left_idx

def _right_level_idx(sr_right: pd.Series, is_max: bool, level_val: float):
    if is_max:
        sr_right_level = sr_right[sr_right < level_val]
    else:
        sr_right_level = sr_right[sr_right > level_val]
    
    if len(sr_right_level) > 0:
        right_idx = sr_right_level.index[0]
    else:
        right_idx = None

    return right_idx