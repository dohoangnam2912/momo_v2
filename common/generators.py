import pandas as pd
from typing import Tuple
from common.classifiers import *
from common.gen_features import *
from common.gen_features_agg import *
from common.gen_labels import generate_labels
from common.gen_signals import *
from common.model_store import *

def generate_feature_set(df: pd.DataFrame, fs:dict, last_rows: int) -> Tuple[pd.DataFrame, list]:
    cp = fs.get("column_prefix")
    if cp:
        cp = cp + "_"
        f_cols = [col for col in df if col.startswith(cp)]
        f_df = df[f_cols]
        f_df = f_df.rename(columns=lambda x: x[len(cp):] if x.startswith(cp) else x) # Removing prefix
    else:
        f_df = df[df.columns.to_list()]
    
    generator = fs.get("generator")
    gen_config = fs.get("config", {})

    # Features
    if generator == "tsfresh":
        print(f"Generating features with {generator}...")
        features = generate_features_tsfresh(f_df, gen_config, last_rows=last_rows)
    elif generator == "talib":
        print(f"Generating features with {generator}...")
        features = generate_features_talib(f_df, gen_config, last_rows=last_rows)
        print(f"Finished generating features with {generator}. Name of the columns: {features}")
    elif generator == "log_diff":
        features = to_log_diff(f_df, gen_config)
    elif generator == 'shift':
        features = shift(f_df, gen_config)
    elif generator == 'body':
        features = candle_body(f_df)
    elif generator == 'z_score':
        features = z_score(f_df)
    elif generator == 'candle_pattern':
        features = candle_pattern(f_df)
    # Labels
    elif generator == "label":
        f_df, features = generate_labels(f_df, gen_config)

    # Signals
    elif generator == "smoothen":
        pass
    elif generator == "combine":
        f_df, features = generate_scores(f_df, gen_config)
    elif generator == "threshold_rule":
        f_df, features = generate_threshold_rule(f_df, gen_config)
    else:
        generator_fn = resolve_generator_name(generator)
        if generator_fn is None:
            raise ValueError(f'Unknown feature generator: {generator}.')

        f_df, features = generator_fn(f_df, gen_config)

    print(f_df.head())
    print(features)
    f_df = f_df[features]
    fp = fs.get("feature_prefix")
    if fp:
        f_df = f_df.add_prefix(fp + "_")

    new_features = f_df.columns.to_list()

    df.drop(list(set(df.columns) & set(new_features)), axis=1, inplace=True) # Delete new columns if they already exist

    df = df.join(f_df) # Attach all derived features to the main frame

    return df, new_features

def predict_feature_set(df: pd.DataFrame, fs:dict, config, models:dict):
    # TODO: Change this better, not so many if else
    
    labels = fs.get("config").get("labels")
    if not labels:
        labels = config.get("labels")
    
    algorithms = fs.get("config").get("functions")
    if not algorithms:
        algorithms = fs.get("config").get("algorithms")
    if not algorithms: # If still not found
        algorithms = config.get("algorithms")
    
    train_features = fs.get("config").get("columns")
    if not train_features:
        train_features = fs.get("config").get("features")
    if not train_features:
        train_features = config.get("train_features")
    
    train_df = df[train_features]

    features = []
    scores = dict()
    output_df = pd.DataFrame(index=train_df.index)

    for label in labels:
        for model_config in algorithms:
            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = label + label_algo_seperator + algo_name

            model_pair = models.get(score_column_name) # Trained model frrom model registry

            print(f"Predict '{score_column_name}'. Algorithm {algo_name}. Label: {label}. Train length {len(train_df)}. Train columns {len(train_df.columns)}")

            if algo_type == "gb":
                df_y_hat = predict_gb(model_pair, train_df, model_config)
            elif algo_type == "nn":
                df_y_hat = predict_nn(model_pair, train_df, model_config)
            else:
                raise ValueError(f"Only support LightGBM and Neural Network model. Your current model: {algo_type}")
            
            output_df[score_column_name] = df_y_hat
            features.append(score_column_name)

            if label in df:
                scores[score_column_name] = compute_scores(df[label], df_y_hat)
    return output_df, features, scores

def resolve_generator_name(gen_name: str):
    """
    Example, fn = resolve_generator_name("common.gen_features:get_labels:etc")
    """
    mod_and_func = gen_name.split(':', 1)
    mod_name = mod_and_func[0] if len(mod_and_func) > 1 else None
    func_name = mod_and_func[-1]
    
    if not mod_name:
        return None
    
    try: 
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return None
    if mod is None:
        return None
    
    try:
        func = getattr(mod, func_name)
    except AttributeError as e:
        return None
    
    return func

def train(df, fs, config):
    labels = fs.get("config").get("labels")
    if not labels:
        labels = config.get("labels")

    algorithms = fs.get("config").get("functions")
    if not algorithms:
        algorithms = fs.get("config").get("algorithms")
    if not algorithms: # If still not found
        algorithms = config.get("algorithms")
    
    train_features = fs.get("config").get("columns")
    if not train_features:
        train_features = fs.get("config").get("features")
    if not train_features:
        train_features = config.get("train_features")

    models = dict()
    scores = dict()
    out_df = pd.DataFrame()

    for label in labels:
        for model in algorithms:
            algo_name = model.get("name")
            algo_type = model.get("algo")
            score_column_name = label + label_algo_seperator + algo_name
            algo_train_length = model.get("train", {}).get("length")


            if algo_train_length:
                train_df = df.tail(algo_train_length)
            else:
                train_df = df
            
            df_x = train_df[train_features]
            df_y = train_df[label]

            print(f"Train '{score_column_name}'. Algorithm {algo_name}. Label: {label}. Train length {len(train_df)}. Train columns {len(train_df.columns)}")

            if algo_type == "gb":
                model_pair = train_gb(df_x, df_y, model)
                models[score_column_name] = model_pair
                df_y_hat = predict_gb(model_pair, df_x, model)
            elif algo_type == "nn":
                model_pair = train_nn(df_x, df_y, model)
                models[score_column_name] = model_pair
                df_y_hat = predict_nn(model_pair, df_x, model)
            else:
                raise ValueError(f"Only support LightGBM and Neural Network model. Your current model: {algo_type}")
            
            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            out_df[score_column_name] = df_y_hat
    return out_df, models, scores