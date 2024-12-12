from typing import List
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import lightgbm as lgbm
import tensorflow as tf
from tensorflow import keras

def train_gb(df_X, df_y, model_config:dict):
    """
    Trains a gradient boosting model using LightGBM.
    Args:
        df_X (pandas.DataFrame): The input features dataframe.
        df_y (pandas.Series): The target variable series.
        model_config (dict): The configuration dictionary for the model.
    Returns:
        tuple: A tuple containing the trained model and the scaler used for feature scaling.
    """
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        max_shift = max(shifts)
        df_X = double_columns(df_X, shifts)
        df_X = df_X.iloc[max_shift:]
        df_y = df_y.iloc[max_shift:]

    # Scaling
    is_scale = model_config.get("train", {}).get("is_scale", False)
    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    # Model
    params = model_config.get("params")
    objective = params.get("objective")
    max_depth = params.get("max_depth")
    learning_rate = params.get("learning_rate")
    num_boost_round = params.get("num_boost_round")
    lambda_l1 = params.get("lambda_l1")
    lambda_l2 = params.get("lambda_l2")

    lgbm_params = {
        'learning_rate': learning_rate,
        'objective': objective,
        'min_data_in_leaf': int(0.02*len(df_X)),
        'num_leaves': 32, # TODO: Finetune
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'is_unbalance': 'true',
        'boosting_type': 'gdbt',
        'objective': objective,
        'metric': {'cross_entropy'},
        'verbose': 0,
        }
    
    model = lgbm.train(lgbm_params, lgbm.Dataset(X_train, y_train), num_boost_round=num_boost_round)

    return (model, scaler)

def predict_gb(models: tuple, df_X_test, model_config: dict):
    """
    Predicts the target variable using a gradient boosting model.
    Args:
        models (tuple): A tuple containing the trained gradient boosting model and the scaler.
        df_X_test: The input data frame for prediction.
        model_config (dict): A dictionary containing the model configuration.
    Returns:
        pd.Series: The predicted values of the target variable.
    """
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        df_X_test = double_columns(df_X_test, shifts)

    scaler = models[1] 
    is_scale = scaler is not None

    input_index = df_X_test.index
    if is_scale:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)
    else:
        df_X_test = df_X_test
    
    df_X_test_nonans = df_X_test.dropna()
    nonans_index = df_X_test_nonans.index
    
    y_test_hat_nonans = models[0].predict(df_X_test_nonans.values)
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)

    df_ret = pd.DataFrame(index=input_index)
    df_ret["y_hat"] = y_test_hat_nonans
    sr_ret = df_ret["y_hat"]

    return sr_ret
def train_predict_gb(df_X, df_y, df_X_test, model_config: dict):
    """
    Trains a gradient boosting model using the given training data and predicts the target variable for the given test data.

    Args:
        df_X (pandas.DataFrame): The training features.
        df_y (pandas.Series): The training target variable.
        df_X_test (pandas.DataFrame): The test features.
        model_config (dict): A dictionary containing the configuration parameters for the gradient boosting model.

    Returns:
        pandas.Series: The predicted values for the test data.

    """
    model_pair = train_gb(df_X, df_y, model_config)
    y_test_hat = predict_gb(model_pair, df_X_test, model_config)
    
    
    return y_test_hat

#TODO: NN model
def predict_nn():
    pass

def compute_scores(y_true, y_hat):
    """
    Computes the scores for the predicted values.
    Args:
        y_true (pandas.Series): The true values.
        y_hat (pandas.Series): The predicted values.
    Returns:
        dict: A dictionary containing the computed scores.
    """
    y_true = y_true.astype(int)
    y_hat_class = np.where(y_hat.values > 0.5, 1, 0) 

    try:
        auc = metrics.roc_auc_score(y_true, y_hat.fillna(0))
    except ValueError:
        auc = 0.0

    try:
        ap = metrics.average_precision_score(y_true, y_hat.fillna(0))
    except ValueError:
        ap = 0.0

    f1 = metrics.f1_score(y_true, y_hat_class)
    precision = metrics.precision_score(y_true, y_hat_class)
    recall = metrics.recall_score(y_true, y_hat_class)


    scores = dict(
        auc=auc,
        ap=ap,
        f1=f1,
        precision=precision,
        recall=recall
    )    
    return scores

def double_columns(df_X, shifts):
    if not shifts:
        return df_X
    df_list = [df_X.shift(shift) for shift in shifts]
    df_list.insert(0, df_X)

    df_X = pd.concat(df_list, axis=1)

    return df_X

if __name__ == "__main__":
    pass