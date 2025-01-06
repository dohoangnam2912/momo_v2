from typing import List
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lightgbm as lgbm
from lightgbm import LGBMClassifier
import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

# def train_gb(df_X, df_y, model_config: dict):
#     """
#     Trains a gradient boosting model using LightGBM.
#     Args:
#         df_X (pandas.DataFrame): The input features dataframe.
#         df_y (pandas.Series): The target variable series.
#         model_config (dict): The configuration dictionary for the model.
#     Returns:
#         tuple: A tuple containing the trained model and the scaler used for feature scaling.
#     """
#     print(df_X.columns)
    
#     # Handle shifting of features
#     shifts = model_config.get("train", {}).get("shifts", None)
#     if shifts:
#         max_shift = max(shifts)
#         df_X = double_columns(df_X, shifts)
#         df_X = df_X.iloc[max_shift:]
#         df_y = df_y.iloc[max_shift:]

#     # Scaling
#     is_scale = model_config.get("train", {}).get("is_scale", False)
#     if is_scale:
#         scaler = StandardScaler()
#         scaler.fit(df_X)
#         X_train = scaler.transform(df_X)
#     else:
#         scaler = None
#         X_train = df_X.values

#     y_train = df_y.values

#     # Split into training and validation sets
#     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

#     # Model parameters with defaults
#     params = model_config.get("params", {})
#     lgbm_params = {
#         'learning_rate': params.get("learning_rate", 0.1),
#         'objective': params.get("objective", 'binary'),
#         'max_depth': params.get("max_depth", -1),
#         'num_leaves': 2**params.get("max_depth", -1) - 1,
#         'lambda_l1': params.get("lambda_l1", 0.0),
#         'lambda_l2': params.get("lambda_l2", 0.0),
#         'scale_pos_weight': len(y_train) / (sum(y_train)),  # Handle imbalance
#         'n_estimators': params.get("num_boost_round", 100),
#         'num_boost_round': params.get("num_boost_round", 100),
#     }

#     # Train the model
#     train_data = lgbm.Dataset(X_train, label=y_train, feature_name=df_X.columns.tolist())
#     valid_data = lgbm.Dataset(X_valid, label=y_valid, feature_name=df_X.columns.tolist())

#     model = lgbm.train(
#         lgbm_params,
#         train_set=train_data,
#         valid_sets=[train_data, valid_data],
#         callbacks=[
#             lgbm.early_stopping(50),
#         ]
#     )

#     return model, scaler

def train_gb(df_X, df_y, model_config: dict):
    """
    Trains a gradient boosting model using LightGBM with Bayesian Optimization for hyperparameter tuning.
    Args:
        df_X (pandas.DataFrame): The input features dataframe.
        df_y (pandas.Series): The target variable series.
        model_config (dict): The configuration dictionary for the model.
    Returns:
        tuple: A tuple containing the best model, the scaler used, and the best parameters found.
    """

    # Handle shifting of features
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

    # Split into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    # Define the objective function for Bayesian Optimization
    def objective(learning_rate, max_depth, num_leaves, lambda_l1, lambda_l2):
        params = {
            'learning_rate': learning_rate,
            'objective': 'binary',
            'max_depth': int(max_depth),
            'num_leaves': int(num_leaves),
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'scale_pos_weight': len(y_train) / (sum(y_train)),
            'n_estimators': 100
        }

        # Create datasets
        train_data = lgbm.Dataset(X_train, label=y_train, feature_name=df_X.columns.tolist())
        valid_data = lgbm.Dataset(X_valid, label=y_valid, feature_name=df_X.columns.tolist())

        # Train model with these hyperparameters
        model = lgbm.train(
            params,
            train_set=train_data,
            valid_sets=[train_data, valid_data],
            callbacks=[lgbm.early_stopping(50)]
        )

        # Evaluate using validation score (maximize F1 score or minimize log loss)
        preds = model.predict(X_valid)
        preds_binary = (preds > 0.5).astype(int)
        score = -np.mean((y_valid - preds_binary) ** 2)  # Example: Negative MSE as the score to minimize
        return score

    # Define the bounds for Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'num_leaves': (20, 100),
            'lambda_l1': (0, 10),
            'lambda_l2': (0, 10)
        },
        random_state=42
    )

    # Run Bayesian Optimization for 20 iterations
    optimizer.maximize(init_points=5, n_iter=20)

    # Retrieve the best parameters found
    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['num_leaves'] = int(best_params['num_leaves'])

    # Train the model with the best hyperparameters
    final_train_data = lgbm.Dataset(X_train, label=y_train, feature_name=df_X.columns.tolist())
    final_valid_data = lgbm.Dataset(X_valid, label=y_valid, feature_name=df_X.columns.tolist())

    final_model = lgbm.train(
        best_params,
        train_set=final_train_data,
        valid_sets=[final_train_data, final_valid_data],
        callbacks=[lgbm.early_stopping(50)]
    )

    print("Best Parameters:", best_params)
    return final_model, scaler


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
def train_nn():
    pass
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
    """
    Concatenates shifted versions of a DataFrame with the original DataFrame.

    Parameters:
    - df_X (pandas.DataFrame): The original DataFrame.
    - shifts (list): A list of integer values representing the number of shifts to apply to the DataFrame.

    Returns:
    - df_X (pandas.DataFrame): The concatenated DataFrame.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> shifts = [1, 2]
    >>> double_columns(df, shifts)
       A  B  A_shift_1  B_shift_1  A_shift_2  B_shift_2
    0  1  4        NaN        NaN        NaN        NaN
    1  2  5        1.0        4.0        NaN        NaN
    2  3  6        2.0        5.0        1.0        4.0
    """


    if not shifts:
        return df_X
    df_list = [df_X.shift(shift) for shift in shifts]
    df_list.insert(0, df_X)

    df_X = pd.concat(df_list, axis=1)

    return df_X

if __name__ == "__main__":
    pass