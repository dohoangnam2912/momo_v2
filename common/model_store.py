import itertools

from pathlib import Path
from joblib import dump, load
from keras.models import save_model, load_model

label_algo_seperator = "_"

"""
Module for storing and loading machine learning models.
"""

def save_model_pair(model_path, score_column_name: str, model_pair: tuple):
    """
    Save a model pair consisting of a model and a scaler to the specified model path.
    Args:
        model_path (str or Path): The path where the model pair will be saved.
        score_column_name (str): The name of the score column.
        model_pair (tuple): A tuple containing the model and scaler to be saved.
    Raises:
        None
    Returns:
        None
    """
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    model_path = model_path.absolute()

    model = model_pair[0]
    scaler = model_pair[1]
    
    # Saving scaler
    scaler_file_name = (model_path / score_column_name).with_suffix(".scaler")
    dump(scaler, scaler_file_name)

    # Saving model
    if score_column_name.endswith("_nn"):
        model_file_name = (model_path / score_column_name).with_suffix(".h5")
        save_model(model, model_file_name)
    else:
        model_file_name = (model_path / score_column_name).with_suffix(".pickle")
        dump(model, model_file_name)

def load_model_pair(model_path, score_column_name: str):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    model_path = model_path.absolute()

    # Loading scaler
    scaler_file_name = (model_path / score_column_name).with_suffix(".scaler")
    scaler = load(scaler_file_name)

    # Loading model
    if score_column_name.endswith("_nn"):
        model_file_name = (model_path / score_column_name).with_suffix(".h5")
        model = load_model(model_file_name)
    else:
        model_file_name = (model_path / score_column_name).with_suffix(".pickle")
        model = load(model_file_name)

    return (model, scaler)

def load_models(model_path, labels: list, algorithms: list):
    models = dict()
    for label_algorithm in itertools.product(labels, algorithms):
        score_column_name = label_algorithm[0] + label_algo_seperator + label_algorithm[1]["name"]
        model_pair = load_model_pair(model_path, score_column_name)
        models[score_column_name] = model_pair
    return models

def score_to_label_algo_pair(score_column_name: str):
    label_name, algo_name = score_column_name.rsplit(label_algo_seperator, 1)
    return label_name, algo_name