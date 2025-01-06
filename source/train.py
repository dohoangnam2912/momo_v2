from pathlib import Path
from datetime import datetime
import click

import numpy as np
import pandas as pd
import shap

from server.App import *
from common.gen_features import *
from common.classifiers import *
from common.model_store import *
from common.generators import train
import matplotlib.pyplot as plt



class P:
    input_nrows = 100_000_000
    tail_rows = 0
    store_predictions = True

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default="", help='Config file directory')
def main(config_file):
    load_config(config_file)

    time_column = App.config['time_column']
    symbol = App.config['symbol']
    data_path = Path(App.config['data_folder']) / symbol

    file_path = data_path / App.config.get('label_file_name')
    if not file_path.is_file():
        raise ValueError(f"Label file does not exist: {file_path}")
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format='ISO8601', nrows=P.input_nrows)
    else:
        raise ValueError(f"Unknown extension of the 'label_file_name' file. Only .csv is supported.")
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:]
    df = df.reset_index(drop=True)
    label_horizon = App.config["label_horizon"]
    train__length = App.config["train_length"]
    train_features = App.config["train_features"]
    labels = App.config["labels"]
    
    # TODO:
    output_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    output_columns = [x for x in output_columns if x in df.columns]
    all_features = train_features + labels
    df = df[output_columns + [x for x in all_features if x not in output_columns]]
    
    for label in labels:
        df[label] = df[label].astype(int) # Changed for NN category
    
    if label_horizon:
        df = df.head(-label_horizon)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=train_features)
    if len(df) == 0:
        raise ValueError("ERROR: No data left after removing NaNs.")
    
    if train__length:
        df = df.tail(train__length)
    
    df = df.reset_index(drop=True)

    train_feature_sets = App.config.get("train_feature_sets", [])
    if not train_feature_sets:
        raise ValueError("ERROR: No feature sets defined.")
    
    print(f"Start training model for {len(df)} input records.")

    output_df = pd.DataFrame()
    models = dict()
    scores = dict()

    for i, fs in enumerate(train_feature_sets):
        print(f"Start training feature set {i}/{len(train_feature_sets)}.")
        fs_output, fs_model, fs_score = train(df, fs, App.config)
        output_df = pd.concat([output_df, fs_output], axis=1)
        models.update(fs_model)
        scores.update(fs_score)
        print(f"Finished training feature set {i}/{len(train_feature_sets)}. With scores: {fs_score}")

    print(f"Finished training model.")
    print(models)
    for model_name, (model, scaler) in models.items():
        print(f"Model Name: {model_name}, Num Features: {len(model.feature_name())}")
        
        # Extract feature importance using LightGBM
        feature_importances = model.feature_importance(importance_type='gain')
        features = model.feature_name()
        
        # Create a DataFrame of feature importances
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance for {model_name}')
        plt.gca().invert_yaxis()
        
        # Save the plot with a unique filename based on the model name
        filename = f'feature_importance_{model_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved as '{filename}'")


    model_path = Path(App.config["model_folder"]) / symbol
    if not model_path.is_absolute():
        model_path = data_path / model_path

    model_path = model_path.resolve()
    model_path.mkdir(parents=True, exist_ok=True)

    for score_name, model_pair in models.items():
        save_model_pair(model_path, score_name, model_pair)
    
    print(f"Models stored in path: {model_path}")

    # Storing scores

    lines = list()
    for score_colume_name, score in scores.items():
        lines.append(f"{score_colume_name}: {score}")
    
    metrics_path = (model_path / 'metrics.txt').resolve()
    with open(metrics_path, 'w') as f:
        f.write('\n'.join(lines) + "\n")

    print(f"Stroring metrics in path: {metrics_path}")

    # Storing predictions

    output_df = output_df.join(df[output_columns + labels])
    output_path = data_path / App.config.get("predict_file_name")
    output_df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Predictions stored in path: {output_path}")
    
if __name__ == '__main__':
    main()