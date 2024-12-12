from pathlib import Path
import click
import pandas as pd
import numpy as np
from server.App import *
from common.generators import generate_feature_set

class P:
    input_nrows = 50000000 # Load only this number of rows
    tail_rows = int(10.0 * 525600) # Process only this number of last rows

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default="", help='Configuration file name')
def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    symbol = App.config["symbol"]
    data_path = App.config["data_folder"] / symbol
    file_path = data_path / App.config.get("merge_file_name")

    if not file_path.is_file():
        raise ValueError(f"Data file does not exist in {file_path}")
    
    print(f"Loading data from source data file {file_path}")
    df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.input_nrows)
    
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    feature_sets = App.config.get("feature_sets", [])
    if not feature_sets:
        raise ValueError(f"No feature sets founded!")
    
    print(f"Start generating features for {len(df)} input records.")

    all_features = []

    for i, fs in enumerate(feature_sets):
        print(f"Start feature set {i}/{len(feature_sets)}. Generator {fs.get('generator')}...")
        df, new_features = generate_feature_set(df, fs, last_rows=0)
        all_features.extend(new_features)
        print(f"Finished feature set {i}/{len(feature_sets)}. Generator {fs.get('generator')}")

    print(f"Finished generating features.")

    print(f"Number of NULL values:")
    print(df[all_features].isnull().sum().sort_values(ascending=False))

    output_file_name = App.config.get("feature_file_name")
    output_path = (data_path / output_file_name).resolve()

    print(f"Storing features with {len(df)} records and {len(df.columns)} columns in output file {output_path}")
    df.to_csv(output_path, index=False, float="%.6f")

    print(f"Stored output file {output_path}!")

    with open(output_path.with_suffix('.txt'), '+a') as f:
        f.write(", ".join([f'"{f}' for f in all_features])+ "\n\n")

    print(f"Stored {len(all_features)} features in output file {output_path.with_suffix('.txt')}. Finished generating")

if __name__ == "__main__":
    main()