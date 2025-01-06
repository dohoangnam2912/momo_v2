from pathlib import Path

import pandas as pd
import click

from server.App import *
from source.features import generate_feature_set

class P:
    input_nrows = 100_000_000
    tail_rows = 0

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file directory')
def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    symbol = App.config["symbol"]
    data_path = Path(App.config['data_folder']) / symbol

    file_path = data_path / App.config.get("feature_file_name")

    if not file_path.is_file():
        raise ValueError(f"Data file does not exist: {file_path}")
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.input_nrows)
    else:
        raise ValueError(f"Unknown extension of the 'feature_file_name' file")
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range[{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    label_sets = App.config.get("label_sets", [])
    if not label_sets:
        raise ValueError("ERROR: no label sets defined.")
    print(f"Start generating labels for {len(df)} input records.")

    all_features = []
    for i, fs in enumerate(label_sets):
        print(f"Start label set {i}/{len(label_sets)}.")
        df, new_features = generate_feature_set(df, fs, last_rows=0)
        all_features.extend(new_features)
        print(f"Finished label set {i}/{len(label_sets)}.")

    print(f"Finished generating labels.")

    output_file_name = App.config.get("label_file_name")
    output_path = (data_path / output_file_name).resolve()

    print(f"Storing file with labels.")

    df.to_csv(output_path, index=False, float_format="%.6f")

    print(f"Stored file with labels in {output_path}.")

    with open(output_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f'"{f}"' for f in all_features]) + "\n")

    print(f"Finished generating labels for {len(df)} records in {datetime.now() - now}.")

if __name__ == "__main__":
    main()