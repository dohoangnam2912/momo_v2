from pathlib import Path
import click

import pandas as pd
from common.generators import generate_feature_set
from server.App import *

class P:
    input_nrows = 100_000_000
    start_index = 0
    end_index = None

@click.command()
@click.option("--config_file", "-c", type=click.Path(), default="", help="Path to configuration file")
def main(config_file):
    load_config(config_file)
    time_column = App.config["time_column"]
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        raise Exception(f"Data folder not found: {data_path}")
    output_path = Path(App.config["data_folder"]) / symbol 
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = data_path / App.config.get("predict_file_name")
    if not file_path.exists():
        raise Exception(f"Predict file not found: {file_path}")
    
    print(f"Loading predict file: {file_path}")
    df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.input_nrows)
    df = df[P.start_index:P.end_index]
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} rows.")

    signal_sets = App.config.get("signal_sets", [])
    if not signal_sets:
        raise Exception("No signal sets found in configuration file.")
    
    print(f"Start generating signals for {len(signal_sets)} signal sets.")

    all_signals = []
    for i, ss in enumerate(signal_sets):
        print(f"Generating signals for signal set {i+1}/{len(signal_sets)}")
        df, new_signals = generate_feature_set(df, ss, last_rows=0)
        all_signals.extend(new_signals)
    
    print(f"Finished generating {len(all_signals)} signals.")

    output_columns = ["timestamp", "open", "high", "low", "close"] 
    output_columns.extend(App.config.get("labels"))
    output_columns.extend(all_signals)

    output_df = df[output_columns]
    output_path = data_path / "signals.csv"
    print(f"Storing signals to {output_path}")
    output_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()