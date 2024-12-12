from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

import click

from server.App import *

"""
Creating one output file from multiple input data files.
For example, in order to predict BTC price, we might want to add ETH prices
"""

depth_file_names = [
]

def load_kline_files(kline_file_path):
    df = pd.read_csv(kline_file_path, parse_dates=['timestamp'], date_format="ISO8601")
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]

    df = df.set_index("timestamp")

    print(f"Loaded kline file with {len(df)} records in total. Range: ({start}, {end})")

    return df, start, end

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file directory')
def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"]

    data_sources = App.config.get("data_source", [])
    if not data_sources:
        raise ValueError(f"Data sources are not defined. Nothing to merge.")
    
    data_path = Path(App.config["data_folder"])
    for ds in data_sources:
        symbol = ds.get("folder")
        if not symbol:
            raise ValueError("The folder is not specified.")

        file = ds.get("file", symbol)
        if not file:
            file = symbol

        file_path = (data_path / symbol / file).with_suffix(".csv")
        if not file_path.is_file():
            raise ValueError(f"Data file does not exist: {file_path}")
        
        print(f"Reading data file in directory: {file_path}")
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
        
        ds["df"] = df #TODO: Explain

    df_out = merge_data_sources(data_sources)

    output_path = data_path / App.config["symbol"] / App.config.get("merge_file_name")

    print(f"Storing output file...")
    df_out = df_out.reset_index()
    df_out.to_csv(output_path, index=False)

    print(f"Finishied merging data, storing output file in {output_path}.")

def merge_data_sources(data_sources: list):
    time_column = App.config["time_column"]
    freq = App.config["freq"]

    for ds in data_sources:
        df = ds.get("df")
        if time_column in df.columns:
            df = df.set_index(time_column)
        elif df.index.name == time_column:
            pass
        else:
            raise(f"Timestamp column is not available!")
        
        if ds['column_prefix']:
            df.columns = [
                ds['column_prefix'] + "_" + col if not col.startwith(ds['column_prefix'] + "_") else col for col in df.columns 
            ]

        ds["start"] = df.first_valid_index()
        ds["end"] = df.last_valid_index()
        ds["df"] = df

    range_start = min([ds["start"] for ds in data_sources])
    range_end = min([ds["end"] for ds in data_sources])

    index = pd.date_range(range_start, range_end, freq=freq)

    df_out = pd.DataFrame(index=index)
    df_out.index.name = time_column

    for ds in data_sources:
        df_out = df_out.join(ds["df"])
    return df_out

if __name__ == '__main__':
    main()