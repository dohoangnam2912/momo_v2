import pandas as pd
import json
import time
from datetime import datetime
import click

from binance.client import Client
from binance import BinanceSocketManager
from binance.enums import *

from common.utils import klines_to_df, binance_freq_from_pandas

from server.App import *

"""
TODO: Write load_config function
"""


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')

def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"] # All configuration will be in App object

    data_path = Path(App.config["data_folder"])

    freq = App.config['freq']  # Pandas frequency, be careful about frequency between klines and pandas
    print(f"Pandas frequency: {freq}")

    freq = binance_freq_from_pandas((freq)) # Converting pandas freq to binance freqs
    print(f"Klines frequency: {freq}")

    save = True
    
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    # TODO: Add futures 

    data_sources = App.config['data_source']
    for ds in data_sources:
        # We are assuming that folder name is the symbol name we want to download
        symbol = ds.get("folder")
        if not symbol:
            raise Exception('ERROR. Folder is not specified')
        
        print(f"Start downloading data for {symbol}...")

        file_path = data_path / symbol
        file_path.mkdir(parents=True, exist_ok=True)

        file_directory = (file_path / "klines").with_suffix(".csv")

        # Get a few latest klines to determine the latest timestamp
        latest_klines = App.client.get_klines(symbol=symbol, interval=freq, limit=5)
        latest_timestamp = pd.to_datetime(latest_klines[-1][0], unit='ms')

        if file_directory.is_file():
            # If already downloaded data, perfoming below:
            # 1. Search for the latest timestamp
            # 2. Download data from that timestamp
            df = pd.read_csv(file_directory)
            df[time_column] = pd.to_datetime(df[time_column], format='ISO8601')

            latest_point = df[time_column].iloc[-5]

            print(f"Data file founded. Downloading data for symbol {symbol} since {str(latest_point)} and appending to the existing file {file_directory}")

        else:
            df = pd.DataFrame()
            latest_point = datetime(2017,1,1)
            print(f"No data file founded. Start downloading data for symbol {symbol} since 2017 January 1st and saves to the directory {file_directory}")

        # Downloading using binance client
        klines = App.client.get_historical_klines(symbol=symbol, interval=freq, start_str=latest_point.isoformat())

        df = klines_to_df(klines)

        df = df.iloc[:-1] # Remove last row because it's incompleted

        if save:
            df.to_csv(file_directory)
        
        print(f"Finishing downloading data for symbol {symbol}, saved in {file_directory}")

if __name__ == '__main__':
    main()