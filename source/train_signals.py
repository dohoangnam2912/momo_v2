from pathlib import Path
import click
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import ParameterGrid
from server.App import *
from common.utils import *
from common.gen_signals import *
from common.classifiers import *
from common.generators import generate_feature_set

"""
Find the best trade parameters by executing backtesting
"""

class P:
    in_nrows = 100_000_000

@click.command()
@click.option("--config_file", '-c', type=click.Path(), default='', help='Configuration file name0')
def main(config_file):
    load_config(config_file)
    time_column = App.config['time_column']
    symbol = App.config['symbol']
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        raise ValueError(f"Data folder does not exist: {data_path}")
    out_path = Path(App.config["data_folder"]) / symbol
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = data_path / "signals.csv"
    if not file_path.exists():
        raise(f"Signals file does not exist in directory: {file_path}")
    
    print(f"Loading signals file.")
    df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    print("Signals file loaded.")

    train_signal_config = App.config["train_signal_model"]
    data_start = train_signal_config.get("data_start", 0)
    if isinstance(data_start, str):
        data_start = find_index(df, data_start)
    data_end = train_signal_config.get("data_end", None)
    if isinstance(data_end, str):
        data_end = find_index(df, data_end)

    df = df.iloc[data_start:data_end]
    df = df.reset_index(drop=True)


    print(f"Input data size {len(df)} records loaded.")
    df = df[-34561:-1]
    months_in_simulation = (df[time_column].iloc[-1] - df[time_column].iloc[0]) / timedelta(days=365/12)
    print(f"Months in simulation: {months_in_simulation:.2f} from {df[time_column].iloc[0]} to {df[time_column].iloc[-1]}")
    parameter_grid = train_signal_config.get("grid")
    direction = train_signal_config.get("direction", "")
    if direction not in ['long', 'short', 'both', '']:
        raise ValueError(f"Unknown value of {direction} in signal train model. Only long, short and both is avaiable.")
    
    topn_to_store = train_signal_config.get("topn_to_store", 10)

    if isinstance(parameter_grid.get("buy_signal_threshold"), str):
        parameter_grid["buy_signal_threshold"] = eval(parameter_grid.get("buy_signal_threshold"))
    if isinstance(parameter_grid.get("buy_signal_threshold_2"), str):
        parameter_grid["buy_signal_threshold_2"] = eval(parameter_grid.get("buy_signal_threshold_2"))
    if isinstance(parameter_grid.get("sell_signal_threshold"), str):
        parameter_grid["sell_signal_threshold"] = eval(parameter_grid.get("sell_signal_threshold"))
    if isinstance(parameter_grid.get("sell_signal_threshold_2"), str):
        parameter_grid["sell_signal_threshold_2"] = eval(parameter_grid.get("sell_signal_threshold_2"))
    
    generator_name = train_signal_config.get("signal_generator")
    print(f"generator_name {generator_name}")
    signal_generator = next((ss for ss in App.config.get("signal_sets", []) if ss.get('generator') == generator_name), None)
    print(signal_generator)
    if not signal_generator:
        raise ValueError(f"Signal generator {generator_name} not found.")
    
    performances = list()
    for parameters in tqdm(ParameterGrid([parameter_grid]), desc="MODELS"):
        signal_generator["config"]["parameters"].update(parameters)

        df, _ = generate_feature_set(df, signal_generator, last_rows=0)

        buy_signal_column = signal_generator["config"]["names"][0]
        sell_signal_column = signal_generator["config"]["names"][1]
        print(f"Buy signal column: {buy_signal_column}")
        print(f"Sell signal column: {sell_signal_column}")
        performance, long_performance, short_performance = simulated_trade_performance(df,  buy_signal_column, sell_signal_column, 'close')

        long_performance.pop("transactions", None)
        short_performance.pop("transactions", None)

        if direction == "long":
            performance = long_performance
        elif direction == "short":
            performance = short_performance
        
        performance["profit_percent"] = performance["profit_percent"]
        performance["transaction"] = performance["transaction_no"]
        performance["profit_percent_per_transaction"] = performance["profit_percent"] / months_in_simulation
        performance["profit"] = performance["profit"]

        performances.append(dict(
            model=parameters,
            performance={k: performance[k] for k in ['profit_percent', 'profitable', 'profit_percent_per_transaction', 'transaction_no']}
        ))

    performances = sorted(performances, key=lambda x: x['performance']['profit_percent'], reverse=True)
    performances = performances[:topn_to_store]

    keys = list(performances[0]['model'].keys()) + list(performances[0]['performance'].keys())
    lines = []
    for p in performances:
        record = list(p['model'].values()) + list(p['performance'].values())
        record = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in record]
        record_str = ",".join(record)
        lines.append(record_str)

    out_path = (out_path / App.config.get("signal_models_file_name")).with_suffix(".csv").resolve()
    
    if out_path.is_file():
        add_header = False
    else:
        add_header = True
    with open(out_path, "a+") as f:
        if add_header:
            f.write(",".join(keys) + "\n")
        f.write("\n".join(lines) + "\n\n")

    print(f"Simulation results stored in: {out_path}")

if __name__ == "__main__":
    main()