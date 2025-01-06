import numpy as np
import pandas as pd
from datetime import datetime

def generate_scores(df, config: dict):
    """
    Generate scores based on the given DataFrame and configuration.
    Args:
        df (pandas.DataFrame): The DataFrame to generate scores from.
        config (dict): The configuration dictionary containing the following keys:
            - 'columns' (list): A list of two column names to use for score generation.
            - 'name' (str): The name of the output column.
            - 'combine' (str): The method to combine the scores, either "relative" or "difference".
            - 'coefficients' (float or int, optional): Coefficients to multiply the scores by.
            - 'constant' (float or int, optional): Constant value to add to the scores.
    Returns:
        tuple: A tuple containing the modified DataFrame and a list of output column names.
    """
    columns = config.get('columns')
    if not columns:
        raise ValueError('columns is required in config')
    elif not isinstance(columns, list) or len(columns) != 2:
        raise ValueError('columns must be a list with 2 elements')
    
    up_column, down_column = columns[0], columns[1]
    out_column = 'trade_score'
    
    if config.get("combine") == "relative":
        combine_scores_relative(df, up_column, down_column, out_column)
    elif config.get("combine") == "difference":
        combine_scores_difference(df, up_column, down_column, out_column)
    else:
        raise ValueError('combine must be either "relative" or "difference"')
    
    if config.get('coefficients'):
        df[out_column] = df[out_column] * config.get('coefficients')
    if config.get('constant'):
        df[out_column] = df[out_column] + config.get('constant')
        
    return df, [out_column]

def combine_scores_relative(df, up_column, down_column, out_column):
    buy_plus_sell = df[up_column] + df[down_column]
    buy_sell_score = ((df[up_column] / buy_plus_sell) * 2) - 1.0

    df[out_column] = buy_sell_score

    return buy_sell_score

def combine_scores_difference(df, up_column, down_column, out_column):
    buy_sell_score = df[up_column] - df[down_column]

    df[out_column] = buy_sell_score

    return buy_sell_score

# Signals

def generate_threshold_rule(df, config):
    parameters = config.get("parameters", {})

    columns = config.get("columns")
    if not columns:
        raise ValueError(f"The 'columns' parameter must be a non-empty string. {type(columns)}")
    elif isinstance(columns, list):
        columns = [columns]

    buy_signal_column = config.get("names")[0]
    sell_signal_column = config.get("names")[1]
    
    df[buy_signal_column] = (df[columns] >= parameters.get("buy_signal_threshold"))
    df[sell_signal_column] = (df[columns] <= parameters.get("sell_signal_threshold"))

    return df, [buy_signal_column, sell_signal_column]


import pandas as pd
from datetime import datetime

def simulated_trade_performance(df, buy_signal_column, sell_signal_column, price_column,
                                initial_capital=10000, fee_percent=0.1, max_concurrent_trades=5,
                                stop_loss=0.03, take_profit=0.03, time_limit=32, log_file='trade_log.txt'):
    capital = initial_capital
    position_size = capital / max_concurrent_trades
    trade_fee = fee_percent / 100
    open_trades = []
    open_short_trades = []
    peak_capital = initial_capital
    max_drawdown = 0

    long_profit = short_profit = 0
    long_transactions = short_transactions = 0
    long_profitable = short_profitable = 0
    longs, shorts = [], []

    with open(log_file, 'w') as log:
        df = df[[sell_signal_column, buy_signal_column, price_column]]

        for (index, sell_signal, buy_signal, price) in df.itertuples(name=None):
            if pd.isnull(price) or price <= 0:
                continue

            trade_date = str(index)

            open_trades = [pos for pos in open_trades if index - pos[0] <= time_limit]
            open_short_trades = [pos for pos in open_short_trades if index - pos[0] <= time_limit]

            for position in open_trades.copy():
                entry_index, entry_price, _ = position
                profit = (price - entry_price) * position_size / entry_price
                profit_percent = (price - entry_price) / entry_price
                exit_fee = profit * trade_fee
                net_profit = profit - exit_fee

                if profit_percent >= take_profit or profit_percent <= -stop_loss or index - entry_index >= time_limit:
                    reason = "Take Profit Hit" if profit_percent >= take_profit else (
                        "Stop Loss Hit" if profit_percent <= -stop_loss else "Time Expiry"
                    )
                    capital_adjustment = position_size + net_profit
                    if capital_adjustment < 0:
                        capital_adjustment = 0
                    capital += capital_adjustment
                    open_trades.remove(position)
                    long_profit += net_profit
                    long_transactions += 1
                    if net_profit > 0:
                        long_profitable += 1
                    log.write(f"{trade_date} - Sell: Entered at {entry_price}, Exited at {price} | Profit: {net_profit} | Reason: {reason} | Active Positions: {len(open_trades)}\n")

            for position in open_short_trades.copy():
                entry_index, entry_price, _ = position
                profit = (entry_price - price) * position_size / entry_price
                profit_percent = (entry_price - price) / entry_price
                exit_fee = profit * trade_fee
                net_profit = profit - exit_fee

                if profit_percent >= take_profit or profit_percent <= -stop_loss or index - entry_index >= time_limit:
                    reason = "Take Profit Hit" if profit_percent >= take_profit else (
                        "Stop Loss Hit" if profit_percent <= -stop_loss else "Time Expiry"
                    )
                    capital_adjustment = position_size + net_profit
                    if capital_adjustment < 0:
                        capital_adjustment = 0
                    capital += capital_adjustment
                    open_short_trades.remove(position)
                    short_profit += net_profit
                    short_transactions += 1
                    if net_profit > 0:
                        short_profitable += 1
                    log.write(f"{trade_date} - Close Short: Entered at {entry_price}, Exited at {price} | Profit: {net_profit} | Reason: {reason} | Active Positions: {len(open_short_trades)}\n")

            peak_capital = max(peak_capital, capital)
            drawdown = ((peak_capital - capital) / peak_capital) * 100
            max_drawdown = max(max_drawdown, drawdown)

            if buy_signal and len(open_trades) < max_concurrent_trades and not any(abs(pos[0] - index) <= 2 for pos in open_trades):
                entry_fee = position_size * trade_fee
                if capital >= (position_size + entry_fee):
                    capital -= (position_size + entry_fee)
                    open_trades.append((index, price, capital))
                    log.write(f"{trade_date} - Buy: {price} | Capital: {capital} | Active Positions: {len(open_trades)}\n")

            if sell_signal and len(open_short_trades) < max_concurrent_trades and not any(abs(pos[0] - index) <= 2 for pos in open_short_trades):
                entry_fee = position_size * trade_fee
                if capital >= (position_size + entry_fee):
                    capital -= (position_size + entry_fee)
                    open_short_trades.append((index, price, capital))
                    log.write(f"{trade_date} - Open Short: {price} | Capital: {capital} | Active Positions: {len(open_short_trades)}\n")

    long_performance = {
        "profit": long_profit,
        "transaction_no": long_transactions,
        "profit_percent": long_profit / initial_capital * 100,
        "profitable": long_profitable / long_transactions if long_transactions else 0.0,
        "transactions": longs
    }

    short_performance = {
        "profit": short_profit,
        "transaction_no": short_transactions,
        "profit_percent": short_profit / initial_capital * 100,
        "profitable": short_profitable / short_transactions if short_transactions else 0.0,
        "transactions": shorts
    }

    total_trades = long_transactions + short_transactions
    performance = {
        "final_capital": capital,
        "profit": long_profit + short_profit,
        "profit_percent": (capital - initial_capital) / initial_capital * 100,
        "transaction_no": total_trades,
        "profit_per_transaction": (long_profit + short_profit) / total_trades if total_trades else 0.0,
        "profitable_percent": 100.0 * ((long_profitable + short_profitable) / total_trades) if total_trades else 0.0,
        "max_drawdown": max_drawdown,
    }

    return performance, long_performance, short_performance





if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    df = pd.read_csv('./data/BTCUSDT/signals.csv', parse_dates=['timestamp'], nrows=100_000_000)
    
    performance, long_performance, short_performance = simulated_trade_performance(df, 'sell_signal_column', 'buy_signal_column', 'close')

    pass