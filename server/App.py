from pathlib import Path
import json
from datetime import datetime
import re
import pandas as pd

PACKAGE_ROOT = Path(__file__).parent.parent

class App:
    # Global used variable

    # System
    loop = None # Asyncio main loop
    sched = None # Scheduler

    analyzer = None # Store and analyze data

    client = None # Binance client

    bm = None # bm and conn_key for online trading
    conn_key = None

    # State of the server
    error_status = 0 # Networks, connections, exceptions ...
    server_status = 0 # Server allowing us to trade or not
    account_status = 0 # Account allowing us to trade or not (maybe not enough money in wallet?)
    trade_state_status = 0  # Something wrong with our trading logic

    df = None

    transaction = None
    status = None # BOUGHT, SOLD, BUYING, SELLING
    order = None # Latest order
    order_time = None

    base_quantity = "0.0363636" # BTC owned
    quote_quantity = "1000.0" # USDT owned
    
    # Status data retrieved from the server
    system_status = {"status": 0, "msg": "Everything's fined :D"}
    symbol_info = {}
    account_info = {}

    # Configuration parameters
    config = {
        # Binance
        "api_key": "",
        "api_secret": "",

        # Telegram
        "telegram_bot_token": "",
        "telegram_chat_id" : "",

        # Name file convention, all the processed file names are here
        "merge_file_name": "klines.csv",
        "feature_file_name": "features.csv",
        "label_file_name": "label.csv",
        "predict_file_name": "predict.csv",
        "signal_file_name": "signal.csv",
        "signal_models_file_name": "signal_model",

        "model_folder": "models",

        "time_column": "timestamp",

        # Location for all generated data/models
        "data_folder": "", 

        # Data downloading
        "symbol": "BTCUSDT",
        "freq": "1min",
        "data_source": [],

        # Feature generation
        "feature_sets": [],

        "label_horizon": 0, # Number of tail rows to be excluded from model training
        "train_length": 0, # Training rows

        # List of all features used for training/predicting
        "train_labels": [],
        
        # List of all labels
        "labels": [],

        # List of algorithms
        "algorithms": [],

        # Minimum length of rows to compute derived features
        "features_horizon": 10,

        # List of signals
        "signal_sets": [],

        # Notification
        "score_notification_model": {},
        "diagram_notification_model": {}, #TODO

        # Trading
        "trade_model" : {
            "no_trades": False, # If true, all below hyperparameters are excluded
            "test_order_before_submit": False,
            "percentage_used_for_trade": 99, # In % of USDT quantity.
            "limit_price_adjustment": 0.005 # Limit price of orders will be better than the latest close price (0 means no change, positive - better for us, negative - worse for us)
        },

        "train_signal_model": {},

        # Binance trader
        "base_asset": "", # Coin symbol: BTC
        "quote_asset": "",

        "collector": {
            "folder": "data",
            "flush_period:": 300, #Seconds
            "stream": {
                "folder": "stream",
                "channels": ["kline_1m", "depth20"],
                "symbols": ["BTCUSDT"],
            }
        }
    }

def data_provider_problems_exist():
    if App.error_status != 0 or App.server_status != 0:
        return True
    else:
        return False

def problems_exist():
    if data_provider_problems_exist() or App.account_status or App.trade_state_status:
        return True
    else:
        return False

def load_config(config_file):
    if config_file:
        config_file_path = PACKAGE_ROOT / config_file
        with open(config_file_path, encoding='utf-8') as json_file:
            conf_str = json_file.read()

            # Remove comment
            conf_str = re.sub(r"//.*$", "", conf_str, flags=re.M)

            conf_json = json.loads(conf_str)
            App.config.update(conf_json)
    