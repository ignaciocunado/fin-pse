import json
import logging
import os.path as osp
from datetime import datetime

import numpy as np
import pandas as pd

# Same tickers as etl/aml/utils.py::convert_currencies_to_usd, which converts the pretraining graphs
CURRENCY_TICKERS = {
    "Yen": "JPYUSD=X",
    "Bitcoin": "BTC-USD",
    "Mexican Peso": "MXNUSD=X",
    "UK Pound": "GBPUSD=X",
    "Rupee": "INRUSD=X",
    "Yuan": "CNYUSD=X",
    "Euro": "EURUSD=X",
    "Australian Dollar": "AUDUSD=X",
    "Swiss Franc": "CHFUSD=X",
    "Shekel": "ILSUSD=X",
    "Canadian Dollar": "CADUSD=X",
    "Saudi Riyal": "SARUSD=X",
    "Ruble": "RUBUSD=X",
    "Brazil Real": "BRLUSD=X",
}

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def read_currency_meta(raw_csv: str, meta_path: str, chunksize: int = 5_000_000) -> dict:
    """Currency ids used in the formatted transactions and the period of the raw transactions.

    AMLData.format_dataset numbers the currencies in order of first appearance, reading the receiving currency and then
    the payment currency of every row, so the same order is rebuilt here. The result is cached in meta_path.
    """
    if osp.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)

    logging.info(f"Reading currencies and period of {raw_csv}")
    currencies = dict()
    start, end = None, None
    for chunk in pd.read_csv(
        raw_csv, usecols=["Timestamp", "Receiving Currency", "Payment Currency"], dtype=str, chunksize=chunksize
    ):
        interleaved = np.column_stack(
            [chunk["Receiving Currency"].to_numpy(), chunk["Payment Currency"].to_numpy()]
        ).ravel()
        for name in pd.unique(interleaved):
            if name not in currencies:
                currencies[name] = len(currencies)

        timestamps = pd.to_datetime(chunk["Timestamp"], format=TIMESTAMP_FORMAT)
        start = timestamps.min() if start is None else min(start, timestamps.min())
        end = timestamps.max() if end is None else max(end, timestamps.max())

    meta = {"currencies": currencies, "start": start.isoformat(), "end": end.isoformat()}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def fetch_usd_rates(start: datetime, end: datetime) -> dict:
    """Average daily close of every currency in USD over the period, as done for the pretraining graphs"""
    import yfinance as yf

    rates = {"US Dollar": 1.0}
    for name, ticker in CURRENCY_TICKERS.items():
        close = yf.Ticker(ticker).history(start=start, end=end)[["Close"]].values
        if len(close) == 0:
            raise RuntimeError(
                f"yfinance returned no prices for {ticker} ({name}) between {start} and {end}. "
                f"Set dataset.currency_rates_file to a JSON file with the USD rate of every currency instead."
            )
        rates[name] = float(close.mean())

    return rates


def get_usd_rates(raw_csv: str, processed_dir: str, table: str, rates_file: str = "") -> dict:
    """USD rate of every currency id in the formatted transactions.

    The rates are read from rates_file ({currency name: rate}) if given. Otherwise they are the yfinance averages over the
    period of the dataset, cached in processed_dir so yfinance is only queried once.
    """
    meta = read_currency_meta(raw_csv, osp.join(processed_dir, f"currency_meta_{table}.json"))

    if rates_file:
        with open(rates_file) as f:
            rates = json.load(f)
    else:
        rates_path = osp.join(processed_dir, f"currency_rates_{table}.json")
        if osp.exists(rates_path):
            with open(rates_path) as f:
                rates = json.load(f)
        else:
            rates = fetch_usd_rates(datetime.fromisoformat(meta["start"]), datetime.fromisoformat(meta["end"]))
            with open(rates_path, "w") as f:
                json.dump(rates, f, indent=2)

    missing = [name for name in meta["currencies"] if name not in rates]
    if missing:
        raise ValueError(f"No USD rate for currencies {missing}")

    logging.info(f"USD rates between {meta['start']} and {meta['end']}: {rates}")
    return {currency_id: rates[name] for name, currency_id in meta["currencies"].items()}
