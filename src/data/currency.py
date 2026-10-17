import importlib.util
import json
import logging
import os.path as osp

import numpy as np
import pandas as pd
from torch_geometric.graphgym import cfg

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def _load_fx():
    """The daily rates module of the ETL, loaded by path so the ETL and the loaders share one implementation"""
    path = osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "etl", "aml", "fx.py")
    spec = importlib.util.spec_from_file_location("fin_pse_fx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fx = _load_fx()


def fx_rates_path() -> str:
    """Daily rates table shared with the ETL (data/fx_daily_usd.csv unless dataset.fx_rates_file is set)"""
    return cfg.dataset.fx_rates_file or osp.join(cfg.root_dir, fx.DEFAULT_RATES_PATH)


def daily_rates(first_day: int, last_day: int, currency_names: list) -> np.ndarray:
    """USD rate of every currency for every day between first_day and last_day (days since 1970-01-01).

    Returns:
        Array of shape [last_day - first_day + 1, len(currency_names)]
    """
    rates = fx.daily_usd_rates(pd.Timestamp(first_day, unit="D"), pd.Timestamp(last_day, unit="D"), fx_rates_path())
    missing = [name for name in currency_names if name not in rates.columns]
    if missing:
        raise ValueError(f"No USD rate for currencies {missing}")
    return rates[currency_names].to_numpy(dtype=np.float64)


def read_raw_meta(raw_csv: str, processed_dir: str, table: str, chunksize: int = 5_000_000):
    """Currency ids used in the formatted transactions and the day of every raw transaction.

    AMLData.format_dataset numbers the currencies in order of first appearance, reading the receiving currency and then
    the payment currency of every row, so the same order is rebuilt here. The day of every transaction (days since
    1970-01-01) is stored by raw row, which is the EdgeID of the formatted transactions. Both are cached in processed_dir.

    Returns:
        Currency name -> id dictionary and an int32 array with the day of every transaction
    """
    meta_path = osp.join(processed_dir, f"currency_meta_{table}.json")
    days_path = osp.join(processed_dir, f"transaction_days_{table}.npy")
    if osp.exists(meta_path) and osp.exists(days_path):
        with open(meta_path) as f:
            return json.load(f)["currencies"], np.load(days_path)

    logging.info(f"Reading currencies and days of {raw_csv}")
    currencies, days = dict(), []
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
        days.append(timestamps.to_numpy().astype("datetime64[D]").astype(np.int32))

    days = np.concatenate(days)
    with open(meta_path, "w") as f:
        json.dump({"currencies": currencies}, f, indent=2)
    np.save(days_path, days)

    return currencies, days


def usd_rates(days: np.ndarray, currency_ids: np.ndarray, currencies: dict) -> np.ndarray:
    """Daily USD rate of every transaction, given its day (days since 1970-01-01) and currency id"""
    names = [name for name, _ in sorted(currencies.items(), key=lambda item: item[1])]
    first_day, last_day = int(days.min()), int(days.max())
    rates = daily_rates(first_day, last_day, names)
    return rates[days - first_day, currency_ids]


def aml_usd_rates(df_edges: pd.DataFrame, currency_column: str, raw_csv: str, processed_dir: str, table: str):
    """Daily USD rate of every formatted AML transaction for the currency in currency_column"""
    currencies, days = read_raw_meta(raw_csv, processed_dir, table)
    edge_days = days[df_edges["EdgeID"].to_numpy()]
    return usd_rates(edge_days, df_edges[currency_column].to_numpy(), currencies)
