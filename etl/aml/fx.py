"""Daily USD exchange rates shared by the ETL scripts and the training data loaders.

Every amount is converted with the daily close of its currency against USD on the day of the transaction. All callers
read the same cached table (data/fx_daily_usd.csv by default), so pretraining and downstream graphs use identical rates
and compute nodes without internet access only need the cached file.
"""

import logging
import os

import pandas as pd

USD = "US Dollar"

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
    "Ether": "ETH-USD",  # Ethereum phishing dataset
}

DEFAULT_RATES_PATH = "data/fx_daily_usd.csv"
LOOKBACK_DAYS = 7  # Fetched before the first day so it can be forward-filled when it falls on a weekend or holiday


def _fetch_daily_closes(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily close of every ticker between start and end (inclusive), one column per currency"""
    import yfinance as yf

    closes = dict()
    for name, ticker in CURRENCY_TICKERS.items():
        close = yf.Ticker(ticker).history(start=start, end=end + pd.Timedelta(days=1))["Close"]
        if close.empty:
            raise RuntimeError(
                f"yfinance returned no prices for {ticker} ({name}) between {start.date()} and {end.date()}. "
                f"Provide a cached rates file instead."
            )
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        closes[name] = close.groupby(level=0).last()

    return pd.DataFrame(closes)


def _covers(table: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    has_currencies = all(name in table.columns for name in list(CURRENCY_TICKERS) + [USD])
    return has_currencies and table.index.min() <= start and table.index.max() >= end


def daily_usd_rates(start, end, path: str = DEFAULT_RATES_PATH) -> pd.DataFrame:
    """USD value of one unit of every currency, for every calendar day between start and end.

    Days without a close (weekends, holidays, gaps in the history) take the last available close. The table is cached
    in path; a cached table that does not cover the requested period is refetched over the union of both periods.

    Returns:
        DataFrame indexed by day (column "Date") with one column per currency, including US Dollar = 1
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()

    if os.path.exists(path):
        cached = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        if _covers(cached, start, end):
            return cached.loc[start:end]
        start, end = min(start, cached.index.min()), max(end, cached.index.max())

    first = start - pd.Timedelta(days=LOOKBACK_DAYS)
    closes = _fetch_daily_closes(first, end)
    table = closes.reindex(pd.date_range(first, end, freq="D"))

    gaps = table.loc[start:end].isna().sum()
    if gaps.any():
        logging.info(f"Forward-filled days without a close per currency: {gaps[gaps > 0].to_dict()}")
    leading = table.ffill().loc[start:end].isna().sum()
    if leading.any():
        logging.warning(f"No earlier close to forward-fill, used the first later close instead: {leading.to_dict()}")

    table = table.ffill().bfill().loc[start:end]
    table[USD] = 1.0
    table.index.name = "Date"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table.to_csv(path)

    return table
