import polars as pl
import yfinance as yf

from fx import CURRENCY_TICKERS, DEFAULT_RATES_PATH, daily_usd_rates


def _daily_rates_long(transactions, rates_path):
    """Daily USD rates as rows (Date, Currency, Rate) covering the period of the transactions"""
    rates = daily_usd_rates(transactions["Timestamp"].min(), transactions["Timestamp"].max(), rates_path)
    return pl.DataFrame(
        {
            "Date": rates.index.values.astype("datetime64[D]").repeat(len(rates.columns)),
            "Currency": list(rates.columns) * len(rates),
            "Rate": rates.to_numpy(dtype=float).ravel(),
        }
    )


def convert_currencies_to_usd(transactions, daily=True, rates_path=DEFAULT_RATES_PATH):
    """Converts the received amounts to USD.

    With daily=True every amount is multiplied by the daily close of its currency on the day of the transaction (see
    fx.py). With daily=False the close is averaged over the whole period of the transactions, as in the first version.
    """
    if daily:
        rates = _daily_rates_long(transactions, rates_path).rename({"Currency": "Receiving Currency"})
        converted = (
            transactions.with_row_index("_row")
            .with_columns(pl.col("Timestamp").dt.date().alias("Date"))
            .join(rates, on=["Date", "Receiving Currency"], how="left")
            .sort("_row")
        )
        missing = converted.filter(pl.col("Rate").is_null())["Receiving Currency"].unique().to_list()
        if missing:
            raise ValueError(f"No USD rate for currencies {missing}")

        return (
            converted.with_columns((pl.col("Amount Received") * pl.col("Rate")).alias("Amount"))
            .drop("_row", "Date", "Rate")
            .drop("Amount Paid", "Amount Received", "Payment Currency", "Receiving Currency")
        )

    zero = transactions["Timestamp"].min()
    hundred = transactions["Timestamp"].max()

    currencies_to_avg_price = {
        k: yf.Ticker(v).history(start=zero, end=hundred)[["Close"]].values.mean() for k, v in CURRENCY_TICKERS.items()
    } | {"US Dollar": 1.0}

    return transactions.with_columns(
        (
            pl.col("Amount Received") * pl.col("Receiving Currency").replace(currencies_to_avg_price).cast(pl.Float64)
        ).alias("Amount")
    ).drop("Amount Paid", "Amount Received", "Payment Currency", "Receiving Currency")


def implied_rates(transactions, rates_path=DEFAULT_RATES_PATH):
    """Compares the exchange rates implied by the simulator with the daily yfinance rates.

    On cross-currency transactions Amount Paid (payment currency) and Amount Received (receiving currency) are the same
    value, so their USD values should agree if the simulator used real exchange rates. Returns, per currency pair, the
    number of transactions, the spread of the implied rate (0 when the simulator uses a constant rate) and the median
    log ratio between the USD values of both sides (0 when the simulator agrees with yfinance).
    """
    rates = _daily_rates_long(transactions, rates_path)
    pay_rates = rates.rename({"Currency": "Payment Currency", "Rate": "pay_rate"})
    receive_rates = rates.rename({"Currency": "Receiving Currency", "Rate": "receive_rate"})

    return (
        transactions.filter(
            (pl.col("Payment Currency") != pl.col("Receiving Currency"))
            & (pl.col("Amount Paid") > 0)
            & (pl.col("Amount Received") > 0)
        )
        .with_columns(pl.col("Timestamp").dt.date().alias("Date"))
        .join(pay_rates, on=["Date", "Payment Currency"], how="left")
        .join(receive_rates, on=["Date", "Receiving Currency"], how="left")
        .with_columns(
            (pl.col("Amount Received") / pl.col("Amount Paid")).log().alias("log_implied_rate"),
            ((pl.col("Amount Received") * pl.col("receive_rate")) / (pl.col("Amount Paid") * pl.col("pay_rate")))
            .log()
            .alias("log_usd_ratio"),
        )
        .group_by(["Payment Currency", "Receiving Currency"])
        .agg(
            pl.len().alias("n"),
            pl.col("log_implied_rate").std().alias("implied_rate_log_std"),
            pl.col("log_usd_ratio").median().alias("median_log_usd_ratio"),
        )
        .sort("n", descending=True)
    )


def remove_strings(transactions, currencies=False):
    nodes = (
        transactions.select(pl.col("From").alias("Acc"))
        .vstack(transactions.select(pl.col("To").alias("Acc")))
        .unique()
        .with_row_index("Node ID")
    )

    payment_formats = transactions.select(pl.col("Payment Format")).unique().with_row_index("format_id")

    final = (
        transactions.join(nodes, left_on="From", right_on="Acc")
        .join(nodes, left_on="To", right_on="Acc", suffix="_to")
        .join(payment_formats, on="Payment Format")
        .drop(
            "From",
            "To",
            "Payment Format",
        )
        .rename(
            {
                "Node ID": "From",
                "Node ID_to": "To",
                "format_id": "Payment Format",
            }
        )
    )

    if currencies:
        curr = (
            transactions.select(pl.col("Receiving Currency").alias("Currency"))
            .vstack(transactions.select(pl.col("Payment Currency").alias("Currency")))
            .unique()
            .with_row_index("currency_id")
        )

        final = (
            final.join(curr, left_on="Payment Currency", right_on="Currency")
            .join(curr, left_on="Receiving Currency", right_on="Currency", suffix="_to")
            .drop("Payment Currency", "Receiving Currency")
            .rename(
                {
                    "currency_id": "Payment Currency",
                    "currency_id_to": "Receiving Currency",
                }
            )
        )

    return final
