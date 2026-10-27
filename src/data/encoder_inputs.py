import torch
import torch.nn.functional as F

from src.data.etl import load_etl_module
from src.util import symlog

DAY = 86400


def currency_vocabulary() -> list:
    """Fixed currency vocabulary of the encoder (the AML currencies and Ether, etl/aml/fx.py)"""
    fx = load_etl_module("fx")
    return list(fx.CURRENCY_TICKERS) + [fx.USD]


def encoder_edge_features(
    seconds, amount_received, receiving, amount_paid, paying, usd, native: bool, time_in_days: bool, log_amounts: bool
):
    """Encoder edge features of transactions, before ports and normalisation, identical in pretraining and downstream.

    USD mode: [time, USD amount]. Native mode (no conversion): [time, amount received, one-hot receiving currency,
    amount paid, one-hot payment currency], with currencies given as indices into currency_vocabulary().

    Returns:
        Float64 tensor of features and their names
    """
    seconds = torch.as_tensor(seconds, dtype=torch.float64)
    time = (seconds / DAY if time_in_days else seconds).view(-1, 1)

    def amount(values):
        values = torch.as_tensor(values, dtype=torch.float64).view(-1, 1)
        return symlog(values) if log_amounts else values

    if not native:
        return torch.cat([time, amount(usd)], dim=1), ["Timestamp", "Amount"]

    vocabulary = currency_vocabulary()

    def one_hot(indices):
        return F.one_hot(torch.as_tensor(indices, dtype=torch.long), len(vocabulary)).double()

    features = torch.cat(
        [time, amount(amount_received), one_hot(receiving), amount(amount_paid), one_hot(paying)], dim=1
    )
    names = (
        ["Timestamp", "Amount Received"]
        + [f"Receiving {name}" for name in vocabulary]
        + ["Amount Paid"]
        + [f"Payment {name}" for name in vocabulary]
    )
    return features, names


# Columns of data.enc_edge_raw, the raw encoder inputs stored with the downstream graphs (independent of the mode)
RAW_SECONDS, RAW_RECEIVED, RAW_RECEIVING, RAW_PAID, RAW_PAYING, RAW_DAY = range(6)


def raw_edge_inputs(seconds, received, receiving, paid, paying, days) -> torch.Tensor:
    """Raw encoder inputs: seconds since the start of the graph, amount received, receiving currency (vocabulary
    index), amount paid, payment currency (vocabulary index) and day (days since 1970-01-01)"""
    columns = [seconds, received, receiving, paid, paying, days]
    return torch.stack([torch.as_tensor(column, dtype=torch.float64) for column in columns], dim=1)


def usd_received(edge_raw: torch.Tensor) -> torch.Tensor:
    """Amount received in USD at the daily rate of the receiving currency (etl/aml/fx.py)"""
    from src.data.currency import usd_rates

    days = edge_raw[:, RAW_DAY].long().numpy()
    if days.min() < 14245 or days.max() > 23741:  # 2009 - 2035
        raise ValueError("Transaction days are implausible: ETH timestamps must be Unix seconds to convert to USD")
    vocabulary = {name: k for k, name in enumerate(currency_vocabulary())}
    rates = usd_rates(days, edge_raw[:, RAW_RECEIVING].long().numpy(), vocabulary)
    return edge_raw[:, RAW_RECEIVED] * torch.as_tensor(rates, dtype=torch.float64)


def features_from_raw(edge_raw: torch.Tensor, native: bool, time_in_days: bool, log_amounts: bool):
    """Encoder edge features (before ports and normalisation) from the raw inputs"""
    edge_raw = edge_raw.double()
    return encoder_edge_features(
        edge_raw[:, RAW_SECONDS],
        edge_raw[:, RAW_RECEIVED],
        edge_raw[:, RAW_RECEIVING].long(),
        edge_raw[:, RAW_PAID],
        edge_raw[:, RAW_PAYING].long(),
        None if native else usd_received(edge_raw),
        native=native,
        time_in_days=time_in_days,
        log_amounts=log_amounts,
    )


def transactions_frame(edge_index: torch.Tensor, edge_raw: torch.Tensor, native: bool):
    """Transactions of a graph as etl/aml/targets.py expects them (USD amounts, or amounts with currency names)"""
    import polars as pl

    edge_raw = edge_raw.double()
    columns = {
        "From": edge_index[0].numpy(),
        "To": edge_index[1].numpy(),
        "Timestamp": pl.from_epoch(pl.Series(edge_raw[:, RAW_SECONDS].round().numpy().astype("int64")), "s"),
    }
    if native:
        vocabulary = currency_vocabulary()
        columns.update(
            {
                "Amount Received": edge_raw[:, RAW_RECEIVED].numpy(),
                "Receiving Currency": [vocabulary[k] for k in edge_raw[:, RAW_RECEIVING].long().tolist()],
                "Amount Paid": edge_raw[:, RAW_PAID].numpy(),
                "Payment Currency": [vocabulary[k] for k in edge_raw[:, RAW_PAYING].long().tolist()],
            }
        )
    else:
        columns["Amount"] = usd_received(edge_raw).numpy()
    return pl.DataFrame(columns)
