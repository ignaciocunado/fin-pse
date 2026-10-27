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
