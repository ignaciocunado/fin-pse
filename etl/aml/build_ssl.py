"""Builds the Fin-PSE pretraining graphs: the whole transaction graph plus time crops, each with its own targets.

The crops are prefixes of the period (the same shape as the downstream train and validation graphs) and a few short
random intervals. Targets are always computed on the transactions of the crop itself, so the encoder learns to compute
statistics of whatever graph it is shown.

Run from the repository root, e.g.:
    python etl/aml/build_ssl.py --input data/raw/HI-Medium_Trans.csv --name HI-Medium

Outputs <out-dir>/<name>_SSL_Nodes_finpse.csv (one row per account and graph, with the targets) and
<out-dir>/<name>_SSL_Trans_finpse.csv (one row per transaction and graph). Both have a graph_id column; transaction
timestamps are seconds since the start of their graph and amounts are in USD (daily rates, see fx.py).
"""

import argparse
import datetime
import json
import logging

import numpy as np
import polars as pl

from fx import DEFAULT_RATES_PATH
from targets import DAY, TARGET_GROUPS, compute_targets
from utils import convert_currencies_to_usd, implied_rates, remove_strings

RAW_COLUMNS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


def read_transactions(path: str) -> pl.DataFrame:
    """Raw IBM AML transactions with one From and one To account column (bank + account, like AMLData)"""
    with open(path) as f:
        header = f.readline().strip().split(",")

    if "From" in header and "To" in header:
        trans = pl.read_csv(path)
    else:
        accounts = ["From Bank", "From Account", "To Bank", "To Account"]
        trans = pl.read_csv(
            path,
            has_header=False,
            skip_rows=1,
            new_columns=RAW_COLUMNS,
            schema_overrides={name: pl.Utf8 for name in accounts},
        )
        trans = trans.with_columns(
            (pl.col("From Bank") + pl.col("From Account")).alias("From"),
            (pl.col("To Bank") + pl.col("To Account")).alias("To"),
        ).drop(accounts)

    return trans.with_columns(pl.col("Timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M", strict=True)).sort(
        "Timestamp"
    )


def crops(trans: pl.DataFrame, prefixes, n_short: int, short_days, seed: int):
    """The whole graph, prefixes covering the given fractions of the period and n_short random short intervals"""
    start, end = trans["Timestamp"].min(), trans["Timestamp"].max()
    span = (end - start).total_seconds()

    yield "whole", trans
    for fraction in prefixes:
        yield f"prefix_{round(fraction * 100)}", trans.filter(
            pl.col("Timestamp") <= start + datetime.timedelta(seconds=fraction * span)
        )

    rng = np.random.default_rng(seed)
    for k in range(n_short):
        length = min(rng.uniform(*short_days) * DAY, span)
        begin = start + datetime.timedelta(seconds=rng.uniform(0, span - length))
        yield f"short_{k}", trans.filter(
            (pl.col("Timestamp") >= begin) & (pl.col("Timestamp") < begin + datetime.timedelta(seconds=length))
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Raw IBM AML transactions CSV")
    parser.add_argument("--name", required=True, help="Prefix of the output files, e.g. HI-Medium")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--rates", default=DEFAULT_RATES_PATH, help="Cached daily USD rates table")
    parser.add_argument("--groups", nargs="+", default=list(TARGET_GROUPS), choices=list(TARGET_GROUPS))
    parser.add_argument("--delta", type=int, default=DAY, help="Window of the time-local targets in seconds")
    parser.add_argument("--prefixes", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--n-short", type=int, default=5, help="Number of short random intervals")
    parser.add_argument("--short-days", type=float, nargs=2, default=[2, 7])
    parser.add_argument("--no-crops", action="store_true", help="Only the whole graph (e.g. targets of a test graph)")
    parser.add_argument("--no-gfp", action="store_true", help="Skip the GFP motif targets (no snapml)")
    parser.add_argument(
        "--gfp-params",
        type=json.loads,
        default=None,
        help='JSON overrides of GFP parameters, e.g. \'{"temp-cycle_tw": 172800, "lc-cycle": true}\'',
    )
    parser.add_argument("--implied-rates", action="store_true", help="Compare simulator and yfinance exchange rates")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s] %(message)s")

    trans = read_transactions(args.input)
    logging.info(f"{trans.shape[0]} transactions from {trans['Timestamp'].min()} to {trans['Timestamp'].max()}")

    if args.implied_rates:
        with pl.Config(tbl_rows=-1):
            print(implied_rates(trans, args.rates))

    trans = remove_strings(convert_currencies_to_usd(trans, rates_path=args.rates))

    if args.no_crops:
        graphs = [("whole", trans)]
    else:
        graphs = crops(trans, args.prefixes, args.n_short, args.short_days, args.seed)

    nodes, edges = [], []
    for graph_id, (name, graph) in enumerate(graphs):
        logging.info(f"Graph {graph_id} ({name}): {graph.shape[0]} transactions")
        targets = compute_targets(
            graph, args.groups, args.delta, not args.no_gfp, args.gfp_params, seed=args.seed + graph_id
        )
        nodes.append(targets.with_columns(pl.lit(graph_id).alias("graph_id"), pl.lit(name).alias("graph_name")))
        edges.append(
            graph.with_columns(
                (pl.col("Timestamp") - pl.col("Timestamp").min()).dt.total_seconds().cast(pl.Int64).alias("Timestamp"),
                pl.lit(graph_id).alias("graph_id"),
            ).with_row_index("Edge ID")
        )

    pl.concat(nodes, how="diagonal").write_csv(f"{args.out_dir}/{args.name}_SSL_Nodes_finpse.csv")
    pl.concat(edges).write_csv(f"{args.out_dir}/{args.name}_SSL_Trans_finpse.csv")
    logging.info(f"Wrote {args.out_dir}/{args.name}_SSL_{{Nodes,Trans}}_finpse.csv")


if __name__ == "__main__":
    main()
