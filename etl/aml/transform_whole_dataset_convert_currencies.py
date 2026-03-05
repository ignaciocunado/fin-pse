import polars as pl
import datetime
from polars import DataFrame

from utils import *


dataset = 'data/raw/HI-Medium_Trans.csv'

trans = (
    pl.read_csv(dataset)
    .with_columns(
        pl.col("Timestamp").str.strptime(pl.Datetime, '%Y/%m/%d %H:%M', strict=True)
    )
    .sort('Timestamp')
)


zero = trans['Timestamp'].min()
hundred = trans['Timestamp'].max()
diff = hundred - zero
days = diff.days
sixty = zero + datetime.timedelta(days=days * 0.6)
eighty = zero + datetime.timedelta(days=days * 0.8)
hundred = zero + datetime.timedelta(days=days)

converted = convert_currencies_to_usd(trans)
assert converted.shape[0] == trans.shape[0]
trans = converted

ssl = remove_strings(trans, currencies=False)
assert ssl.shape[0] == trans.shape[0]

def _prep(df: pl.DataFrame) -> pl.LazyFrame:
    return (
        df.lazy()
        .filter(
             pl.col('Timestamp').is_not_null()
             & pl.col('From').is_not_null()
             & pl.col('To').is_not_null()
        )
    )

def add_temporal_stats(df: pl.DataFrame) -> pl.DataFrame:
    df = _prep(df).select(
        pl.col('Timestamp'),
        pl.col('From'),
        pl.col('To'),
    )

    in_gaps = (
        df.sort(["Timestamp", "To"])
        .group_by(["To"])
        .agg(
            pl.col('Timestamp').diff().dt.total_seconds().alias('gap')
        )
        .with_columns(
            pl.col("gap").list.drop_nulls().alias("gap")
        )
        .with_columns([
            pl.col("gap").list.mean().alias("mu_gap_in_sec"),
            pl.col("gap").list.var().alias("var_gap_in_sec"),
        ])
        .select(['To', "mu_gap_in_sec", "var_gap_in_sec"])
        .rename({'To': 'Node'})
    )

    out_gaps = (
        df.sort(["Timestamp", "From" ])
        .group_by("From")
        .agg(
            pl.col('Timestamp').diff().dt.total_seconds().alias('gap')
        )
        .with_columns(
            pl.col("gap").list.drop_nulls().alias("gap")
        )
        .with_columns([
            pl.col("gap").list.mean().alias("mu_gap_out_sec"),
            pl.col("gap").list.var().alias("var_gap_out_sec"),
        ])
        .select(['From', "mu_gap_out_sec", "var_gap_out_sec"])
        .rename({'From': 'Node'})
    )

    return (
        in_gaps
        .join(out_gaps, on="Node", how="full", coalesce=True)
        .fill_null(-1)
        .collect()
    )

temporal_features = add_temporal_stats(ssl)
assert temporal_features.shape[0] == 2076999

def count_reciprocal_neighbours(df: DataFrame) -> DataFrame:
    lf = _prep(df)

    edges = (
        lf.select(["From", "To"])
        .filter(pl.col("From") != pl.col("To"))
        .group_by(["From", "To"])
        .agg(pl.len().alias("w"))
    )

    recip = (
        edges.join(
            edges,
            left_on=["From", "To"],
            right_on=["To", "From"],
            how="inner",
            suffix="_rev",
        )
        .with_columns(
            pl.min_horizontal("w", "w_rev").alias("cycle_count")
        )
        .group_by("From")
        .agg(
            pl.col("cycle_count").sum().alias("r_2cycle")
        )
        .rename({"From": "Node"})
    )

    nodes = (
        pl.concat([
            lf.select(pl.col("From").alias("Node")),
            lf.select(pl.col("To").alias("Node")),
        ])
        .unique()
    )

    return (
        nodes.join(recip, on="Node", how="left")
        .with_columns(
            pl.col("r_2cycle").fill_null(0).cast(pl.Int64)
        )
        .collect()
    )


two_cycles = count_reciprocal_neighbours(ssl)
assert two_cycles.shape[0] == 2076999

EPS = 1e-12

def compute_ego_profiles(trans: DataFrame) -> DataFrame:
    lf = _prep(trans)

    out_stats = (
        lf.group_by("From")
        .agg(
            pl.len().alias("deg_out"),
            pl.col("To").n_unique().alias("fan_out"),
            pl.col("Amount").sum().alias("vol_out"),
        )
        .rename({"From": "Node"})
    )

    in_stats = (
        lf.group_by("To")
        .agg(
            pl.len().alias("deg_in"),
            pl.col("From").n_unique().alias("fan_in"),
            pl.col("Amount").sum().alias("vol_in"),
        )
        .rename({"To": "Node"})
    )

    ego = (
        out_stats.join(in_stats, on="Node", how="full")
        .with_columns(
            pl.col("deg_out").fill_null(0).cast(pl.Int64),
            pl.col("fan_out").fill_null(0).cast(pl.Int64),
            pl.col("vol_out").fill_null(0.0),
            pl.col("deg_in").fill_null(0).cast(pl.Int64),
            pl.col("fan_in").fill_null(0).cast(pl.Int64),
            pl.col("vol_in").fill_null(0.0),
        )
        .with_columns(
            ((pl.col("vol_in") - pl.col("vol_out"))/ (pl.col("vol_in") + pl.col("vol_out") + pl.lit(EPS))).alias("flow_imbalance")
        )
        .with_columns(
            pl.when(pl.col('Node').is_null()).then(pl.col('Node_right')).otherwise(pl.col('Node')).alias('Node'),
        )
        .drop('Node_right')
    )

    return ego.collect()

ego_profile = compute_ego_profiles(ssl)
assert ego_profile.shape[0] == 2076999


EPS = 1e-12

def flow_targets_out_entropy_amount(trans: DataFrame, k2: bool = True) -> DataFrame:
    lf = _prep(trans).select(["From", "To", "Amount"]).lazy()

    W = (
        lf.group_by(["From", "To"])
        .agg(pl.col("Amount").sum().alias("w"))
    )

    P = (
        W.with_columns(
            pl.col("w").sum().over("From").alias("out_wsum")
        )
        .with_columns(
            (pl.col("w") / (pl.col("out_wsum") + pl.lit(EPS))).alias("p")
        )
        .select(["From", "To", "p"])
        .cache()
    )

    one = (
        P.group_by(["From"])
        .agg(
            (-pl.col("p") * (pl.col("p") + pl.lit(EPS)).log()).sum().alias("H_out_1_amt"),
            pl.len().alias("supp_out_1_amt"),
            pl.col("p").max().alias("pmax_out_1_amt"),
        )
        .rename({"From": "Node"})
    )

    if not k2:
        return one.collect()

    P1 = P.rename({"From": "src", "To": "mid", "p": "p1"})
    P2 = P.rename({"From": "mid", "To": "dst", "p": "p2"})

    two_step = (
        P1.join(P2, on="mid", how="inner")
        .with_columns((pl.col("p1") * pl.col("p2")).alias("p"))
        .group_by(["src", "dst"])
        .agg(pl.col("p").sum().alias("p"))
    )

    two = (
        two_step.group_by("src")
        .agg(
            (-pl.col("p") * (pl.col("p") + pl.lit(EPS)).log()).sum().alias("H_out_2_amt"),
            pl.len().alias("supp_out_2_amt"),
            pl.col("p").max().alias("pmax_out_2_amt"),
        )
        .rename({"src": "Node"})
    )

    return one.join(two, on="Node", how="left").collect()

train_pos_pred = flow_targets_out_entropy_amount(ssl)

node_features = (
    temporal_features
    .join(train_pos_pred, on="Node", how="full", coalesce=True)
    .join(ego_profile, on="Node", how="full", coalesce=True)
    .join(two_cycles, on="Node", how="full", coalesce=True)
    .with_columns(
        pl.lit(1).alias("Feature"),
    )
)

trans = (
    ssl.with_columns(
        (pl.col("Timestamp") - pl.col("Timestamp").min())
        .dt.total_seconds()
        .cast(pl.Int64)
        .add(10)
        .alias("Timestamp"),
    )
    .sort("Timestamp")
    .with_row_index("Edge ID")
)

node_features.write_csv('data/HI-Medium_SSL_Nodes_whole_convert.csv')
trans.write_csv('data/HI-Medium_SSL_Trans_whole_convert.csv')

