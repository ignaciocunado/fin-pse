import polars as pl
import datetime
from polars import DataFrame

from utils import *

dataset = "data/raw/HI-Medium_Trans.csv"

trans = (
    pl.read_csv(dataset)
    .with_columns(pl.col("Timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M", strict=True))
    .sort("Timestamp")
)


zero = trans["Timestamp"].min()
hundred = trans["Timestamp"].max()
diff = hundred - zero
days = diff.days
sixty = zero + datetime.timedelta(days=days * 0.6)
eighty = zero + datetime.timedelta(days=days * 0.8)
hundred = zero + datetime.timedelta(days=days)

ssl = remove_strings(trans, currencies=True)
every = "2d"


def _prep(df: pl.DataFrame) -> pl.LazyFrame:
    return (
        df.lazy()
        .with_columns(pl.col("Timestamp").dt.truncate(every).alias("window_start"))
        .filter(pl.col("Timestamp").is_not_null() & pl.col("From").is_not_null() & pl.col("To").is_not_null())
    )


_prep(ssl).sort(pl.col("window_start")).collect()


def add_temporal_stats(df: pl.DataFrame) -> pl.DataFrame:
    df = _prep(df).select(
        pl.col("window_start"),
        pl.col("Timestamp"),
        pl.col("From"),
        pl.col("To"),
    )

    in_gaps = (
        df.sort(["window_start", "To", "Timestamp"])
        .group_by(["window_start", "To"])
        .agg(pl.col("Timestamp").diff().dt.total_seconds().alias("gap"))
        .with_columns(pl.col("gap").list.drop_nulls().alias("gap"))
        .with_columns(
            [
                pl.col("gap").list.mean().alias("mu_gap_in_sec"),
                pl.col("gap").list.var().alias("var_gap_in_sec"),
            ]
        )
        .select(["window_start", "To", "mu_gap_in_sec", "var_gap_in_sec"])
        .rename({"To": "Node"})
    )

    out_gaps = (
        df.sort(["window_start", "From", "Timestamp"])
        .group_by(["window_start", "From"])
        .agg(pl.col("Timestamp").diff().dt.total_seconds().alias("gap"))
        .with_columns(pl.col("gap").list.drop_nulls().alias("gap"))
        .with_columns(
            [
                pl.col("gap").list.mean().alias("mu_gap_out_sec"),
                pl.col("gap").list.var().alias("var_gap_out_sec"),
            ]
        )
        .select(["window_start", "From", "mu_gap_out_sec", "var_gap_out_sec"])
        .rename({"From": "Node"})
    )

    return in_gaps.join(out_gaps, on=["Node", "window_start"], how="full", coalesce=True).fill_null(-1).collect()


temporal_features = add_temporal_stats(ssl)


def count_reciprocal_neighbours(df: DataFrame) -> DataFrame:
    lf = _prep(df)

    edges = (
        lf.select(["window_start", "From", "To"])
        .filter(pl.col("From") != pl.col("To"))
        .group_by(["window_start", "From", "To"])
        .agg(pl.len().alias("w"))
    )

    recip = (
        edges.join(
            edges,
            left_on=["window_start", "From", "To"],
            right_on=["window_start", "To", "From"],
            how="inner",
            suffix="_rev",
        )
        .with_columns(pl.min_horizontal("w", "w_rev").alias("cycle_count"))
        .group_by(["window_start", "From"])
        .agg(pl.col("cycle_count").sum().alias("r_2cycle"))
        .rename({"From": "Node"})
    )

    nodes = pl.concat(
        [
            lf.select(["window_start", pl.col("From").alias("Node")]),
            lf.select(["window_start", pl.col("To").alias("Node")]),
        ]
    ).unique()

    return (
        nodes.join(recip, on=["window_start", "Node"], how="left")
        .with_columns(pl.col("r_2cycle").fill_null(0).cast(pl.Int64))
        .collect()
    )


two_cycles = count_reciprocal_neighbours(ssl)
EPS = 1e-12


def compute_ego_profiles(trans: DataFrame) -> DataFrame:
    lf = _prep(trans)

    out_stats = (
        lf.group_by(["window_start", "From"])
        .agg(
            pl.len().alias("deg_out"),
            pl.col("To").n_unique().alias("fan_out"),
            pl.col("Amount Paid").sum().alias("vol_out"),
        )
        .rename({"From": "Node"})
    )

    in_stats = (
        lf.group_by(["window_start", "To"])
        .agg(
            pl.len().alias("deg_in"),
            pl.col("From").n_unique().alias("fan_in"),
            pl.col("Amount Received").sum().alias("vol_in"),
        )
        .rename({"To": "Node"})
    )

    ego = (
        out_stats.join(in_stats, on=["window_start", "Node"], how="full")
        .with_columns(
            pl.col("deg_out").fill_null(0).cast(pl.Int64),
            pl.col("fan_out").fill_null(0).cast(pl.Int64),
            pl.col("vol_out").fill_null(0.0),
            pl.col("deg_in").fill_null(0).cast(pl.Int64),
            pl.col("fan_in").fill_null(0).cast(pl.Int64),
            pl.col("vol_in").fill_null(0.0),
        )
        .with_columns(
            ((pl.col("vol_in") - pl.col("vol_out")) / (pl.col("vol_in") + pl.col("vol_out") + pl.lit(EPS))).alias(
                "flow_imbalance"
            )
        )
        .with_columns(
            pl.when(pl.col("window_start").is_null())
            .then(pl.col("window_start_right"))
            .otherwise(pl.col("window_start"))
            .alias("window_start"),
            pl.when(pl.col("Node").is_null()).then(pl.col("Node_right")).otherwise(pl.col("Node")).alias("Node"),
        )
        .drop("Node_right", "window_start_right")
    )

    per_cur_in = (
        lf.group_by(["window_start", "To", "Receiving Currency"])
        .agg(pl.col("Amount Received").sum().alias("vol_in_cur"))
        .rename({"To": "Node", "Receiving Currency": "currency"})
    )

    mix_in = (
        per_cur_in.join(
            per_cur_in.group_by(["window_start", "Node"]).agg(pl.col("vol_in_cur").sum().alias("vol_in_sum")),
            on=["window_start", "Node"],
        )
        .with_columns((pl.col("vol_in_cur") / (pl.col("vol_in_sum") + pl.lit(EPS))).alias("p"))
        .group_by(["window_start", "Node"])
        .agg(
            [
                pl.len().alias("n_currencies_in"),
                (-pl.col("p") * (pl.col("p") + pl.lit(EPS)).log()).sum().alias("currency_entropy_in"),
                pl.col("p").max().alias("top_currency_share_in"),
            ]
        )
    )

    per_cur_out = (
        lf.group_by(["window_start", "From", "Payment Currency"])
        .agg(pl.col("Amount Paid").sum().alias("vol_out_cur"))
        .rename({"From": "Node", "Payment Currency": "currency"})
    )

    mix_out = (
        per_cur_out.join(
            per_cur_out.group_by(["window_start", "Node"]).agg(pl.col("vol_out_cur").sum().alias("vol_out_sum")),
            on=["window_start", "Node"],
        )
        .with_columns((pl.col("vol_out_cur") / (pl.col("vol_out_sum") + pl.lit(EPS))).alias("p"))
        .group_by(["window_start", "Node"])
        .agg(
            [
                pl.len().alias("n_currencies_out"),
                (-pl.col("p") * (pl.col("p") + pl.lit(EPS)).log()).sum().alias("currency_entropy_out"),
                pl.col("p").max().alias("top_currency_share_out"),
            ]
        )
    )

    full = (
        ego.join(mix_in, on=["window_start", "Node"], how="left")
        .join(mix_out, on=["window_start", "Node"], how="left")
        .with_columns(
            pl.col("n_currencies_in").fill_null(0).cast(pl.Int64),
            pl.col("currency_entropy_in").fill_null(0.0),
            pl.col("top_currency_share_in").fill_null(0.0),
            pl.col("n_currencies_out").fill_null(0).cast(pl.Int64),
            pl.col("currency_entropy_out").fill_null(0.0),
            pl.col("top_currency_share_out").fill_null(0.0),
        )
    )

    return full.collect()


ego_profile = compute_ego_profiles(ssl)
EPS = 1e-12


def flow_targets_out_entropy_count(trans: DataFrame, k2: bool = True) -> DataFrame:
    lf = _prep(trans).select(["window_start", "From", "To"])

    # Edge weight = count of transactions (currency invariant)
    W = lf.group_by(["window_start", "From", "To"]).agg(pl.len().alias("w"))

    out_sum = W.group_by(["window_start", "From"]).agg(pl.col("w").sum().alias("out_wsum"))

    P = (
        W.join(out_sum, on=["window_start", "From"], how="left")
        .with_columns((pl.col("w") / (pl.col("out_wsum") + pl.lit(EPS))).alias("p"))
        .select(["window_start", "From", "To", "p"])
    )

    one = (
        P.group_by(["window_start", "From"])
        .agg(
            (-pl.col("p") * (pl.col("p") + pl.lit(EPS)).log()).sum().alias("H_out_1_cnt"),
            pl.len().alias("supp_out_1_cnt"),
            pl.col("p").max().alias("pmax_out_1_cnt"),
        )
        .rename({"From": "Node"})
    )

    if not k2:
        return one.collect()

    # 2-step distribution via mid join: sum_mid P(src->mid)*P(mid->dst)
    P1 = P.rename({"From": "src", "To": "mid", "p": "p1"})
    P2 = P.rename({"From": "mid", "To": "dst", "p": "p2"})

    two_step = (
        P1.join(P2, on=["window_start", "mid"], how="inner")
        .with_columns((pl.col("p1") * pl.col("p2")).alias("p2step"))
        .group_by(["window_start", "src", "dst"])
        .agg(pl.col("p2step").sum().alias("p"))
    )

    two = (
        two_step.group_by(["window_start", "src"])
        .agg(
            (-pl.col("p") * (pl.col("p") + pl.lit(EPS)).log()).sum().alias("H_out_2_cnt"),
            pl.len().alias("supp_out_2_cnt"),
            pl.col("p").max().alias("pmax_out_2_cnt"),
        )
        .rename({"src": "Node"})
    )

    return one.join(two, on=["window_start", "Node"], how="left").collect()


def currency_mix_out(trans: DataFrame) -> DataFrame:
    lf = _prep(trans).select(["window_start", "From", "Payment Currency", "Amount Paid"])

    per_cur = (
        lf.group_by(["window_start", "From", "Payment Currency"])
        .agg(pl.col("Amount Paid").sum().alias("vol_out_cur"))
        .rename({"From": "Node", "Payment Currency": "currency"})
    )

    mix = (
        per_cur.join(
            per_cur.group_by(["window_start", "Node"]).agg(pl.col("vol_out_cur").sum().alias("vol_out_sum")),
            on=["window_start", "Node"],
        )
        .with_columns((pl.col("vol_out_cur") / (pl.col("vol_out_sum") + pl.lit(EPS))).alias("p"))
        .group_by(["window_start", "Node"])
        .agg(
            [
                pl.count().alias("n_currencies_out"),
                (-pl.col("p") * (pl.col("p") + pl.lit(EPS)).log()).sum().alias("currency_entropy_out"),
                pl.col("p").max().alias("top_currency_share_out"),
            ]
        )
    )

    return mix.collect()


def flow_heads_A_B(trans: DataFrame, k2: bool = True) -> DataFrame:
    head_a = flow_targets_out_entropy_count(trans, k2=k2)  # (window_start, node, ...)
    head_b = currency_mix_out(trans)  # (window_start, node, ...)

    # Node universe: all active nodes in the window (senders or receivers)
    lf = _prep(trans).select(["window_start", "From", "To"])
    nodes = (
        pl.concat(
            [
                lf.select(["window_start", pl.col("From").alias("Node")]),
                lf.select(["window_start", pl.col("To").alias("Node")]),
            ]
        )
        .unique()
        .collect()
    )

    full = (
        nodes.lazy()
        .join(head_a.lazy(), on=["window_start", "Node"], how="left")
        .join(head_b.lazy(), on=["window_start", "Node"], how="left")
        .with_columns(
            # Fill missing: nodes with no outgoing edges => no flow / no currency mix
            pl.col("H_out_1_cnt").fill_null(0.0),
            pl.col("supp_out_1_cnt").fill_null(0).cast(pl.Int64),
            pl.col("pmax_out_1_cnt").fill_null(0.0),
            pl.col("H_out_2_cnt").fill_null(0.0),
            pl.col("supp_out_2_cnt").fill_null(0).cast(pl.Int64),
            pl.col("pmax_out_2_cnt").fill_null(0.0),
            pl.col("n_currencies_out").fill_null(0).cast(pl.Int64),
            pl.col("currency_entropy_out").fill_null(0.0),
            pl.col("top_currency_share_out").fill_null(0.0),
        )
        .collect()
    )

    return full


train_pos_pred = flow_heads_A_B(ssl)
node_features = (
    temporal_features.join(train_pos_pred, on=["window_start", "Node"], how="full", coalesce=True)
    .join(ego_profile, on=["window_start", "Node"], how="full", coalesce=True)
    .join(two_cycles, on=["window_start", "Node"], how="full", coalesce=True)
    .with_columns(
        (pl.col("window_start") - pl.col("window_start").min())
        .dt.total_seconds()
        .cast(pl.Int64)
        .add(10)
        .alias("window_start"),
        pl.lit(1).alias("Feature"),
    )
    .sort("window_start")
)
trans = (
    _prep(ssl)
    .with_columns(
        (pl.col("Timestamp") - pl.col("Timestamp").min()).dt.total_seconds().cast(pl.Int64).add(10).alias("Timestamp"),
    )
    .with_columns(
        (pl.col("window_start") - pl.col("window_start").min())
        .dt.total_seconds()
        .cast(pl.Int64)
        .add(10)
        .alias("window_start"),
        pl.lit(1).alias("Feature"),
    )
    .sort("window_start")
    .with_row_index("Edge ID")
    .collect()
)


node_features.write_csv("data/HI-Medium_SSL_Nodes_2d.csv")
trans.write_csv("data/HI-Medium_SSL_Trans_2d.csv")
