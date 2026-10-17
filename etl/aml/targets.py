"""Pretraining targets of the Fin-PSE encoder.

Every function takes the transactions of one graph as a polars DataFrame with integer "From" and "To" account ids, a
Datetime "Timestamp" and the USD "Amount", and returns one row per account ("Node") of that graph. Targets that are
undefined for an account (e.g. the dwell time of an account that never sends money on) are null; the pretraining loss
skips them.

The targets are grouped so each group can be switched on and off in the pretraining config (ssl.target_groups):
- ego: what an account sends and receives
- temporal: when it does so, how long money stays and how bursty it is
- flow: where its outgoing money ends up and where its incoming money comes from
- motifs: laundering structures it takes part in (reciprocity, gather-scatter, scatter-gather, cycles)
"""

import os

import numpy as np
import polars as pl

EPS = 1e-12
DAY = 86400  # Seconds

TARGET_GROUPS = {
    "ego": [
        "deg_in",
        "deg_out",
        "fan_in",
        "fan_out",
        "vol_in",
        "vol_out",
        "flow_imbalance",
        "max_amt_in",
        "max_amt_out",
        "amt_cv_in",
        "amt_cv_out",
    ],
    "temporal": [
        "mu_gap_in_sec",
        "var_gap_in_sec",
        "mu_gap_out_sec",
        "var_gap_out_sec",
        "dwell_median_sec",
        "dwell_mean_sec",
        "pass_through_24h",
        "peak_day_deg_in",
        "peak_day_deg_out",
        "peak_day_fan_in",
        "peak_day_fan_out",
        "peak_day_vol_in",
        "peak_day_vol_out",
        "burstiness_in",
        "burstiness_out",
        "active_span_days",
    ],
    "flow": [
        "H_out_1_amt",
        "supp_out_1_amt",
        "pmax_out_1_amt",
        "H_out_2_amt",
        "supp_out_2_amt",
        "pmax_out_2_amt",
        "H_in_1_amt",
        "supp_in_1_amt",
        "pmax_in_1_amt",
        "H_in_2_amt",
        "supp_in_2_amt",
        "pmax_in_2_amt",
        "return_2",
        "return_3",
        "return_4",
        "sink_share_1",
        "sink_share_2",
        "onward_share_24h",
    ],
    "motifs": [
        "r_2cycle",
        "gather_scatter_daily",
        "scatter_gather_in",
        "scatter_gather_out",
        "temporal_cycles",
        "simple_cycles",
    ],
}

SINK_KEEP_RATIO = 0.1  # Sinks send on at most this share of what they receive


def nodes_of(trans: pl.DataFrame) -> pl.DataFrame:
    """Every account of the graph"""
    return pl.concat([trans.select(pl.col("From").alias("Node")), trans.select(pl.col("To").alias("Node"))]).unique()


def _seconds(trans: pl.DataFrame) -> np.ndarray:
    """Timestamps as integer seconds since the first transaction"""
    return (trans["Timestamp"] - trans["Timestamp"].min()).dt.total_seconds().to_numpy().astype(np.int64)


# ---------------------------------------------------------------------------------------------------------------------
# Ego profile


def ego_profile(trans: pl.DataFrame) -> pl.DataFrame:
    """Degrees, number of distinct counterparties, volumes, flow imbalance and the shape of the amounts per direction"""

    def side(node, other, suffix):
        return (
            trans.group_by(node)
            .agg(
                pl.len().alias(f"deg_{suffix}"),
                pl.col(other).n_unique().alias(f"fan_{suffix}"),
                pl.col("Amount").sum().alias(f"vol_{suffix}"),
                pl.col("Amount").max().alias(f"max_amt_{suffix}"),
                (pl.col("Amount").std(ddof=0) / (pl.col("Amount").mean() + EPS)).alias(f"amt_cv_{suffix}"),
            )
            .rename({node: "Node"})
        )

    ego = (
        nodes_of(trans)
        .join(side("From", "To", "out"), on="Node", how="left")
        .join(side("To", "From", "in"), on="Node", how="left")
    )
    counts = [f"{name}_{suffix}" for name in ("deg", "fan") for suffix in ("in", "out")]
    return ego.with_columns(
        [pl.col(name).fill_null(0).cast(pl.Int64) for name in counts]
        + [pl.col(name).fill_null(0.0) for name in ("vol_in", "vol_out")]
    ).with_columns(
        ((pl.col("vol_in") - pl.col("vol_out")) / (pl.col("vol_in") + pl.col("vol_out") + EPS)).alias("flow_imbalance")
    )


# ---------------------------------------------------------------------------------------------------------------------
# Temporal behaviour


def _gaps(trans: pl.DataFrame, node: str, suffix: str) -> pl.DataFrame:
    """Mean and variance of the time between consecutive transactions of an account in one direction"""
    return (
        trans.sort([node, "Timestamp"])
        .group_by(node)
        .agg(pl.col("Timestamp").diff().dt.total_seconds().drop_nulls().alias("gap"))
        .with_columns(
            pl.col("gap").list.mean().alias(f"mu_gap_{suffix}_sec"),
            pl.col("gap").list.var().alias(f"var_gap_{suffix}_sec"),
        )
        .select(pl.col(node).alias("Node"), f"mu_gap_{suffix}_sec", f"var_gap_{suffix}_sec")
    )


def _dwell(trans: pl.DataFrame, delta: int) -> pl.DataFrame:
    """Time from every incoming transaction to the next outgoing transaction of the same account.

    pass_through_24h is the share of incoming transactions followed by an outgoing one within delta seconds.
    """
    moves = trans.filter(pl.col("From") != pl.col("To"))
    incoming = moves.select(pl.col("To").alias("Node"), pl.col("Timestamp").alias("t_in")).sort("t_in")
    outgoing = moves.select(pl.col("From").alias("Node"), pl.col("Timestamp").alias("t_out")).sort("t_out")

    return (
        incoming.join_asof(outgoing, left_on="t_in", right_on="t_out", by="Node", strategy="forward")
        .with_columns((pl.col("t_out") - pl.col("t_in")).dt.total_seconds().alias("lag"))
        .group_by("Node")
        .agg(
            pl.col("lag").median().alias("dwell_median_sec"),
            pl.col("lag").mean().alias("dwell_mean_sec"),
            (pl.col("lag") <= delta).fill_null(False).mean().alias("pass_through_24h"),
        )
    )


def _peak_days(trans: pl.DataFrame, node: str, other: str, suffix: str) -> pl.DataFrame:
    """Activity on the busiest calendar day of an account and how much busier that day is than an average active day"""
    return (
        trans.with_columns(pl.col("Timestamp").dt.truncate("1d").alias("day"))
        .group_by([node, "day"])
        .agg(pl.len().alias("deg"), pl.col(other).n_unique().alias("fan"), pl.col("Amount").sum().alias("vol"))
        .group_by(node)
        .agg(
            pl.col("deg").max().alias(f"peak_day_deg_{suffix}"),
            pl.col("fan").max().alias(f"peak_day_fan_{suffix}"),
            pl.col("vol").max().alias(f"peak_day_vol_{suffix}"),
            (pl.col("deg").max() / pl.col("deg").mean()).alias(f"burstiness_{suffix}"),
        )
        .rename({node: "Node"})
    )


def temporal_profile(trans: pl.DataFrame, delta: int = DAY) -> pl.DataFrame:
    """Gaps between transactions, dwell time, pass-through, busiest-day activity, burstiness and active span"""
    endpoints = pl.concat(
        [
            trans.select(pl.col("From").alias("Node"), "Timestamp"),
            trans.select(pl.col("To").alias("Node"), "Timestamp"),
        ]
    )
    span = endpoints.group_by("Node").agg(
        ((pl.col("Timestamp").max() - pl.col("Timestamp").min()).dt.total_seconds() / DAY).alias("active_span_days")
    )

    counts = [f"peak_day_{name}_{suffix}" for name in ("deg", "fan") for suffix in ("in", "out")]
    return (
        nodes_of(trans)
        .join(_gaps(trans, "To", "in"), on="Node", how="left")
        .join(_gaps(trans, "From", "out"), on="Node", how="left")
        .join(_dwell(trans, delta), on="Node", how="left")
        .join(_peak_days(trans, "To", "From", "in"), on="Node", how="left")
        .join(_peak_days(trans, "From", "To", "out"), on="Node", how="left")
        .join(span, on="Node", how="left")
        .with_columns(
            [pl.col(name).fill_null(0).cast(pl.Int64) for name in counts]
            + [pl.col(f"peak_day_vol_{suffix}").fill_null(0.0) for suffix in ("in", "out")]
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# Money flow


def _transition(trans: pl.DataFrame, src: str, dst: str) -> pl.DataFrame:
    """Share of the money of every src account that goes to each dst account (amount-weighted transition matrix)"""
    return (
        trans.group_by([src, dst])
        .agg(pl.col("Amount").sum().alias("w"))
        .with_columns((pl.col("w") / (pl.col("w").sum().over(src) + EPS)).alias("p"))
        .select(pl.col(src).alias("src"), pl.col(dst).alias("dst"), "p")
    )


def _distribution_summaries(P: pl.DataFrame, suffix: str) -> pl.DataFrame:
    """Entropy, support and largest share of the 1- and 2-step distributions of the rows of P"""

    def summarise(dist, k):
        return dist.group_by("src").agg(
            (-pl.col("p") * (pl.col("p") + EPS).log()).sum().alias(f"H_{suffix}_{k}_amt"),
            pl.len().alias(f"supp_{suffix}_{k}_amt"),
            pl.col("p").max().alias(f"pmax_{suffix}_{k}_amt"),
        )

    two_step = (
        P.join(P.rename({"src": "dst", "dst": "end", "p": "p2"}), on="dst", how="inner")
        .group_by(["src", "end"])
        .agg((pl.col("p") * pl.col("p2")).sum().alias("p"))
    )
    return summarise(P, 1).join(summarise(two_step, 2), on="src", how="left").rename({"src": "Node"})


def _csr(P: pl.DataFrame, num_nodes: int):
    """Row-sorted transition matrix with the cumulative probabilities of every row offset by the row index"""
    P = P.sort(["src", "dst"])
    src, dst, p = P["src"].to_numpy(), P["dst"].to_numpy(), P["p"].to_numpy()
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    cum = np.cumsum(p) - np.repeat(np.concatenate([[0.0], np.cumsum(p)])[indptr[:-1]], np.diff(indptr))
    cum[indptr[1:][np.diff(indptr) > 0] - 1] = 1.0  # Every row sums to exactly 1
    return indptr, dst, src + cum


def _return_mass(P: pl.DataFrame, num_nodes: int, steps=(3, 4), walks: int = 64, seed: int = 0, chunk: int = 50_000):
    """Monte-Carlo estimate of the share of an account's money that is back at the account after k steps.

    Walks follow the amount-weighted transition matrix and stop at accounts without outgoing money.
    """
    indptr, dst, keys = _csr(P, num_nodes)
    starts = np.flatnonzero(np.diff(indptr) > 0)
    rng = np.random.default_rng(seed)
    returned = {k: np.zeros(num_nodes) for k in steps}

    for begin in range(0, len(starts), chunk):
        origin = np.repeat(starts[begin : begin + chunk], walks)
        position, alive = origin.copy(), np.ones(len(origin), dtype=bool)
        for step in range(1, max(steps) + 1):
            alive &= np.diff(indptr)[position] > 0
            query = position[alive] + rng.random(alive.sum())
            position[alive] = dst[np.minimum(np.searchsorted(keys, query, side="right"), len(dst) - 1)]
            if step in steps:
                np.add.at(returned[step], origin, (alive & (position == origin)).astype(float))

    nodes = pl.DataFrame({"Node": starts})
    for k in steps:
        nodes = nodes.with_columns(pl.Series(f"return_{k}", returned[k][starts] / walks))
    return nodes


def _onward_share(trans: pl.DataFrame, delta: int) -> pl.DataFrame:
    """Share of an account's outgoing money that its recipients send on within delta seconds of receiving it"""
    moves = trans.filter(pl.col("From") != pl.col("To"))
    src, dst = moves["From"].to_numpy().astype(np.int64), moves["To"].to_numpy().astype(np.int64)
    amount, t = moves["Amount"].to_numpy().astype(np.float64), _seconds(moves)

    # Outgoing transactions sorted by (account, time) with cumulative amounts, to sum any account's outflow in a window
    scale = int(t.max()) + delta + 1
    key = src * scale + t
    order = np.argsort(key, kind="stable")
    sorted_key, cumulative = key[order], np.concatenate([[0.0], np.cumsum(amount[order])])

    lower = np.searchsorted(sorted_key, dst * scale + t, side="left")
    upper = np.searchsorted(sorted_key, dst * scale + t + delta, side="right")
    onward = np.minimum(amount, cumulative[upper] - cumulative[lower])

    return (
        pl.DataFrame({"Node": src, "amount": amount, "onward": onward})
        .group_by("Node")
        .agg((pl.col("onward").sum() / (pl.col("amount").sum() + EPS)).alias("onward_share_24h"))
    )


def flow_profile(trans: pl.DataFrame, delta: int = DAY, walks: int = 64, seed: int = 0) -> pl.DataFrame:
    """Where an account's outgoing money ends up and where its incoming money comes from.

    With P[i, j] the share of i's outgoing money sent to j:
    - H/supp/pmax_out_k: entropy, number of destinations and largest destination share of P^k[i, .], k = 1, 2
    - H/supp/pmax_in_k: the same for the share of i's incoming money coming from each source (k steps back)
    - return_k: P^k[i, i], the share of i's money back at i after k steps (k = 2 exact, k = 3, 4 Monte-Carlo)
    - sink_share_k: share of i's money ending, within k steps, at sinks (accounts that keep at least 90% of what
      they receive)
    - onward_share_24h: share of i's outgoing money that the recipients send on within delta seconds (time-respecting)
    """
    nodes = nodes_of(trans)
    num_nodes = int(nodes["Node"].max()) + 1

    P = _transition(trans, "From", "To")
    out_summaries = _distribution_summaries(P, "out")
    in_summaries = _distribution_summaries(_transition(trans, "To", "From"), "in")

    return_2 = (
        P.join(P.rename({"src": "dst", "dst": "src", "p": "p_back"}), on=["src", "dst"], how="inner")
        .group_by("src")
        .agg((pl.col("p") * pl.col("p_back")).sum().alias("return_2"))
        .rename({"src": "Node"})
    )

    volumes = ego_profile(trans).select("Node", "vol_in", "vol_out")
    sinks = volumes.select(
        pl.col("Node").alias("dst"),
        ((pl.col("vol_in") > 0) & (pl.col("vol_out") <= SINK_KEEP_RATIO * pl.col("vol_in")))
        .cast(pl.Float64)
        .alias("sink"),
    )
    hop = P.join(sinks, on="dst", how="left").with_columns(pl.col("sink").fill_null(0.0))
    sink_1 = hop.group_by("src").agg((pl.col("p") * pl.col("sink")).sum().alias("sink_share_1"))
    sink_2 = (
        hop.join(sink_1.rename({"src": "dst", "sink_share_1": "next"}), on="dst", how="left")
        .with_columns(pl.col("next").fill_null(0.0))
        .group_by("src")
        .agg((pl.col("p") * (pl.col("sink") + (1 - pl.col("sink")) * pl.col("next"))).sum().alias("sink_share_2"))
    )

    return (
        nodes.join(out_summaries, on="Node", how="left")
        .join(in_summaries, on="Node", how="left")
        .join(return_2, on="Node", how="left")
        .join(_return_mass(P, num_nodes, walks=walks, seed=seed), on="Node", how="left")
        .join(sink_1.rename({"src": "Node"}), on="Node", how="left")
        .join(sink_2.rename({"src": "Node"}), on="Node", how="left")
        .join(_onward_share(trans, delta), on="Node", how="left")
        .join(volumes, on="Node", how="left")
        # Accounts that send money but never get any back have a return mass of 0, not an undefined one
        .with_columns(pl.when(pl.col("vol_out") > 0).then(pl.col("return_2").fill_null(0.0)).alias("return_2"))
        .drop("vol_in", "vol_out")
    )


# ---------------------------------------------------------------------------------------------------------------------
# Laundering motifs


def _reciprocity(trans: pl.DataFrame) -> pl.DataFrame:
    """Number of 2-cycles: transactions to a counterparty matched by transactions back (as in the first ETL)"""
    edges = trans.filter(pl.col("From") != pl.col("To")).group_by(["From", "To"]).agg(pl.len().alias("w"))
    return (
        edges.join(edges, left_on=["From", "To"], right_on=["To", "From"], how="inner", suffix="_rev")
        .with_columns(pl.min_horizontal("w", "w_rev").alias("cycle_count"))
        .group_by("From")
        .agg(pl.col("cycle_count").sum().alias("r_2cycle"))
        .rename({"From": "Node"})
    )


def _gather_scatter(trans: pl.DataFrame) -> pl.DataFrame:
    """Largest day on which an account both collects from and pays out to several counterparties: max over days of
    min(distinct senders, distinct recipients)"""
    day = pl.col("Timestamp").dt.truncate("1d").alias("day")
    fan_in = trans.group_by([pl.col("To").alias("Node"), day]).agg(pl.col("From").n_unique().alias("fan_in"))
    fan_out = trans.group_by([pl.col("From").alias("Node"), day]).agg(pl.col("To").n_unique().alias("fan_out"))
    return (
        fan_in.join(fan_out, on=["Node", "day"], how="inner")
        .group_by("Node")
        .agg(pl.min_horizontal("fan_in", "fan_out").max().alias("gather_scatter_daily"))
    )


GFP_PATTERNS = ("scatter-gather", "temp-cycle", "lc-cycle")


def gfp_edge_patterns(trans: pl.DataFrame, num_threads: int = None) -> dict:
    """Number of scatter-gather, temporal cycle and simple cycle patterns every transaction takes part in, from IBM's
    Graph Feature Preprocessor (snapml) with its default time windows"""
    from snapml import GraphFeaturePreprocessor

    t = _seconds(trans)
    order = np.argsort(t, kind="stable")
    edges = np.column_stack(
        [
            np.arange(len(t), dtype=np.float64),
            trans["From"].to_numpy().astype(np.float64),
            trans["To"].to_numpy().astype(np.float64),
            t.astype(np.float64),
        ]
    )[order]

    gfp = GraphFeaturePreprocessor()
    params = gfp.get_params()
    params.update({"num_threads": num_threads or os.cpu_count(), "vertex_stats": False, "fan": False, "degree": False})
    params.update({pattern: True for pattern in GFP_PATTERNS})
    gfp.set_params(params)
    features = gfp.transform(edges)

    counts, column = dict(), edges.shape[1]
    for pattern in GFP_PATTERNS:
        bins = len(params[f"{pattern}_bins"])
        per_edge = np.zeros(len(t))
        per_edge[features[:, 0].astype(np.int64)] = features[:, column : column + bins].sum(axis=1)
        counts[pattern] = per_edge
        column += bins
    return counts


def motif_profile(trans: pl.DataFrame, use_gfp: bool = True) -> pl.DataFrame:
    """Reciprocity, gather-scatter and, with GFP, scatter-gather and cycle participation.

    scatter_gather_in/out count the patterns an account's incoming/outgoing transactions take part in (a source only
    has outgoing ones, a sink only incoming ones, an intermediary both); temporal_cycles and simple_cycles count the
    cycles its outgoing transactions take part in.
    """
    profile = (
        nodes_of(trans)
        .join(_reciprocity(trans), on="Node", how="left")
        .join(_gather_scatter(trans), on="Node", how="left")
        .with_columns(pl.col("r_2cycle").fill_null(0), pl.col("gather_scatter_daily").fill_null(0))
    )
    if not use_gfp:
        return profile

    counts = gfp_edge_patterns(trans)
    edges = trans.select("From", "To").with_columns(
        pl.Series("sg", counts["scatter-gather"]),
        pl.Series("tc", counts["temp-cycle"]),
        pl.Series("lc", counts["lc-cycle"]),
    )
    outgoing = edges.group_by(pl.col("From").alias("Node")).agg(
        pl.col("sg").sum().alias("scatter_gather_out"),
        pl.col("tc").sum().alias("temporal_cycles"),
        pl.col("lc").sum().alias("simple_cycles"),
    )
    incoming = edges.group_by(pl.col("To").alias("Node")).agg(pl.col("sg").sum().alias("scatter_gather_in"))
    gfp = ["scatter_gather_in", "scatter_gather_out", "temporal_cycles", "simple_cycles"]
    return (
        profile.join(outgoing, on="Node", how="left")
        .join(incoming, on="Node", how="left")
        .with_columns([pl.col(name).fill_null(0.0) for name in gfp])
    )


# ---------------------------------------------------------------------------------------------------------------------


def compute_targets(
    trans: pl.DataFrame, groups=tuple(TARGET_GROUPS), delta: int = DAY, use_gfp: bool = True, seed: int = 0
) -> pl.DataFrame:
    """All targets of the selected groups for every account of the graph, sorted by account"""
    profiles = {
        "ego": lambda: ego_profile(trans),
        "temporal": lambda: temporal_profile(trans, delta),
        "flow": lambda: flow_profile(trans, delta, seed=seed),
        "motifs": lambda: motif_profile(trans, use_gfp),
    }

    targets = nodes_of(trans)
    for group in groups:
        columns = TARGET_GROUPS[group]
        if group == "motifs" and not use_gfp:
            columns = ["r_2cycle", "gather_scatter_daily"]
        targets = targets.join(profiles[group]().select(["Node"] + columns), on="Node", how="left")
    return targets.sort("Node")
