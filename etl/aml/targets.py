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

Without currency conversion (native=True) the transactions keep "Amount Received" in "Receiving Currency" and "Amount
Paid" in "Payment Currency" instead of a USD "Amount". Amounts of different currencies are never added up: the amount
targets (CURRENCY_TARGETS) are computed within every currency, with incoming money in the receiving currency and outgoing
money (and the money flow that follows it) in the payment currency. They are stored per currency (per_currency group)
and for the main currency of every account (main_currency group), together with its currency mix (currency group). The
other targets of ego, temporal and flow only count transactions, counterparties or time and stay as they are.
"""

import importlib.util
import os
import sys

import numpy as np
import polars as pl


def _sibling(name: str):
    """Loads another module of etl/aml by path (this module is also loaded by path from the training code)"""
    module_name = f"fin_pse_etl_{name}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(os.path.dirname(__file__), f"{name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


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

# Targets that add up or compare amounts: without currency conversion they are computed within every currency
CURRENCY_TARGETS = [
    "vol_in",
    "vol_out",
    "flow_imbalance",
    "max_amt_in",
    "max_amt_out",
    "amt_cv_in",
    "amt_cv_out",
    "peak_day_vol_in",
    "peak_day_vol_out",
] + [name for name in TARGET_GROUPS["flow"]]

# Fixed currency vocabulary (the AML currencies and Ether), so every dataset has the same columns and one-hot features
_fx = _sibling("fx")
CURRENCIES = list(_fx.CURRENCY_TICKERS) + [_fx.USD]


def currency_column(name: str, currency: str) -> str:
    return f"{name}__{currency.replace(' ', '_')}"


TARGET_GROUPS["currency"] = [
    "n_currencies_in",
    "n_currencies_out",
    "currency_entropy_in",
    "currency_entropy_out",
    "top_currency_share_in",
    "top_currency_share_out",
]
TARGET_GROUPS["main_currency"] = [f"{name}__main" for name in CURRENCY_TARGETS]
TARGET_GROUPS["per_currency"] = [currency_column(name, cur) for name in CURRENCY_TARGETS for cur in CURRENCIES]
USD_GROUPS = ("ego", "temporal", "flow", "motifs")
NATIVE_GROUPS = USD_GROUPS + ("currency", "main_currency", "per_currency")

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


# GFP patterns and the targets derived from them. Simple (length-constrained) cycles are off by default: enumerating
# them is exponential in dense neighbourhoods, while temporal cycles already capture money coming back.
GFP_PATTERNS = {
    "scatter-gather": ["scatter_gather_in", "scatter_gather_out"],
    "temp-cycle": ["temporal_cycles"],
    "lc-cycle": ["simple_cycles"],
}
DEFAULT_GFP_PARAMS = {"scatter-gather": True, "temp-cycle": True, "lc-cycle": False}


def gfp_edge_patterns(trans: pl.DataFrame, gfp_params: dict = None) -> dict:
    """Number of patterns of every enabled type that every transaction takes part in, from IBM's Graph Feature
    Preprocessor (snapml). gfp_params overrides GFP parameters, e.g. {"temp-cycle_tw": 172800, "lc-cycle": True};
    time windows are in seconds and default to the GFP defaults."""
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
    params.update({"num_threads": os.cpu_count(), "vertex_stats": False, "fan": False, "degree": False})
    params.update(DEFAULT_GFP_PARAMS)
    params.update(gfp_params or {})
    gfp.set_params(params)
    features = gfp.transform(edges)

    # Pattern columns follow the edge columns in the order of GFP_PATTERNS, one column per size bin
    counts, column = dict(), edges.shape[1]
    for pattern in GFP_PATTERNS:
        if not params[pattern]:
            continue
        bins = len(params[f"{pattern}_bins"])
        per_edge = np.zeros(len(t))
        per_edge[features[:, 0].astype(np.int64)] = features[:, column : column + bins].sum(axis=1)
        counts[pattern] = per_edge
        column += bins
    return counts


def motif_profile(trans: pl.DataFrame, use_gfp: bool = True, gfp_params: dict = None) -> pl.DataFrame:
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

    counts = gfp_edge_patterns(trans, gfp_params)
    edges = trans.select("From", "To").with_columns([pl.Series(pattern, count) for pattern, count in counts.items()])
    outgoing, incoming = [], []
    if "scatter-gather" in counts:
        outgoing.append(pl.col("scatter-gather").sum().alias("scatter_gather_out"))
        incoming.append(pl.col("scatter-gather").sum().alias("scatter_gather_in"))
    if "temp-cycle" in counts:
        outgoing.append(pl.col("temp-cycle").sum().alias("temporal_cycles"))
    if "lc-cycle" in counts:
        outgoing.append(pl.col("lc-cycle").sum().alias("simple_cycles"))

    if outgoing:
        profile = profile.join(edges.group_by(pl.col("From").alias("Node")).agg(outgoing), on="Node", how="left")
    if incoming:
        profile = profile.join(edges.group_by(pl.col("To").alias("Node")).agg(incoming), on="Node", how="left")
    computed = [name for pattern in counts for name in GFP_PATTERNS[pattern]]
    return profile.with_columns([pl.col(name).fill_null(0.0) for name in computed])


# ---------------------------------------------------------------------------------------------------------------------
# Native currencies


def _currency_views(trans: pl.DataFrame, currency: str):
    """Transactions paid in the currency (amount paid) and transactions received in it (amount received)"""
    paid = trans.filter(pl.col("Payment Currency") == currency).with_columns(pl.col("Amount Paid").alias("Amount"))
    received = trans.filter(pl.col("Receiving Currency") == currency).with_columns(
        pl.col("Amount Received").alias("Amount")
    )
    return paid, received


def currency_profile(trans: pl.DataFrame, currency: str, delta: int = DAY, seed: int = 0) -> pl.DataFrame:
    """The amount targets of one currency for every account that pays or receives in it.

    Outgoing amounts and the money flow (where the money goes, return mass, sinks, onward money) follow the transactions
    paid in the currency; incoming amounts and the in-flow summaries the transactions received in it.
    """
    paid, received = _currency_views(trans, currency)
    nodes = pl.concat([paid.select(pl.col("From").alias("Node")), received.select(pl.col("To").alias("Node"))]).unique()
    if nodes.is_empty():
        return nodes.with_columns([pl.lit(None, dtype=pl.Float64).alias(name) for name in CURRENCY_TARGETS])

    out_side = [name for name in CURRENCY_TARGETS if "_out" in name or name.startswith(("return", "sink", "onward"))]
    in_side = [name for name in CURRENCY_TARGETS if "_in" in name and name not in out_side]

    profile = nodes
    if not paid.is_empty():
        outgoing = (
            ego_profile(paid)
            .join(_peak_days(paid, "From", "To", "out"), on="Node", how="left")
            .join(flow_profile(paid, delta, seed=seed), on="Node", how="left")
        )
        profile = profile.join(outgoing.select(["Node"] + out_side), on="Node", how="left")
    if not received.is_empty():
        incoming = (
            ego_profile(received)
            .join(_peak_days(received, "To", "From", "in"), on="Node", how="left")
            .join(_distribution_summaries(_transition(received, "To", "From"), "in"), on="Node", how="left")
        )
        profile = profile.join(incoming.select(["Node"] + in_side), on="Node", how="left")

    for name in out_side + in_side:
        if name not in profile.columns:
            profile = profile.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))

    # Accounts using the currency on only one side sent or received nothing in it on the other
    return (
        profile.with_columns(pl.col("vol_in").fill_null(0.0), pl.col("vol_out").fill_null(0.0))
        .with_columns(
            ((pl.col("vol_in") - pl.col("vol_out")) / (pl.col("vol_in") + pl.col("vol_out") + EPS)).alias(
                "flow_imbalance"
            )
        )
        .select(["Node"] + [pl.col(name).cast(pl.Float64) for name in CURRENCY_TARGETS])
    )


def _main_currency(trans: pl.DataFrame) -> pl.DataFrame:
    """Currency of most of an account's transactions (received in or paid in), ties broken alphabetically"""
    uses = pl.concat(
        [
            trans.select(pl.col("From").alias("Node"), pl.col("Payment Currency").alias("currency")),
            trans.select(pl.col("To").alias("Node"), pl.col("Receiving Currency").alias("currency")),
        ]
    )
    return (
        uses.group_by(["Node", "currency"])
        .agg(pl.len().alias("n"))
        .sort(["Node", "n", "currency"], descending=[False, True, False])
        .group_by("Node", maintain_order=True)
        .first()
        .select("Node", "currency")
    )


def currency_mix(trans: pl.DataFrame) -> pl.DataFrame:
    """Number of currencies, entropy and top share of the currencies of an account's transactions per direction
    (shares by number of transactions, so amounts of different currencies are never added up)"""

    def side(node, column, suffix):
        return (
            trans.group_by([pl.col(node).alias("Node"), pl.col(column).alias("currency")])
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / pl.col("n").sum().over("Node")).alias("p"))
            .group_by("Node")
            .agg(
                pl.len().alias(f"n_currencies_{suffix}"),
                (-pl.col("p") * (pl.col("p") + EPS).log()).sum().alias(f"currency_entropy_{suffix}"),
                pl.col("p").max().alias(f"top_currency_share_{suffix}"),
            )
        )

    return (
        nodes_of(trans)
        .join(side("To", "Receiving Currency", "in"), on="Node", how="left")
        .join(side("From", "Payment Currency", "out"), on="Node", how="left")
        .with_columns(pl.col("n_currencies_in").fill_null(0), pl.col("n_currencies_out").fill_null(0))
    )


def native_currency_targets(trans: pl.DataFrame, delta: int = DAY, seed: int = 0):
    """Amount targets of every currency (one column per target and currency) and of each account's main currency"""
    per_currency = nodes_of(trans)
    long = []
    for currency in CURRENCIES:
        profile = currency_profile(trans, currency, delta, seed)
        long.append(profile.with_columns(pl.lit(currency).alias("currency")))
        per_currency = per_currency.join(
            profile.rename({name: currency_column(name, currency) for name in CURRENCY_TARGETS}), on="Node", how="left"
        )

    main = (
        _main_currency(trans)
        .join(pl.concat(long), on=["Node", "currency"], how="left")
        .rename({name: f"{name}__main" for name in CURRENCY_TARGETS})
        .drop("currency")
    )
    return per_currency, main


# ---------------------------------------------------------------------------------------------------------------------


def compute_targets(
    trans: pl.DataFrame,
    groups=USD_GROUPS,
    delta: int = DAY,
    use_gfp: bool = True,
    gfp_params: dict = None,
    seed: int = 0,
    native: bool = False,
) -> pl.DataFrame:
    """All targets of the selected groups for every account of the graph, sorted by account.

    With native=True the transactions keep their currencies (see the module docstring): ego, temporal and flow only
    keep their targets that do not add up amounts, and the currency, main_currency and per_currency groups hold the
    amount targets per currency. Motif targets of GFP patterns that are switched off are left out.
    """
    allowed = NATIVE_GROUPS if native else USD_GROUPS
    unknown = [group for group in groups if group not in allowed]
    if unknown:
        raise ValueError(f"Target groups {unknown} need {'USD' if native else 'native'} amounts, choose from {allowed}")

    if native:
        counted = trans.with_columns(pl.lit(1.0).alias("Amount"))  # Amounts are unused by the kept targets
        native_tables = {}

        def per_currency_tables():
            if not native_tables:
                native_tables["per_currency"], native_tables["main_currency"] = native_currency_targets(
                    trans, delta, seed
                )
            return native_tables

    else:
        counted = trans

    profiles = {
        "ego": lambda: ego_profile(counted),
        "temporal": lambda: temporal_profile(counted, delta),
        "flow": lambda: flow_profile(counted, delta, seed=seed),
        "motifs": lambda: motif_profile(trans, use_gfp, gfp_params),
        "currency": lambda: currency_mix(trans),
        "main_currency": lambda: per_currency_tables()["main_currency"],
        "per_currency": lambda: per_currency_tables()["per_currency"],
    }

    targets = nodes_of(trans)
    for group in groups:
        columns = target_columns([group], native, use_gfp, gfp_params)
        targets = targets.join(profiles[group]().select(["Node"] + columns), on="Node", how="left")
    return targets.sort("Node")


def target_columns(groups, native: bool = False, use_gfp: bool = True, gfp_params: dict = None) -> list:
    """Names of the targets compute_targets returns for the given groups and options, in order"""
    gfp = dict(DEFAULT_GFP_PARAMS, **(gfp_params or {}))
    computed_gfp = {name for pattern, names in GFP_PATTERNS.items() if use_gfp and gfp[pattern] for name in names}
    all_gfp = {name for names in GFP_PATTERNS.values() for name in names}

    columns = []
    for group in groups:
        for name in TARGET_GROUPS[group]:
            if name in all_gfp and name not in computed_gfp:
                continue
            # Without conversion, ego/temporal/flow keep what does not add up amounts (supports only count)
            if native and group in USD_GROUPS[:3] and name in CURRENCY_TARGETS and not name.startswith("supp_"):
                continue
            columns.append(name)
    return columns
