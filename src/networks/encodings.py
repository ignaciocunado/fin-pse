import hashlib
import json
import logging
import os
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.graphgym import cfg
from torch_geometric.loader import NeighborLoader

from src.data.etl import load_etl_module
from src.data.hetero import node_store, to_hetero_data
from src.data.ports import add_ports
from src.data.ssl_data import z_stats
from src.data.ssl_targets import target_groups
from src.networks.fin_pse import FinPSEEncoder
from src.util import symlog

DAY = 86400


def load_encoder(path: str, load_weights: bool = True):
    """Frozen Fin-PSE encoder rebuilt from a pretraining checkpoint and its metadata (inputs, normalisation)"""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    meta = checkpoint.get("meta")
    if meta is None:
        raise ValueError(
            f"{path} has no Fin-PSE metadata. Precomputed encodings need an encoder pretrained with ssl.target_groups"
        )

    arch = meta["encoder"]
    with torch.random.fork_rng():
        torch.manual_seed(cfg.seed)
        encoder = FinPSEEncoder(
            arch["dim_in"],
            arch["edge_dim"],
            arch["dim_inner"],
            num_layers=arch["layers"],
            reverse_mp=arch["reverse_mp"],
        )
    if load_weights:
        encoder.load_state_dict(checkpoint["model_state_dict"])
    for param in encoder.parameters():
        param.requires_grad = False

    logging.info(f"Encoder {'from ' + path if load_weights else 'with random weights'}: {arch}")
    return encoder.eval(), meta


def exact_feature_names() -> list:
    """Targets used as node features by encodings_source=exact (simple cycles are off, as in the pretraining data)"""
    gfp = {"scatter_gather_in", "scatter_gather_out", "temporal_cycles", "simple_cycles"}
    groups = target_groups()
    return [
        name
        for group in cfg.gnn.exact_target_groups
        for name in groups[group]
        if name != "simple_cycles" and (cfg.gnn.exact_gfp or name not in gfp)
    ]


def encoder_edge_features(edge_index: torch.Tensor, edge_raw: torch.Tensor, meta: dict) -> torch.Tensor:
    """Encoder edge features from [seconds since graph start, USD amount], transformed and normalised exactly as in
    pretraining"""
    edge_raw = edge_raw.double()
    edge_attr = torch.stack(
        [
            edge_raw[:, 0] / DAY if meta["time_in_days"] else edge_raw[:, 0],
            symlog(edge_raw[:, 1]) if meta["log_amounts"] else edge_raw[:, 1],
        ],
        dim=1,
    )
    if meta["ports"]:
        edge_attr = add_ports(edge_index, edge_attr, edge_raw[:, 0])
    return ((edge_attr - meta["edge_mean"].double()) / meta["edge_std"].double()).float()


@torch.no_grad()
def encode_graph(encoder: FinPSEEncoder, meta: dict, edge_index, edge_raw, num_nodes: int) -> torch.Tensor:
    """Encodings of every account of a graph, each computed as a seed with the encoder's own neighbour sampling (as in
    pretraining) and averaged over encodings_draws random node feature draws"""
    arch = meta["encoder"]
    graph = Data(
        x=torch.zeros((num_nodes, arch["dim_in"])),
        y=torch.zeros(num_nodes),
        edge_index=edge_index,
        edge_attr=encoder_edge_features(edge_index, edge_raw, meta),
        num_nodes=num_nodes,
    )
    if arch["reverse_mp"]:
        graph = to_hetero_data(graph, target="node", ports=meta["ports"])

    encoder = encoder.to(cfg.accelerator).eval()
    encodings = torch.zeros((num_nodes, arch["dim_inner"]))
    generator = torch.Generator().manual_seed(cfg.seed)
    for _ in range(cfg.gnn.encodings_draws):
        node_store(graph).x = torch.randn((num_nodes, arch["dim_in"]), generator=generator)
        loader = NeighborLoader(
            graph,
            input_nodes="node" if arch["reverse_mp"] else None,
            num_neighbors=arch["num_neighs"],
            batch_size=cfg.gnn.encodings_batch_size,
            shuffle=False,
        )
        for batch in loader:
            batch = batch.to(cfg.accelerator)
            nodes = node_store(batch)
            if arch["reverse_mp"]:
                h, _ = encoder.encode(nodes.x, batch.edge_index_dict, batch.edge_attr_dict)
            else:
                h, _ = encoder.encode(batch.x, batch.edge_index, batch.edge_attr)
            encodings[nodes.n_id[: nodes.batch_size].cpu()] += h[: nodes.batch_size].cpu()

    return encodings / cfg.gnn.encodings_draws


def exact_features(edge_index, edge_raw, num_nodes: int) -> torch.Tensor:
    """The pretraining targets computed exactly on a graph (log-scaled; NaN where undefined or without transactions)"""
    import polars as pl

    targets = load_etl_module("targets")
    trans = pl.DataFrame(
        {
            "From": edge_index[0].numpy(),
            "To": edge_index[1].numpy(),
            "Timestamp": pl.from_epoch(pl.Series(edge_raw[:, 0].double().round().numpy().astype(np.int64)), "s"),
            "Amount": edge_raw[:, 1].double().numpy(),
        }
    )
    names = exact_feature_names()
    table = targets.compute_targets(trans, cfg.gnn.exact_target_groups, use_gfp=cfg.gnn.exact_gfp, seed=cfg.seed)

    features = torch.full((num_nodes, len(names)), float("nan"), dtype=torch.float64)
    features[table["Node"].to_numpy()] = torch.tensor(table.select(names).to_numpy().astype(np.float64))
    return symlog(features)


def _cache_path(cache_prefix: str, split: int, network) -> str:
    options = {
        "source": cfg.gnn.encodings_source,
        "file": cfg.gnn.encodings_file,
        "draws": cfg.gnn.encodings_draws,
        "seed": cfg.seed,
        "exact": [list(cfg.gnn.exact_target_groups), cfg.gnn.exact_gfp],
        "meta": getattr(network, "encodings_meta", {}).get("encoder"),
    }
    digest = hashlib.md5(json.dumps(options, sort_keys=True, default=str).encode()).hexdigest()[:10]
    directory, stem = osp.dirname(cache_prefix), osp.splitext(osp.basename(cache_prefix))[0]
    return osp.join(directory, "encodings", f"{stem}_{split}_{cfg.gnn.encodings_source}_{digest}.pt")


def attach_node_encodings(model, splits, cache_prefix: str):
    """Precomputes the node encodings of every split graph (cached on disk) and stores them as data.node_encodings.

    The encoder inputs (data.enc_edge_raw) are dropped afterwards; the baseline features and loaders are untouched.
    """
    network = getattr(model, "model", model)
    source = cfg.gnn.encodings_source

    encodings = []
    for k, data in enumerate(splits):
        path = _cache_path(cache_prefix, k, network)
        if osp.exists(path):
            logging.info(f"Loading cached encodings {path}")
            encodings.append(torch.load(path))
            continue

        logging.info(f"Computing {source} encodings of split {k}")
        # Neighbour sampling draws from the global generators: keep them untouched, so training is identical with or
        # without cached encodings
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            if source == "exact":
                values = exact_features(data.edge_index, data.enc_edge_raw, data.num_nodes)
            else:
                values = encode_graph(
                    network.encodings, network.encodings_meta, data.edge_index, data.enc_edge_raw, data.num_nodes
                )
        os.makedirs(osp.dirname(path), exist_ok=True)
        torch.save(values, path)
        encodings.append(values)

    if source == "exact":  # Normalise with the statistics of the training graph, accounts without data get the mean
        mean, std = z_stats(encodings[0])
        encodings = [torch.nan_to_num((values - mean) / std) for values in encodings]

    for data, values in zip(splits, encodings):
        data.node_encodings = values.float()
        del data.enc_edge_raw
