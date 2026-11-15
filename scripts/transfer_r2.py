"""Per-target R^2 of a pretrained Fin-PSE encoder (with its pretraining head) on a downstream graph.

The targets are computed exactly on the full downstream graph (etl/aml/targets.py) and compared with the encoder's
predictions, both in the normalised target space of pretraining. Run from the repository root with a downstream
config, e.g.:
    python scripts/transfer_r2.py --cfg configs/aml/aml_small_hi_gin_finpse.yaml gnn.encodings_file <checkpoint>.tar
"""

import json
import logging
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.graphgym import auto_select_device, cfg, load_cfg, parse_args, set_cfg  # noqa: E402
from torch_geometric.graphgym.loader import load_dataset  # noqa: E402

import src  # noqa: E402,F401  (registers the custom configs, datasets and networks)
from src.data.encoder_inputs import transactions_frame  # noqa: E402
from src.data.etl import load_etl_module  # noqa: E402
from src.heads.NodePredictionHead import NodePredictionHead  # noqa: E402
from src.networks.encodings import encode_graph, load_encoder  # noqa: E402
from src.util import set_seed, symlog  # noqa: E402

MIN_ACCOUNTS = 10  # Accounts a target needs on the downstream graph to report its R^2


def load_head(path: str, meta: dict) -> NodePredictionHead:
    """The pretraining head, rebuilt with the encoder width and number of targets of the checkpoint"""
    dim_inner, dim_out = cfg.gnn.dim_inner, cfg.gnn.dim_out
    cfg.gnn.dim_inner, cfg.gnn.dim_out = meta["encoder"]["dim_inner"], len(meta["target_names"])
    head = NodePredictionHead(cfg.gnn.dim_inner, cfg.gnn.dim_out)
    cfg.gnn.dim_inner, cfg.gnn.dim_out = dim_inner, dim_out

    head.load_state_dict(torch.load(path, weights_only=False, map_location="cpu")["head_state_dict"])
    return head.eval()


def exact_targets(data, meta: dict) -> torch.Tensor:
    """The pretraining targets of every account, transformed and normalised as in pretraining (NaN if undefined)"""
    native = meta["currency"] == "native"
    trans = transactions_frame(data.edge_index, data.enc_edge_raw, native)
    names = meta["target_names"]
    use_gfp = any(name in names for name in ("scatter_gather_in", "temporal_cycles"))
    table = load_etl_module("targets").compute_targets(
        trans, meta["target_groups"], use_gfp=use_gfp, seed=cfg.seed, native=native
    )

    y = torch.full((data.num_nodes, len(names)), float("nan"), dtype=torch.float64)
    y[table["Node"].to_numpy()] = torch.tensor(table.select(names).to_numpy().astype(np.float64))
    if meta["log_targets"]:
        y = symlog(y)
    return (y - meta["target_mean"].double()) / meta["target_std"].double()


def main():
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s] %(message)s")
    set_seed(cfg.seed)
    auto_select_device()

    if not cfg.gnn.encodings_precompute:
        raise ValueError("Use a *_finpse.yaml config (gnn.encodings_precompute) so the encoder inputs are prepared")

    path = osp.join(cfg.checkpoint_dir, cfg.gnn.encodings_file)
    encoder, meta = load_encoder(path)
    head = load_head(path, meta)

    data = load_dataset()[2]  # Test graph: all transactions

    # The head was trained on single random feature draws, so its predictions (not the encodings) are averaged
    draws, seed, pred = cfg.gnn.encodings_draws, cfg.seed, 0
    cfg.gnn.encodings_draws = 1
    with torch.no_grad():
        for draw in range(draws):
            cfg.seed = seed + draw
            encodings = encode_graph(encoder, meta, data.edge_index, data.enc_edge_raw, data.num_nodes)
            pred = pred + head.mlp(encodings).double() / draws
    cfg.seed = seed
    y = exact_targets(data, meta)

    has_edges = torch.zeros(data.num_nodes, dtype=torch.bool)
    has_edges[data.edge_index.reshape(-1)] = True
    r2 = dict()
    for k, name in enumerate(meta["target_names"]):
        valid = has_edges & ~torch.isnan(y[:, k])
        total = ((y[valid, k] - y[valid, k].mean()) ** 2).sum()
        # R^2 is undefined without spread (e.g. a per-currency target hardly anyone has in this graph)
        if valid.sum() >= MIN_ACCOUNTS and total / valid.sum() > 1e-8:
            residual = ((pred[valid, k] - y[valid, k]) ** 2).sum()
            r2[name] = float(1 - residual / total)

    pretraining = meta.get("val_r2", {})
    for name, value in r2.items():
        logging.info(
            f"{name:24s} transfer R2 {value:7.3f}   pretraining val R2 {pretraining.get(name, float('nan')):7.3f}"
        )
    skipped = len(meta["target_names"]) - len(r2)
    logging.info(
        f"Transfer R2 over {len(r2)} targets: mean {np.mean(list(r2.values())):.3f}, median "
        f"{np.median(list(r2.values())):.3f} ({skipped} targets without enough accounts or spread)"
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    encoder = osp.splitext(osp.basename(cfg.gnn.encodings_file))[0]
    out = osp.join(cfg.out_dir, f"transfer_r2_{cfg.dataset.name}_{cfg.dataset.table}_{encoder}.json")
    with open(out, "w") as f:
        json.dump({"transfer_r2": r2, "pretraining_val_r2": pretraining}, f, indent=2)
    logging.info(f"Saved {out}")


if __name__ == "__main__":
    main()
