import logging

import random

import os.path as osp

import torch
from torch_geometric.data import HeteroData
from torch_geometric.graphgym.register import loss_dict
from torch_geometric.loader import NeighborLoader

import wandb
from torch.optim import Optimizer
from torch_geometric.config_store import LRScheduler
from torch_geometric.graphgym import register_train, cfg
from tqdm import tqdm

from src.data.hetero import node_store, to_hetero_data, uses_hetero_data
from src.data.ssl_data import AMLSSL
from src.data.ssl_targets import target_weights
from src.util import save_model


def get_loaders(data, input_nodes=None, shuffle=True):
    # if 'cumulative' in cfg.dataset.nodes or not cfg.ssl.windowed_features:
    if isinstance(data, HeteroData):
        input_nodes = "node" if input_nodes is None else ("node", input_nodes)
    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=cfg.train.num_neighs,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
    )
    # TODO: Check which loader to use
    return data


def masked_loss(pred: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean over targets of the L1 (or L2) error, skipping undefined (NaN) target values"""
    valid = ~torch.isnan(y)
    errors = pred - torch.nan_to_num(y)
    errors = (errors**2 if cfg.model.loss_fun == "l2" else errors.abs()) * valid
    per_target = errors.sum(0) / valid.sum(0).clamp(min=1)
    weights = weights * valid.any(0)
    return (per_target * weights).sum() / weights.sum().clamp(min=1e-12)


def graph_for_model(data):
    if uses_hetero_data():  # Reverse message passing
        return to_hetero_data(data, target="node")
    return data


def train_seeds(data) -> torch.Tensor:
    """Accounts whose targets are trained on this epoch: the non held-out ones, optionally subsampled"""
    seeds = data.train_mask.nonzero().view(-1)
    if 0 < cfg.ssl.max_seeds_per_graph < len(seeds):
        seeds = seeds[torch.randperm(len(seeds))[: cfg.ssl.max_seeds_per_graph]]
    return seeds


@torch.no_grad()
def evaluate(dataset, model, names) -> dict:
    """Loss and per-target R^2 (in normalised target space) on the held-out accounts of every graph"""
    model.eval()
    count = torch.zeros(len(names), dtype=torch.float64)
    total, total_sq, sse, sae = count.clone(), count.clone(), count.clone(), count.clone()

    for i in range(len(dataset)):
        data = dataset[i]
        if not data.val_mask.any():
            continue
        for batch in get_loaders(graph_for_model(data), data.val_mask.nonzero().view(-1), shuffle=False):
            batch.to(cfg.accelerator)
            nodes = node_store(batch)
            if cfg.ssl.resample_random_feats:
                nodes.x = torch.randn_like(nodes.x)
            y = nodes.y[: nodes.batch_size].double().cpu()
            pred = model(batch)[: nodes.batch_size].double().cpu()

            valid = ~torch.isnan(y)
            y, error = torch.nan_to_num(y), (pred - torch.nan_to_num(y)) * valid
            count += valid.sum(0)
            total += (y * valid).sum(0)
            total_sq += (y**2 * valid).sum(0)
            sse += (error**2).sum(0)
            sae += error.abs().sum(0)

    variance = total_sq - total**2 / count.clamp(min=1)
    r2 = 1 - sse / variance.clamp(min=1e-12)
    metrics = {"val_mae": float(sae.sum() / count.sum().clamp(min=1))}
    metrics.update({f"val_r2/{name}": float(r2[k]) for k, name in enumerate(names) if count[k] > 1})
    metrics["val_r2_mean"] = float(r2[count > 1].mean()) if (count > 1).any() else float("nan")
    return metrics


def pretrain_finpse(dataset, model, optimizer):
    """Pretraining on the Fin-PSE data: graphs in random order, held-out accounts for per-target validation"""
    names = dataset.meta["target_names"]
    weights = torch.tensor(
        target_weights(cfg.ssl.target_groups, names, list(cfg.ssl.group_weights)), device=cfg.accelerator
    )
    logging.info(f"Pretraining on {len(names)} targets: {names}")

    idxs = list(range(len(dataset)))
    metrics = dict()
    for epoch in tqdm(range(1, cfg.optim.max_epoch + 1)):
        random.shuffle(idxs)
        model.train()

        epoch_loss, total_seeds = 0.0, 0
        for i in idxs:
            data = dataset[i]
            for batch in get_loaders(graph_for_model(data), train_seeds(data)):
                batch.to(cfg.accelerator)
                nodes = node_store(batch)
                if cfg.ssl.resample_random_feats:
                    nodes.x = torch.randn_like(nodes.x)

                pred = model(batch)[: nodes.batch_size]
                loss = masked_loss(pred, nodes.y[: nodes.batch_size], weights)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss) * nodes.batch_size
                total_seeds += nodes.batch_size

        metrics = evaluate(dataset, model, names)
        logging.info(
            f"Epoch {epoch:03d} | avg_loss={epoch_loss / total_seeds:.6f} | val_mae={metrics['val_mae']:.4f} "
            f"| val_r2_mean={metrics['val_r2_mean']:.4f}"
        )
        wandb.log({"epoch": epoch, "loss": epoch_loss / total_seeds, **metrics})

    logging.info({name: round(value, 3) for name, value in metrics.items() if name.startswith("val_r2/")})
    return metrics


def encoder_meta(dataset, metrics) -> dict:
    """Everything needed to rebuild the encoder and its inputs downstream"""
    meta = dict(dataset.meta)
    meta["encoder"] = {
        "dim_in": cfg.gnn.dim_in,
        "edge_dim": cfg.gnn.edge_dim + (2 if cfg.dataset.ports else 0),
        "dim_inner": cfg.gnn.dim_inner,
        "layers": cfg.gnn.layers_mp,
        "reverse_mp": cfg.gnn.reverse_mp,
        "act": cfg.gnn.act,
        "num_neighs": list(cfg.train.num_neighs),
        "random_feats": True,
    }
    meta["val_r2"] = {name[len("val_r2/") :]: value for name, value in metrics.items() if name.startswith("val_r2/")}
    return meta


def pretrain(
    dataset,
    model,
    optimizer,
    loss_fn,
):
    network = getattr(model, "model", model)  # The GraphGym wrapper holds the FinPSE network

    if dataset.finpse:
        metrics = pretrain_finpse(dataset, model, optimizer)
        if cfg.save_model:
            extra = {"head_state_dict": network.head.state_dict(), "meta": encoder_meta(dataset, metrics)}
            filename = save_model(network.encoder, optimizer, cfg.optim.max_epoch - 1, extra=extra)
            artifact = wandb.Artifact("model_checkpoint", type="model")
            artifact.add_file(osp.join(cfg.checkpoint_dir, filename))
            wandb.log_artifact(artifact)
        return model

    idxs = list(range(len(dataset)))

    step = 0
    for epoch in tqdm(range(1, cfg.optim.max_epoch + 1)):
        random.shuffle(idxs)
        model.train()

        epoch_loss = 0.0
        total_examples = 0
        for i in idxs:
            data = graph_for_model(dataset[i])

            loader = get_loaders(data)

            if isinstance(loader, NeighborLoader):
                for batch in loader:
                    batch.to(cfg.accelerator)
                    nodes = node_store(batch)
                    if cfg.ssl.resample_random_feats:
                        nodes.x = torch.randn_like(nodes.x)

                    pred = model(batch)

                    seeds = nodes.batch_size
                    pred_seeds = pred[:seeds]
                    ground_truth_seeds = nodes.y[:seeds]

                    loss = loss_fn(pred_seeds, ground_truth_seeds)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    optimizer.step()

                    epoch_loss += float(loss.item()) * pred_seeds.numel()
                    total_examples += pred_seeds.numel()
            else:
                pass  # TODO: Implement if needed

            step += 1
        epoch_loss /= total_examples

        logging.info(f"Epoch {epoch:03d} | avg_loss={epoch_loss:.6f}")
        wandb.log({"epoch": epoch, "loss": epoch_loss})

    if cfg.save_model:
        filename = save_model(network.encoder, optimizer, 100)

        artifact = wandb.Artifact("model_checkpoint", type="model")
        artifact.add_file(osp.join(cfg.checkpoint_dir, filename))
        wandb.log_artifact(artifact)

    return model


@register_train("ssl")
def pretrain_model(dataset: AMLSSL, model: torch.nn.Module, optimizer: Optimizer, scheduler: LRScheduler):
    model = pretrain(
        dataset,
        model,
        optimizer,
        loss_dict[cfg.model.loss_fun],
    )

    return model
