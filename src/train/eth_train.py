import torch
from typing import Any, Tuple

import wandb
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.graphgym import register_train, cfg
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import summary
import logging
import numpy as np
import sklearn.metrics

import copy
from torch.optim import Optimizer
from torch.nn import Module
from tqdm import tqdm

from src.data.eth_data import ETHData
from src.util import save_model


def get_loaders(tr_data, val_data, te_data):
    tr_loader = NeighborLoader(
        tr_data,
        input_nodes=tr_data.inds,
        num_neighbors=cfg.train.num_neighs,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        val_data,
        input_nodes=val_data.inds,
        num_neighbors=cfg.train.num_neighs,
        batch_size=cfg.train.batch_size,
        shuffle=False,
    )

    te_loader = NeighborLoader(
        te_data,
        input_nodes=te_data.inds,
        num_neighbors=cfg.train.num_neighs,
        batch_size=cfg.train.batch_size,
        shuffle=False,
    )

    return tr_loader, val_loader, te_loader


def compute_binary_metrics(preds: np.array, labels: np.array):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """
    probs = preds[:, 1]
    preds = preds.argmax(axis=-1)

    precisions, recalls, _ = sklearn.metrics.precision_recall_curve(
        labels, probs
    )  # probs: probabilities for the positive class
    f1 = sklearn.metrics.f1_score(labels, preds, zero_division=0)
    auc = sklearn.metrics.auc(recalls, precisions)

    precision = sklearn.metrics.precision_score(labels, preds, zero_division=0)
    recall = sklearn.metrics.recall_score(labels, preds, zero_division=0)

    return f1, auc, precision, recall


def train_epoch(
    loader: Any,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Trains the model for one epoch.

    Args:
        loader: Data loader for training
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        tr_inds: Training indices

    Returns:
        Tuple containing:
            - Average loss for the epoch
            - Model predictions
            - Ground truth labels
    """
    model.train()
    total_loss = total_seeds = 0
    preds = []
    ground_truths = []

    if cfg.gnn.add_encodings:
        model.model.encodings.eval()  # re-freeze after model.train()

    for batch in tqdm(loader):
        optimizer.zero_grad()
        batch.to(cfg.accelerator)

        out = model(batch)
        seed_out = out[: batch.batch_size]
        seed_y = batch.y[: batch.batch_size]

        loss = loss_fn(seed_out, seed_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_seeds += batch.batch_size

        preds.append(seed_out.detach().cpu())
        ground_truths.append(seed_y.detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    return total_loss / total_seeds, pred, ground_truth


@torch.no_grad()
def eval_epoch(
    loader: Any,
    model: Module,
) -> Tuple[float, float, float, float]:
    """Evaluates the model on the given loader.

    Args:
        loader: Data loader for evaluation
        inds: Evaluation indices
        model: Model to evaluate

    Returns:
        Tuple containing evaluation metrics:
            - F1 score
            - AUC score
            - Precision
            - Recall
    """
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"

    preds = []
    ground_truths = []
    for batch in loader:
        with torch.no_grad():
            batch.to(cfg.accelerator)

            out = model(batch)
            seed_out = out[: batch.batch_size]
            seed_y = batch.y[: batch.batch_size]

            preds.append(seed_out.detach().cpu())
            ground_truths.append(seed_y.detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    # Compute Metrics
    f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)
    return f1, auc, precision, recall


def train(
    tr_loader: Any,
    val_loader: Any,
    te_loader: Any,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
) -> Module:
    """Main training loop with validation and model selection.

    Args:
        tr_loader: Training data loader
        val_loader: Validation data loader
        te_loader: Test data loader
        tr_inds: Training indices
        val_inds: Validation indices
        te_inds: Test indices
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        cfg: Training cfguration

    Returns:
        Module: Best model
    """
    best_val_f1 = 0
    best_state_dict = None

    for epoch in range(cfg.optim.max_epoch):
        logging.info(f"****** EPOCH {epoch} ******")

        # Training phase
        total_loss, pred, ground_truth = train_epoch(tr_loader, model, optimizer, loss_fn)

        # Compute training metrics
        f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)

        # Log training metrics
        logging.info(
            {
                "Train": {
                    "F1": f"{f1:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "PR-AUC": f"{auc:.4f}",
                }
            }
        )

        # Evaluation phase
        val_f1, val_auc, val_precision, val_recall = eval_epoch(val_loader, model)
        te_f1, te_auc, te_precision, te_recall = eval_epoch(te_loader, model)

        # Log validation metrics
        logging.info(
            {
                "Val": {
                    "F1": f"{val_f1:.4f}",
                    "Precision": f"{val_precision:.4f}",
                    "Recall": f"{val_recall:.4f}",
                    "PR-AUC": f"{val_auc:.4f}",
                }
            }
        )

        # Log test metrics
        logging.info(
            {
                "Test": {
                    "F1": f"{te_f1:.4f}",
                    "Precision": f"{te_precision:.4f}",
                    "Recall": f"{te_recall:.4f}",
                    "PR-AUC": f"{te_auc:.4f}",
                }
            }
        )

        # Log loss
        logging.info(f"Loss: {total_loss}")

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": total_loss,
                "val_metrics": {
                    "f1": val_f1,
                    "precision": val_precision,
                    "recall": val_recall,
                    "AUC": val_auc,
                },
                "test_metrics": {
                    "f1": te_f1,
                    "precision": te_precision,
                    "recall": te_recall,
                    "AUC": te_auc,
                },
            }
        )

        # Model selection based on validation F1 score
        if epoch == 0:
            logging.info({"best_test_f1": f"{te_f1:.4f}"})
            best_state_dict = copy.deepcopy(model.state_dict())
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())

            logging.info({"best_test_f1": f"{te_f1:.4f}"})
            wandb.log(
                {
                    "epoch": epoch,
                    "best_val_metric": best_val_f1,
                    "best_test_metric": te_f1,
                }
            )

            # Save best model
            if cfg.save_model:
                save_model(model, optimizer, epoch)

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


@register_train("eth_train")
def train_gnn(dataset: ETHData, model: torch.nn.Module, optimizer: Optimizer, scheduler: LRScheduler):
    tr_data, val_data, te_data = dataset[0], dataset[1], dataset[2]

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data)

    sample_batch = next(iter(tr_loader))
    sample_batch.to(cfg.accelerator)
    logging.info(summary(model, sample_batch))

    # Define loss function and Initialize optimizer
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([cfg.model.w_ce1, cfg.model.w_ce2]).to(cfg.accelerator)
    )

    # Train the model
    model = train(
        tr_loader,
        val_loader,
        te_loader,
        model,
        optimizer,
        loss_fn,
    )

    return model
