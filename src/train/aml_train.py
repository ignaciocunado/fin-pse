import torch
from typing import Any, Tuple

from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.graphgym import register_train, cfg
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import summary
import logging
import numpy as np
import sklearn.metrics

import copy
from torch.optim import Optimizer
from torch.nn import Module
from tqdm import tqdm

from src.data.aml_data import AMLData
from src.util import add_arange_ids, save_model


def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform):
    tr_loader = LinkNeighborLoader(
        tr_data, num_neighbors=cfg.train.num_neighs, batch_size=cfg.train.batch_size, shuffle=True, transform=transform
    )
    val_loader = LinkNeighborLoader(
        val_data,
        num_neighbors=cfg.train.num_neighs,
        edge_label_index=val_data.edge_index[:, val_inds],
        edge_label=val_data.y[val_inds],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        transform=transform,
    )
    te_loader = LinkNeighborLoader(
        te_data,
        num_neighbors=cfg.train.num_neighs,
        edge_label_index=te_data.edge_index[:, te_inds],
        edge_label=te_data.y[te_inds],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        transform=transform,
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
    tr_inds: torch.Tensor,
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
    total_loss = total_examples = 0
    preds = []
    ground_truths = []

    for batch in tqdm(loader):
        optimizer.zero_grad()

        # Select the seed edges from which the batch was created
        inds = tr_inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        # Remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        batch.to(cfg.accelerator)

        out = model(batch)

        pred = out[mask]
        ground_truth = batch.y[mask]
        loss = loss_fn(pred, ground_truth)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        preds.append(pred.detach().cpu())
        ground_truths.append(ground_truth.detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    return total_loss / total_examples, pred, ground_truth


@torch.no_grad()
def eval_epoch(
    loader: Any,
    inds: torch.Tensor,
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
        # select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        # remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        with torch.no_grad():
            batch.to(cfg.accelerator)

            out = model(batch)
            pred = out[mask]
            preds.append(pred.detach().cpu())
            ground_truths.append(batch.y[mask].detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    # Compute Metrics
    f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)
    return f1, auc, precision, recall


def train(
    tr_loader: Any,
    val_loader: Any,
    te_loader: Any,
    tr_inds: torch.Tensor,
    val_inds: torch.Tensor,
    te_inds: torch.Tensor,
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

    for epoch in range(cfg.epochs):
        logging.info(f"****** EPOCH {epoch} ******")

        # Training phase
        total_loss, pred, ground_truth = train_epoch(tr_loader, model, optimizer, loss_fn, tr_inds)

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
        val_f1, val_auc, val_precision, val_recall = eval_epoch(val_loader, val_inds, model)
        te_f1, te_auc, te_precision, te_recall = eval_epoch(te_loader, te_inds, model)

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

        # Model selection based on validation F1 score
        if epoch == 0:
            logging.info({"best_test_f1": f"{te_f1:.4f}"})
            best_state_dict = copy.deepcopy(model.state_dict())
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            logging.info({"best_test_f1": f"{te_f1:.4f}"})

            # Save best model
            if cfg.save_model:
                save_model(model, optimizer, epoch)

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


@register_train("aml_train")
def train_gnn(dataset: AMLData, model: torch.nn.Module, optimizer: Optimizer, scheduler: LRScheduler):
    tr_data, val_data, te_data = dataset[0], dataset[1], dataset[2]
    tr_inds, val_inds, te_inds = tr_data.inds, val_data.inds, te_data.inds

    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform=None)

    # Get a sample batch and initialize the model
    sample_batch = next(iter(tr_loader))
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]

    # Move sample batch to device and log model summary
    sample_batch.to(cfg.accelerator)
    logging.info(summary(model, sample_batch))

    # Define loss function and Initialize optimizer
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([cfg.model.w_ce1, cfg.model.w_ce2]).to(cfg.accelerator))

    # Train the model
    model = train(
        tr_loader,
        val_loader,
        te_loader,
        tr_inds,
        val_inds,
        te_inds,
        model,
        optimizer,
        loss_fn,
    )

    return model
