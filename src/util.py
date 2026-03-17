import numpy as np
import torch
import random
import logging
import os
from datetime import datetime
from torch_geometric.graphgym import cfg


def add_arange_ids(data_list):
    """
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    """
    for data in data_list:
        data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def save_model(model, optimizer, epoch):
    # Save the model in a dictionary
    table = cfg.dataset.nodes.replace(".csv", "").replace("_Nodes", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{table}_epoch_{epoch+1}_{timestamp}.tar"
    torch.save(
        {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(cfg.checkpoint_dir, filename),
    )

    return filename


def get_optimizer(optimizer: str, model: torch.nn.Module) -> torch.optim.Optimizer:
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr)
    elif optimizer == "adamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)


def z_norm(data, mean=None, std=None):
    if mean is None and std is None:
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
        return (data - mean) / std, mean, std
    else:
        return (data - mean) / std
