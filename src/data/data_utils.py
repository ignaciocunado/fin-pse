import torch
from torch_geometric.loader import NeighborLoader


def build_ssl_loader_for_window(data, cfg):
    if hasattr(data, "train_mask") and data.train_mask is not None:
        input_nodes = data.train_mask
    else:
        input_nodes = torch.arange(data.num_nodes)

    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=cfg.train.neighbor_sizes,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )
