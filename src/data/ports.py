import numpy as np
import torch


def port_numbers(center: np.ndarray, other: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Port number of every edge at its center node (port numbering from Multi-GNN / MEGA-GNN).

    The neighbours of every center node are numbered in order of their first transaction with it (ties broken by
    neighbour id). The port of an edge is the number of its other endpoint, so parallel edges share a port. Vectorised
    version of Multi-GNN's ports() that scales to AML Large.

    Args:
        center: Node the ports belong to, for every edge
        other: Other endpoint of every edge
        timestamps: Timestamp of every edge
    """
    center, other, timestamps = (
        np.asarray(center, dtype=np.int64),
        np.asarray(other, dtype=np.int64),
        np.asarray(timestamps),
    )
    if len(center) == 0:
        return np.zeros(0, dtype=np.int64)
    num_nodes = int(max(center.max(), other.max())) + 1

    # Unique (center, other) pairs and the time of their first edge
    pair = center * num_nodes + other
    order = np.lexsort((timestamps, pair))
    sorted_pair = pair[order]
    is_first = np.ones(len(order), dtype=bool)
    is_first[1:] = sorted_pair[1:] != sorted_pair[:-1]
    pair_ids = np.cumsum(is_first) - 1
    first_pairs, first_times = sorted_pair[is_first], timestamps[order][is_first]
    pair_center, pair_other = first_pairs // num_nodes, first_pairs % num_nodes

    # Number the neighbours of every center node by their first contact
    pair_order = np.lexsort((pair_other, first_times, pair_center))
    sorted_center = pair_center[pair_order]
    group_start = np.ones(len(pair_order), dtype=bool)
    group_start[1:] = sorted_center[1:] != sorted_center[:-1]
    positions = np.arange(len(pair_order))
    start_positions = np.maximum.accumulate(np.where(group_start, positions, 0))
    pair_ports = np.empty(len(pair_order), dtype=np.int64)
    pair_ports[pair_order] = positions - start_positions

    edge_ports = np.empty(len(pair), dtype=np.int64)
    edge_ports[order] = pair_ports[pair_ids]
    return edge_ports


def add_ports(edge_index: torch.Tensor, edge_attr: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    """Appends the in-port (at the receiver) and the out-port (at the sender) of every edge to its features"""
    src, dst = edge_index.cpu().numpy()
    timestamps = timestamps.cpu().numpy()
    in_ports = port_numbers(dst, src, timestamps)
    out_ports = port_numbers(src, dst, timestamps)
    ports = torch.tensor(np.stack([in_ports, out_ports], axis=1), dtype=edge_attr.dtype)
    return torch.cat([edge_attr, ports], dim=1)
