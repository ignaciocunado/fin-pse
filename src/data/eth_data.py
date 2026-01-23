import itertools
import logging
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym import register_loader, cfg

from src.data.graph_data import GraphData
from src.util import z_norm

import os.path as osp


class ETHData(InMemoryDataset):

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['eth_nodes.csv', 'eth_edges.csv']

    @property
    def processed_file_names(self):
        return ['eth_data.pt']

    def process(self):
        nodes = pd.read_csv(self.raw_paths[0]).sort_values('node_id')
        edges = pd.read_csv(self.raw_paths[1])

        nodes = nodes.sort_values("node_id").reset_index(drop=True)

        node_ids = nodes["node_id"].to_numpy()
        id2idx = {int(nid): i for i, nid in enumerate(node_ids)}
        N = len(node_ids)

        edges["from_idx"] = edges["from"].map(id2idx)
        edges["to_idx"] = edges["to"].map(id2idx)
        edges = edges.dropna(subset=["from_idx", "to_idx"]).copy()
        edges["from_idx"] = edges["from_idx"].astype(np.int64)
        edges["to_idx"] = edges["to_idx"].astype(np.int64)

        edges["timestamp"] = pd.to_numeric(edges["timestamp"], errors="coerce")
        edges = edges.dropna(subset=["timestamp"]).copy()
        edges["timestamp"] = edges["timestamp"].astype(np.float64)

        x = torch.ones((N, 1), dtype=torch.float32)
        y = torch.tensor(nodes["isp"].to_numpy(), dtype=torch.long)

        first_ts = nodes["first_timestamp"].to_numpy().astype(np.float64)
        t1 = float(np.quantile(first_ts, 0.65))
        t2 = float(np.quantile(first_ts, 0.80))
        tmax = float(edges["timestamp"].max())

        tr_inds  = torch.tensor(np.where(first_ts <= t1)[0], dtype=torch.long)
        val_inds = torch.tensor(np.where((first_ts > t1) & (first_ts <= t2))[0], dtype=torch.long)
        te_inds  = torch.tensor(np.where(first_ts > t2)[0], dtype=torch.long)

        tr_edges  = edges[edges["timestamp"] <= t1]
        val_edges = edges[edges["timestamp"] <= t2]
        te_edges  = edges

        def pack(df):
            ei = torch.tensor(df[["from_idx", "to_idx"]].to_numpy().T, dtype=torch.long)
            ea = torch.tensor(df[["amount", "timestamp"]].to_numpy(), dtype=torch.float32)
            et = torch.tensor(df["timestamp"].to_numpy(), dtype=torch.float32)
            return ei, ea, et

        tr_edge_index, tr_edge_attr, tr_edge_times = pack(tr_edges)
        val_edge_index, val_edge_attr, val_edge_times = pack(val_edges)
        te_edge_index, te_edge_attr, te_edge_times = pack(te_edges)

        tr_data = GraphData(x=x, y=y, edge_index=tr_edge_index, edge_attr=tr_edge_attr, timestamps=tr_edge_times)
        val_data = GraphData(x=x, y=y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
        te_data = GraphData(x=x, y=y, edge_index=te_edge_index, edge_attr=te_edge_attr, timestamps=te_edge_times)

        tr_data.edge_attr, mean, var = z_norm(tr_data.edge_attr)
        val_data.edge_attr = z_norm(val_data.edge_attr, mean, var)
        te_data.edge_attr  = z_norm(te_data.edge_attr, mean, var)

        tr_data.inds, val_data.inds, te_data.inds = tr_inds, val_inds, te_inds

        self.save([tr_data, val_data, te_data], self.processed_paths[0])


@register_loader("eth")
def get_aml(format, name, dataset_dir):
    root = osp.join(cfg.root_dir, "data")
    return ETHData(root=root)