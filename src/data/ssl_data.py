import numpy as np
import pandas as pd
import os.path as osp

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym import register_loader, cfg

from src.data.graph_data import GraphData
from src.util import z_norm


class AMLSSL(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, nodes=None, edges=None):
        self.nodes_csv = nodes
        self.trans_csv = edges # CSV file names

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.nodes_csv.replace('.csv', '').replace('_Nodes', '') + "data.pt"]

    @property
    def raw_file_names(self):
        return [self.nodes_csv, self.trans_csv]

    def process(self):
        nodes_path = osp.join(self.raw_dir, self.nodes_csv)
        trans_path = osp.join(self.raw_dir, self.trans_csv)

        nodes = pd.read_csv(nodes_path)
        transactions = pd.read_csv(trans_path)

        if 'window_start' not in transactions.columns:
            transactions['window_start'] = 0 # Case when we do not group data by windows

        all_ids = pd.unique(
            pd.concat([transactions["From"], transactions["To"], nodes["Node"]], ignore_index=True)
        )
        all_ids = pd.Series(all_ids).sort_values().to_numpy()

        id2idx = {int(nid): i for i, nid in enumerate(all_ids)}
        num_nodes = len(all_ids)

        x = torch.ones((num_nodes, 1), dtype=torch.float)

        node_label_names = [
            "mu_gap_in_sec",
            "var_gap_in_sec",
            "mu_gap_out_sec",
            "var_gap_out_sec",
            "deg_in",
            "deg_out",
            "fan_in",
            "fan_out",
            "vol_in",
            "vol_out",
            "flow_imbalance",
            "r_2cycle",
        ]

        edge_feature_names = ["Timestamp", "Payment Format"]

        if not cfg.ssl.convert_currencies:
            node_label_names = node_label_names + [
                "n_currencies_in",
                "currency_entropy_in",
                "top_currency_share_in",
                "n_currencies_out",
                "currency_entropy_out",
                "top_currency_share_out"
            ]
            edge_feature_names = edge_feature_names + ["Amount Received", "Amount Paid", "Receiving Currency", "Payment Currency"]
        else:
            edge_feature_names = edge_feature_names + ["Amount"]

        y_dim = len(node_label_names)

        node_idx = nodes["Node"].map(id2idx).astype("int64").to_numpy()
        node_time = torch.tensor(nodes["window_start"].to_numpy()).long()
        node_y_vals = torch.tensor(nodes[node_label_names].to_numpy(), dtype=torch.float)

        src = transactions["From"].map(id2idx).astype("int64").to_numpy()
        dst = transactions["To"].map(id2idx).astype("int64").to_numpy()
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

        edge_attr = torch.tensor(transactions[edge_feature_names].to_numpy(), dtype=torch.float)
        edge_attr, _, _ = z_norm(edge_attr)

        edge_time = torch.tensor(transactions["window_start"].to_numpy()).long()
        unique_t = edge_time.unique(sorted=True)
        edges_by_t = {t.item(): (edge_time == t).nonzero(as_tuple=False).view(-1) for t in unique_t}

        node_rows_by_t = {t.item(): (node_time == t).nonzero(as_tuple=False).view(-1) for t in unique_t}

        datas = []
        for t, eidx in edges_by_t.items():
            ei = edge_index[:, eidx]

            y_ego = torch.zeros((num_nodes, y_dim), dtype=torch.float)
            y_ego_mask = torch.zeros((num_nodes,), dtype=torch.bool)

            if t in node_rows_by_t:
                ridx = node_rows_by_t[t]
                nidx = torch.tensor(node_idx, dtype=torch.long)[ridx]
                y_ego[nidx] = node_y_vals[ridx]
                y_ego_mask[nidx] = True

            d = GraphData(
                x=x,
                edge_index=ei,
                edge_attr=edge_attr[eidx],
                edge_time=torch.full((ei.size(1),), t, dtype=edge_time.dtype),
                t=t,
                y_ego=y_ego,
                y_ego_mask=y_ego_mask,
            )

            d.train_mask = torch.ones(num_nodes, dtype=torch.bool)
            d.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            d.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            datas.append(d)

        data, slices = self.collate(datas)
        torch.save((data, slices), self.processed_paths[0])


@register_loader("AMLSSL")
def get_aml_ssl(format, name, dataset_dir):
    if name != "AMLSSL":
        return None
    root = osp.join("data")
    return AMLSSL(root=root, nodes=cfg.dataset.nodes, edges=cfg.dataset.edges)
