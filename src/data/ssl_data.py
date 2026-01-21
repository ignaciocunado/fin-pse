import pandas as pd
import os.path as osp

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym import cfg, register_loader

from src.util import z_norm


class AMLSSL(InMemoryDataset):

    nodes_csv = "HI-Medium_SSL_Nodes.csv"
    trans_csv = "HI-Medium_SSL_Trans.csv"

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_file_names(self):
        return [self.nodes_csv, self.trans_csv]

    def process(self):
        nodes_path = osp.join(self.root, self.nodes_csv)
        trans_path = osp.join(self.root, self.trans_csv)

        nodes = pd.read_csv(nodes_path)
        transactions = pd.read_csv(trans_path)

        # Edges
        edge_features_names = ["Timestamp", "Amount Received", "Received Currency", "Payment Format"]
        edge_index = torch.tensor(transactions[["From", "To"]].to_numpy().T).long()
        edge_attr = z_norm(torch.tensor(transactions[edge_features_names].to_numpy()).float())
        edge_time = torch.tensor(transactions["window_start"].to_numpy()).long()

        # Nodes
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
            "n_currencies_in",
            "currency_entropy_in",
            "top_currency_share_in",
            "n_currencies_out",
            "currency_entropy_out",
            "top_currency_share_out",
            "r2cycle",
        ]
        distinct_nodes = nodes[["Node"]].unique().to_numpy()
        x = torch.ones(len(distinct_nodes)).long()
        node_y = torch.tensor(nodes[node_label_names].to_numpy())
        node_time = torch.tensor(nodes[["window_time"]].to_numpy()).long()

        unique_t = edge_time.unique(sorted=True)
        edges_by_t = {t.item(): (edge_time == t).nonzero(as_tuple=False).view(-1) for t in unique_t}
        node_labels_by_t = {t.item(): (node_time == t).nonzero(as_tuple=False).view(-1) for t in unique_t}

        datas = []
        for t, idx in edges_by_t.items():
            ei = edge_index[:, idx]
            d = Data(
                x=x,
                edge_index=ei,
                edge_attr=edge_attr[idx],
                edge_time=torch.full((ei.size(1),), t, dtype=edge_time.dtype),
                t=t,
                y=node_y[node_labels_by_t[t], :],
            )
            n = x.size(0)
            d.train_mask = torch.ones(n, dtype=torch.bool)
            d.val_mask = torch.zeros(n, dtype=torch.bool)
            d.test_mask = torch.zeros(n, dtype=torch.bool)
            datas.append(d)

        data, slices = self.collate(datas)
        torch.save((data, slices), self.processed_paths[0])


@register_loader("amlssl")
def get_aml_ssl(format, name, dataset_dir):
    root = osp.join('data')
    return AMLSSL(root=root)
