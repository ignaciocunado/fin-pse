import numpy as np
import pandas as pd
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym import register_loader, cfg

from src.data.graph_data import GraphData
from src.util import z_norm


class AMLSSL(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, nodes=None, edges=None):
        self.nodes_csv = nodes
        self.trans_csv = edges  # CSV file names

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.nodes_csv.replace(".csv", "").replace("_Nodes", "") + "data.pt"]

    @property
    def raw_file_names(self):
        return [self.nodes_csv, self.trans_csv]

    def process(self):
        nodes_path = osp.join(self.raw_dir, self.nodes_csv)
        trans_path = osp.join(self.raw_dir, self.trans_csv)

        nodes = pd.read_csv(nodes_path)
        transactions = pd.read_csv(trans_path)

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
        ]  # mugapin / mugapout

        edge_feature_names = ["Timestamp"]

        if not cfg.ssl.convert_currencies:
            node_label_names = node_label_names + [
                "n_currencies_in",
                "currency_entropy_in",
                "top_currency_share_in",
                "n_currencies_out",
                "currency_entropy_out",
                "top_currency_share_out",
            ]
            edge_feature_names = edge_feature_names + [
                "Amount Received",
                "Amount Paid",
                "Receiving Currency",
                "Payment Currency",
            ]
        else:
            edge_feature_names = edge_feature_names + ["Amount"]

        if not cfg.ssl.windowed_features:  # When data is not grouped
            all_ids = pd.unique(pd.concat([transactions["From"], transactions["To"], nodes["Node"]], ignore_index=True))
            all_ids = pd.Series(all_ids).sort_values().to_numpy()

            id2idx = {int(nid): i for i, nid in enumerate(all_ids)}
            num_nodes = len(all_ids)

            x = torch.randn((num_nodes, 1), dtype=torch.float)

            src = transactions["From"].map(id2idx).astype("int64").to_numpy()
            dst = transactions["To"].map(id2idx).astype("int64").to_numpy()
            edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

            edge_attr = torch.tensor(transactions[edge_feature_names].to_numpy(), dtype=torch.float)
            edge_attr, _, _ = z_norm(edge_attr)

            labels = nodes[node_label_names + ["Node"]]
            labels["Index"] = labels["Node"].map(id2idx)
            y = labels.sort_values("Index").drop(["Index", "Node"], axis=1)
            y = torch.tensor(y.to_numpy(), dtype=torch.float)
            y, _, _ = z_norm(y)

            data = GraphData(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            data, slices = self.collate([data])
            torch.save((data, slices), self.processed_paths[0])
            return

        grouped_nodes = nodes.sort_values("window_start").groupby("window_start")
        grouped_trans = transactions.sort_values("window_start").groupby("window_start")

        datas = []
        for (date, n), (_, e) in zip(grouped_nodes, grouped_trans):
            all_ids = pd.unique(pd.concat([e["From"], e["To"], n["Node"]], ignore_index=True))
            all_ids = pd.Series(all_ids).sort_values().to_numpy()
            num_nodes = len(all_ids)

            x = torch.randn((num_nodes, 1), dtype=torch.float)
            id2idx = {int(nid): i for i, nid in enumerate(all_ids)}

            src = e["From"].map(id2idx).astype("int64").to_numpy()
            dst = e["To"].map(id2idx).astype("int64").to_numpy()
            edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
            edge_attr = torch.tensor(e[edge_feature_names].to_numpy(), dtype=torch.float)
            edge_attr, _, _ = z_norm(edge_attr)

            n["Node ID"] = n["Node"].map(id2idx)
            y = torch.tensor(
                n[node_label_names + ["Node ID"]].sort_values("Node ID").drop("Node ID", axis=1).to_numpy(),
                dtype=torch.float,
            )
            y, _, _ = z_norm(y)

            data = GraphData(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_attr=edge_attr,
                t=date,
            )
            datas.append(data)

            data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data, slices = self.collate(datas)
        torch.save((data, slices), self.processed_paths[0])
        return


@register_loader("AMLSSL")
def get_aml_ssl(format, name, dataset_dir):
    if name != "AMLSSL":
        return None
    root = osp.join(cfg.root_dir, "data")
    return AMLSSL(root=root, nodes=cfg.dataset.nodes, edges=cfg.dataset.edges)
