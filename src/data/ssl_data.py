import numpy as np
import pandas as pd
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym import register_loader, cfg

from src.data.graph_data import GraphData
from src.data.ports import add_ports
from src.data.ssl_targets import target_names
from src.util import processed_suffix, symlog, z_norm

DAY = 86400


class AMLSSL(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, nodes=None, edges=None):
        self.nodes_csv = nodes
        self.trans_csv = edges  # CSV file names

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        # Target names and normalisation statistics of the Fin-PSE pretraining data, saved with the encoder
        self.meta = torch.load(self.processed_paths[1], weights_only=False) if self.finpse else None

    @property
    def finpse(self) -> bool:
        """Pretraining data built by etl/aml/build_ssl.py (graphs with a graph_id, configurable target groups)"""
        return len(cfg.ssl.target_groups) > 0

    @property
    def processed_file_names(self):
        suffix = processed_suffix(log_targets=cfg.ssl.log_targets)
        name = self.nodes_csv.replace(".csv", "").replace("_Nodes", "")
        if self.finpse:
            options = "-".join(cfg.ssl.target_groups) + suffix + ("_days" if cfg.ssl.time_in_days else "")
            options += f"_val{round(cfg.ssl.val_fraction * 100)}"
            return [f"{name}_{options}.pt", f"{name}_{options}_meta.pt"]
        return [name + f"data{suffix}.pt"]

    @property
    def raw_file_names(self):
        return [self.nodes_csv, self.trans_csv]

    def process(self):
        nodes_path = osp.join(self.raw_dir, self.nodes_csv)
        trans_path = osp.join(self.raw_dir, self.trans_csv)

        nodes = pd.read_csv(nodes_path)
        transactions = pd.read_csv(trans_path)

        if self.finpse:
            return self.process_finpse(nodes, transactions)

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

        amount_features = [i for i, name in enumerate(edge_feature_names) if name.startswith("Amount")]

        def edge_features(e, edge_index):
            edge_attr = torch.tensor(e[edge_feature_names].to_numpy(), dtype=torch.float)
            if cfg.dataset.log_amounts:
                edge_attr[:, amount_features] = symlog(edge_attr[:, amount_features])
            if cfg.dataset.ports:
                edge_attr = add_ports(edge_index, edge_attr, torch.tensor(e["Timestamp"].to_numpy()))
            return z_norm(edge_attr)[0]

        def targets(y):
            y = torch.tensor(y, dtype=torch.float)
            if cfg.ssl.log_targets:
                y = symlog(y)
            return z_norm(y)[0]

        if not cfg.ssl.windowed_features:  # When data is not grouped
            all_ids = pd.unique(pd.concat([transactions["From"], transactions["To"], nodes["Node"]], ignore_index=True))
            all_ids = pd.Series(all_ids).sort_values().to_numpy()

            id2idx = {int(nid): i for i, nid in enumerate(all_ids)}
            num_nodes = len(all_ids)

            x = torch.randn((num_nodes, 1), dtype=torch.float)

            src = transactions["From"].map(id2idx).astype("int64").to_numpy()
            dst = transactions["To"].map(id2idx).astype("int64").to_numpy()
            edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

            edge_attr = edge_features(transactions, edge_index)

            labels = nodes[node_label_names + ["Node"]]
            labels["Index"] = labels["Node"].map(id2idx)
            y = labels.sort_values("Index").drop(["Index", "Node"], axis=1)
            y = targets(y.to_numpy())

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
            edge_attr = edge_features(e, edge_index)

            n["Node ID"] = n["Node"].map(id2idx)
            y = targets(n[node_label_names + ["Node ID"]].sort_values("Node ID").drop("Node ID", axis=1).to_numpy())

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

    def process_finpse(self, nodes: pd.DataFrame, transactions: pd.DataFrame):
        """One graph per graph_id (whole graph and time crops) with the targets of the selected groups.

        Edge features are [Timestamp, Amount] (+ ports): timestamps in days since the start of the graph, amounts in
        USD (optionally log-scaled). Targets may be NaN where undefined; the loss skips them. Edge features and targets
        are normalised with statistics over all graphs (not per graph), which are stored with the data and later with
        the encoder, so downstream graphs are normalised identically.
        """
        names = target_names(cfg.ssl.target_groups, nodes.columns)
        edge_names = ["Timestamp", "Amount"] + (["in_port", "out_port"] if cfg.dataset.ports else [])
        generator = torch.Generator().manual_seed(0)  # Same held-out accounts for every run

        graphs = []
        grouped_nodes, grouped_trans = nodes.groupby("graph_id", sort=True), transactions.groupby("graph_id", sort=True)
        for (graph_id, n), (_, e) in zip(grouped_nodes, grouped_trans):
            ids = np.sort(pd.unique(pd.concat([e["From"], e["To"], n["Node"]], ignore_index=True)))
            id2idx = pd.Series(np.arange(len(ids)), index=ids)
            edge_index = torch.tensor(
                np.array([id2idx[e["From"]].to_numpy(), id2idx[e["To"]].to_numpy()]), dtype=torch.long
            )

            edge_attr = torch.tensor(e[["Timestamp", "Amount"]].to_numpy(dtype=np.float64))
            if cfg.ssl.time_in_days:
                edge_attr[:, 0] /= DAY
            if cfg.dataset.log_amounts:
                edge_attr[:, 1] = symlog(edge_attr[:, 1])
            if cfg.dataset.ports:
                edge_attr = add_ports(edge_index, edge_attr, torch.tensor(e["Timestamp"].to_numpy()))

            y = torch.full((len(ids), len(names)), float("nan"), dtype=torch.float64)
            y[id2idx[n["Node"]].to_numpy()] = torch.tensor(n[names].to_numpy(dtype=np.float64))
            if cfg.ssl.log_targets:
                y = symlog(y)

            val_mask = torch.rand(len(ids), generator=generator) < cfg.ssl.val_fraction
            graphs.append((int(graph_id), edge_index, edge_attr, y, val_mask))

        edge_mean, edge_std = z_stats(torch.cat([graph[2] for graph in graphs]))
        y_mean, y_std = z_stats(torch.cat([graph[3] for graph in graphs]))

        datas = []
        for graph_id, edge_index, edge_attr, y, val_mask in graphs:
            num_nodes = y.shape[0]
            data = GraphData(
                x=torch.randn((num_nodes, 1), dtype=torch.float),
                y=((y - y_mean) / y_std).float(),
                edge_index=edge_index,
                edge_attr=((edge_attr - edge_mean) / edge_std).float(),
                graph_id=graph_id,
            )
            data.train_mask, data.val_mask = ~val_mask, val_mask
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            datas.append(data)

        data, slices = self.collate(datas)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(
            {
                "target_groups": list(cfg.ssl.target_groups),
                "target_names": names,
                "target_mean": y_mean.float(),
                "target_std": y_std.float(),
                "log_targets": cfg.ssl.log_targets,
                "edge_feature_names": edge_names,
                "edge_mean": edge_mean.float(),
                "edge_std": edge_std.float(),
                "time_in_days": cfg.ssl.time_in_days,
                "log_amounts": cfg.dataset.log_amounts,
                "ports": cfg.dataset.ports,
                "currency": "daily_usd",
            },
            self.processed_paths[1],
        )


def z_stats(values: torch.Tensor):
    """Column means and standard deviations ignoring NaNs (std 1 for constant or empty columns)"""
    mean = torch.nanmean(values, dim=0)
    std = torch.sqrt(torch.nanmean((values - mean) ** 2, dim=0))
    mean = torch.nan_to_num(mean, nan=0.0)
    std = torch.where(torch.isnan(std) | (std == 0), torch.ones_like(std), std)
    return mean, std


@register_loader("AMLSSL")
def get_aml_ssl(format, name, dataset_dir):
    if name != "AMLSSL":
        return None
    root = osp.join(cfg.root_dir, "data")
    dataset = AMLSSL(root=root, nodes=cfg.dataset.nodes, edges=cfg.dataset.edges)
    if dataset.finpse:
        cfg.gnn.dim_out = len(dataset.meta["target_names"])  # One output per target
    return dataset
