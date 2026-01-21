import pandas as pd
import os.path as osp

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym import cfg, register_dataset

from util import z_norm

@register_dataset('amlssl')
class AMLSSL(InMemoryDataset):

    nodes_csv = 'HI-Medium_SSL_Nodes.csv'
    trans_csv= 'HI-Medium_SSL_Trans.csv'

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
        nodes_path = osp.join(self.root, 'data', self.nodes_csv)
        trans_path = osp.join(self.root, 'data', self.trans_csv)

        nodes = pd.read_csv(nodes_path)
        transactions = pd.read_csv(trans_path)


        # Edges
        edge_features_names = ["Timestamp", "Amount Received", "Received Currency", "Payment Format"]
        edge_index = torch.tensor(transactions[['From', 'To']].to_numpy().T).long()
        edge_attr = z_norm(torch.tensor(transactions[edge_features_names].to_numpy()).float())
        edge_time = torch.tensor(transactions['window_start'].to_numpy()).long()

        # Nodes
        node_feature_names = ['Feature']
        node_label_names = ['mu_gap_in_sec', 'var_gap_in_sec', 'mu_gap_out_sec', 'var_gap_out_sec', 'deg_in', 'deg_out',
                              'fan_in', 'fan_out', 'vol_in', 'vol_out', 'flow_imbalance', 'n_currencies_in',
                              'currency_entropy_in', 'top_currency_share_in', 'n_currencies_out', 'currency_entropy_out',
                              'top_currency_share_out', 'r2cycle']
        x = torch.tensor(nodes[node_feature_names].to_numpy())
        node_y = torch.tensor(nodes[node_label_names].to_numpy())
        window_start = torch.load(osp.join(self.root, "raw", "window_start.pt")).long()

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_time=edge_time,
            edge_attr=edge_attr if "edge_attr" in locals() else None,
            y = node_y
        )

        n = data.num_nodes
        data.train_mask = torch.ones(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

