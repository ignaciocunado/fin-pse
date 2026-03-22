import os.path as osp

import torch.nn as nn
from torch_geometric.graphgym import cfg
from torch_geometric.nn import GINEConv, BatchNorm, PNAConv
import torch.nn.functional as F
import torch

from torch_geometric.graphgym.register import register_network, head_dict

from src.networks.fin_pse import FinPSEEncoder


@register_network("MPNN")
class MPNN(torch.nn.Module):
    def __init__(self, dim_in=0, dim_out=0):
        super().__init__()
        self.n_hidden = cfg.gnn.dim_inner
        self.final_dropout = cfg.gnn.dropout

        emb_dim = cfg.gnn.dim_inner // 2 if cfg.gnn.add_encodings else cfg.gnn.dim_inner

        self.node_emb = nn.Linear(cfg.gnn.dim_in, emb_dim)
        self.edge_emb = nn.Linear(cfg.gnn.edge_dim, emb_dim)

        if cfg.gnn.add_encodings:
            self.encodings = FinPSEEncoder(cfg.gnn.dim_in, 2, 64)
            self.encodings.load_state_dict(
                torch.load(
                    osp.join(cfg.checkpoint_dir, cfg.gnn.encodings_file),
                    weights_only=True,
                    map_location=cfg.accelerator,
                )["model_state_dict"]
            )
            for param in self.encodings.parameters():
                param.requires_grad = False

            self.project_node_encodings = nn.Sequential(
                nn.Linear(64, cfg.gnn.dim_inner),
                nn.ReLU(),
                nn.BatchNorm1d(cfg.gnn.dim_inner),
                nn.Linear(cfg.gnn.dim_inner, emb_dim),
            )
            self.project_edge_encodings = nn.Sequential(
                nn.Linear(64, cfg.gnn.dim_inner),
                nn.ReLU(),
                nn.BatchNorm1d(cfg.gnn.dim_inner),
                nn.Linear(cfg.gnn.dim_inner, emb_dim),
            )

        self.gnn = GnnHelper(
            num_gnn_layers=cfg.gnn.layers_mp,
            n_hidden=cfg.gnn.dim_inner,
            edge_updates=cfg.gnn.emlps,
            final_dropout=cfg.gnn.dropout,
            deg=torch.tensor(cfg.gnn.pna_deg, dtype=torch.float) if cfg.gnn.layer_type == "pna" else None,
        )

        self.head = head_dict[cfg.gnn.head](cfg.gnn.dim_inner, cfg.gnn.dim_out)

    def forward(self, data):
        # Initial Embedding Layers
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)

        if cfg.gnn.add_encodings:
            data_encodings = data.clone()
            data_encodings.edge_attr = data_encodings.edge_attr[:, :2]

            if cfg.gnn.encodings_random_feats:
                data_encodings.x = torch.randn(data_encodings.x.shape, device=cfg.accelerator)

            with torch.no_grad():
                data_encodings = self.encodings(data_encodings)
            projected_node_encodings = self.project_node_encodings(data_encodings.x)
            projected_edge_encodings = self.project_edge_encodings(data_encodings.edge_attr)
            x = torch.cat([x, projected_node_encodings], dim=1)
            edge_attr = torch.cat([edge_attr, projected_edge_encodings], dim=1)

        # Message Passing Layers
        x, edge_attr = self.gnn(x, data.edge_index, edge_attr)

        data.x = x
        data.edge_attr = edge_attr

        return self.head(data)


class GnnHelper(torch.nn.Module):
    def __init__(self, num_gnn_layers, n_hidden=100, edge_updates=False, final_dropout=0.5, deg=None):
        super().__init__()

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.final_dropout = final_dropout
        self.edge_updates = edge_updates

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            if cfg.gnn.layer_type == "gin":
                conv = GINEConv(
                    nn.Sequential(
                        nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Linear(self.n_hidden, self.n_hidden)
                    ),
                    edge_dim=self.n_hidden,
                )
            elif cfg.gnn.layer_type == "pna":
                aggregators = ["mean", "min", "max", "std"]
                scalers = ["identity", "amplification", "attenuation"]
                conv = PNAConv(
                    in_channels=n_hidden,
                    out_channels=n_hidden,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    edge_dim=n_hidden,
                    towers=5,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )

            if self.edge_updates:
                self.emlps.append(
                    nn.Sequential(
                        nn.Linear(3 * self.n_hidden, self.n_hidden),
                        nn.ReLU(),
                        nn.Linear(self.n_hidden, self.n_hidden),
                    )
                )

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        for i in range(self.num_gnn_layers):
            n_new = self.convs[i](x, edge_index, edge_attr)
            x = (x + F.relu(self.batch_norms[i](n_new))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        return x, edge_attr
