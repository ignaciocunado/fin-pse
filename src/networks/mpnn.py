import torch.nn as nn
from torch_geometric.graphgym import cfg
from torch_geometric.nn import GINEConv, BatchNorm, Linear, PNAConv
import torch.nn.functional as F
import torch

from torch_geometric.graphgym.register import register_network


@register_network("MPNN")
class MPNN(torch.nn.Module):
    def __init__(self, dim_in=0, dim_out=0):
        super().__init__()
        self.n_hidden = cfg.gnn.dim_inner
        self.final_dropout = cfg.gnn.dropout

        self.node_emb = nn.Linear(cfg.gnn.dim_in, cfg.gnn.dim_inner)
        self.edge_emb = nn.Linear(cfg.gnn.edge_dim, cfg.gnn.dim_inner)

        self.gnn = GnnHelper(
            num_gnn_layers=cfg.gnn.layers_mp,
            n_hidden=cfg.gnn.dim_inner,
            edge_updates=cfg.gnn.emlps,
            final_dropout=cfg.gnn.dropout,
            deg=torch.tensor(cfg.gnn.pna_deg, dtype=torch.float),
        )

    def forward(self, data):
        # Initial Embedding Layers
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)

        # Message Passing Layers
        x, edge_attr = self.gnn(x, data.edge_index, edge_attr)

        data.x = x
        data.edge_attr = edge_attr

        return data

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
