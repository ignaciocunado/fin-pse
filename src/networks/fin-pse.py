from torch import nn
from torch_geometric.graphgym import register_network, cfg
from torch_geometric.graphgym.register import head_dict
from torch_geometric.nn import ResGatedGraphConv, BatchNorm
import torch.nn.functional as F


@register_network("fin-pse")
class FinPSE(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(FinPSE, self).__init__()

        self.dim_in = cfg.gnn.dim_in
        self.dim_out = cfg.gnn.dim_out
        self.dim_hidden = int(cfg.gnn.dim_inner) # For type hinting

        self.node_emb = nn.Linear(cfg.gnn.dim_in, self.dim_hidden)
        self.edge_emb = nn.Linear(cfg.gnn.edge_dim, self.dim_hidden)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(cfg.gnn.layers_mp):
            self.convs.append(
                ResGatedGraphConv(self.dim_hidden, self.dim_hidden, edge_dim=self.dim_hidden, act=cfg.gnn.act)
            )
            self.batch_norms.append(BatchNorm(self.dim_hidden))

        self.head = head_dict[cfg.gnn.head](self.dim_hidden, cfg.gnn.dim_out) # TODO: Add multiple head outputs

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(len(self.convs)):
            n_new = self.convs[i](x, edge_index, edge_attr)
            x = (x + F.relu(self.batch_norms[i](n_new))) / 2

        data.x = x

        return self.head(data)