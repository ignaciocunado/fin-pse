from torch import nn
from torch_geometric.graphgym import register_network, cfg
from torch_geometric.graphgym.register import head_dict, act_dict
from torch_geometric.nn import ResGatedGraphConv, BatchNorm
import torch.nn.functional as F


@register_network("fin-pse")
class FinPSE(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(FinPSE, self).__init__()

        self.encoder = FinPSEEncoder(cfg.gnn.dim_in, cfg.gnn.edge_dim, cfg.gnn.dim_inner)
        self.head = head_dict[cfg.gnn.head](cfg.gnn.dim_inner, cfg.gnn.dim_out)  # TODO: Add multiple head outputs

    def forward(self, data):
        data = self.encoder(data)

        return self.head(data)


class FinPSEEncoder(nn.Module):
    def __init__(self, dim_in, edge_dim, dim_inner):
        super(FinPSEEncoder, self).__init__()
        self.node_emb = nn.Linear(dim_in, dim_inner)
        self.edge_emb = nn.Linear(edge_dim, dim_inner)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(cfg.gnn.layers_mp):
            self.convs.append(
                ResGatedGraphConv(
                    dim_inner, dim_inner, edge_dim=dim_inner, act=act_dict[cfg.gnn.act]()
                )
            )
            self.batch_norms.append(BatchNorm(dim_inner))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(len(self.convs)):
            n_new = self.convs[i](x, edge_index, edge_attr)
            x = (x + F.relu(self.batch_norms[i](n_new))) / 2

        data.x = x
        data.edge_attr = edge_attr
        return data
