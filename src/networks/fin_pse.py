from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.graphgym import register_network, cfg
from torch_geometric.graphgym.register import head_dict, act_dict
from torch_geometric.nn import ResGatedGraphConv, BatchNorm, HeteroConv
import torch.nn.functional as F

from src.data.hetero import EDGE, REV_EDGE


@register_network("fin-pse")
class FinPSE(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(FinPSE, self).__init__()

        edge_dim = cfg.gnn.edge_dim + (2 if cfg.dataset.ports else 0)
        self.encoder = FinPSEEncoder(cfg.gnn.dim_in, edge_dim, cfg.gnn.dim_inner, reverse_mp=cfg.gnn.reverse_mp)
        self.head = head_dict[cfg.gnn.head](cfg.gnn.dim_inner, cfg.gnn.dim_out)  # TODO: Add multiple head outputs

    def forward(self, data):
        data = self.encoder(data)

        return self.head(data)


class FinPSEEncoder(nn.Module):
    def __init__(self, dim_in, edge_dim, dim_inner, num_layers=None, reverse_mp=False):
        super(FinPSEEncoder, self).__init__()
        self.reverse_mp = reverse_mp

        self.node_emb = nn.Linear(dim_in, dim_inner)
        self.edge_emb = nn.Linear(edge_dim, dim_inner)
        if reverse_mp:
            self.edge_emb_rev = nn.Linear(edge_dim, dim_inner)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        num_layers = cfg.gnn.layers_mp if num_layers is None else num_layers
        for _ in range(num_layers):
            conv = self._make_conv(dim_inner)
            if reverse_mp:
                # Separate weights per edge direction, averaged per node like to_hetero(..., aggr="mean") in MEGA-GNN
                conv = HeteroConv({EDGE: conv, REV_EDGE: self._make_conv(dim_inner)}, aggr="mean")
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(dim_inner))

    @staticmethod
    def _make_conv(dim_inner):
        return ResGatedGraphConv(dim_inner, dim_inner, edge_dim=dim_inner, act=act_dict[cfg.gnn.act]())

    def encode(self, x, edge_index, edge_attr):
        """Computes node and edge embeddings.

        With reverse message passing, edge_index and edge_attr are dictionaries keyed by edge type (EDGE, REV_EDGE)
        and the returned edge embeddings are a dictionary too.
        """
        x = self.node_emb(x)
        if self.reverse_mp:
            edge_attr = {EDGE: self.edge_emb(edge_attr[EDGE]), REV_EDGE: self.edge_emb_rev(edge_attr[REV_EDGE])}
        else:
            edge_attr = self.edge_emb(edge_attr)

        for i in range(len(self.convs)):
            if self.reverse_mp:
                n_new = self.convs[i]({"node": x}, edge_index, edge_attr_dict=edge_attr)["node"]
            else:
                n_new = self.convs[i](x, edge_index, edge_attr)
            x = (x + F.relu(self.batch_norms[i](n_new))) / 2

        return x, edge_attr

    def forward(self, data):
        if isinstance(data, HeteroData) and self.reverse_mp:
            x, edge_attr = self.encode(data["node"].x, data.edge_index_dict, data.edge_attr_dict)
            data["node"].x = x
            data[EDGE].edge_attr, data[REV_EDGE].edge_attr = edge_attr[EDGE], edge_attr[REV_EDGE]
        elif isinstance(data, HeteroData):
            data["node"].x, data[EDGE].edge_attr = self.encode(
                data["node"].x, data[EDGE].edge_index, data[EDGE].edge_attr
            )
        else:
            data.x, data.edge_attr = self.encode(data.x, data.edge_index, data.edge_attr)

        return data
