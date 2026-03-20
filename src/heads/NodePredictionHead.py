import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head("NodePredictionHead")
class NodePredictionHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NodePredictionHead, self).__init__()

        self.final_dropout = cfg.gnn.dropout

        dim_in = cfg.gnn.dim_inner
        dim_out = cfg.gnn.dim_out

        self.mlp = nn.Sequential(
            Linear(dim_in, 50),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            Linear(25, dim_out),
        )

    def forward(self, data):
        x, edge_attr = data.x, data.edge_attr

        return self.mlp(x)
