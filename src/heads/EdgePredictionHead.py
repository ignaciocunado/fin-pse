import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('EdgePredictionHead')
class EdgePredictionHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(EdgePredictionHead, self).__init__()

        self.final_dropout = cfg.gnn.dropout

        self.mlp = nn.Sequential(
            Linear(cfg.gnn.dim_inner * 3, 50),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            Linear(25, cfg.gnn.dim_out),
        )

    def forward(self, data):
        x, edge_attr = data.x, data.edge_attr

        x = x[data.edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        return self.mlp(x)