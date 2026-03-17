from torch import nn
from torch_geometric.graphgym import cfg


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, final_act = None):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            cfg.act_dict[cfg.gnn.act],
        )

        for i in range(num_layers - 2):
            self.mlp.append(nn.Linear(dim_hidden, dim_hidden))
            self.mlp.append(cfg.act_dict[cfg.gnn.act]())

        self.mlp.append(nn.Linear(dim_hidden, dim_out))
        if final_act:
            self.mlp.append(final_act)



    def forward(self, data):
        return self.mlp(data)