import torch.nn.functional as F

from torch_geometric.graphgym import register_loss

register_loss('l2', F.mse_loss)