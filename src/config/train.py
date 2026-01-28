from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode


@register_config("extend_train_config")
def extend_train(cfg):
    cfg.train.mode = "aml_train"
    cfg.train.num_neighs = [100, 100]

    cfg.train.ssl = CfgNode()
    cfg.train.ssl.edge_drop_p = 0.2
    cfg.train.ssl.edge_attr_noise_std = 0.0
