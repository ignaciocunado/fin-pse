from torch_geometric.graphgym import register_config
from yacs.config import CfgNode

@register_config("add_ssl_config")
def add_ssl_config(cfg):
    cfg.ssl = CfgNode()

    cfg.ssl.convert_currencies = False
    cfg.ssl.windowed_features = False
