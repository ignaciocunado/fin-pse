from torch_geometric.graphgym import register_config
from yacs.config import CfgNode


@register_config("add_ssl_config")
def add_ssl_config(cfg):
    cfg.ssl = CfgNode()

    cfg.ssl.convert_currencies = False
    cfg.ssl.windowed_features = False

    # Apply sign(x) * log(1 + |x|) to the pretraining targets before normalisation
    cfg.ssl.log_targets = False
    # Sample new random node features for every batch instead of the fixed ones stored in the dataset
    cfg.ssl.resample_random_feats = False
