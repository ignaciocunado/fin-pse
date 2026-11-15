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

    # Fin-PSE pretraining data built by etl/aml/build_ssl.py (one graph per graph_id: whole graph and time crops).
    # Empty keeps the original 12 targets; otherwise any of ego, temporal, flow, motifs (etl/aml/targets.py)
    cfg.ssl.target_groups = []
    cfg.ssl.group_weights = []  # Loss weight per target group, empty for equal weights
    cfg.ssl.val_fraction = 0.1  # Accounts held out per graph to report per-target R^2
    cfg.ssl.max_seeds_per_graph = 0  # Training seed accounts sampled per graph and epoch, 0 for all
    cfg.ssl.time_in_days = True  # Transaction timestamps as days since the start of the graph (fixed scale)
