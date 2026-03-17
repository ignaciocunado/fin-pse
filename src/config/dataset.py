from torch_geometric.graphgym import register_config


@register_config("extend_dataset_config")
def extend_dataset_config(cfg):
    cfg.dataset.table = None

    cfg.dataset.nodes = "HI-Medium_SSL_Nodes.csv"
    cfg.dataset.edges = "HI-Medium_SSL_Trans.csv"
