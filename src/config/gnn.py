from torch_geometric.graphgym import register_config


@register_config("extend_gnn_config")
def extend_gnn_config(cfg):
    # Dimensions
    cfg.gnn.dim_in = 1
    cfg.gnn.dim_out = 2
    cfg.gnn.edge_dim = 4

    cfg.gnn.emlps = False  # To use edge updates via MLP
    cfg.gnn.pna_deg = None
