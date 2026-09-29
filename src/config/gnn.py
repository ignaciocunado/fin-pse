from torch_geometric.graphgym import register_config


@register_config("extend_gnn_config")
def extend_gnn_config(cfg):
    # Dimensions
    cfg.gnn.dim_in = 1
    cfg.gnn.dim_out = 2
    cfg.gnn.edge_dim = 4  # Raw edge features, excluding port numbers (added automatically when dataset.ports)

    cfg.gnn.emlps = False  # To use edge updates via MLP
    cfg.gnn.pna_deg = None

    # Reverse message passing (MEGA-GNN / Multi-GNN): graphs become heterogeneous with a ('node', 'to', 'node')
    # and a flipped ('node', 'rev_to', 'node') edge type, so nodes also aggregate over their outgoing edges.
    # Applies to the model being trained (FinPSE when pretraining, MPNN downstream)
    cfg.gnn.reverse_mp = False

    # Ego IDs: extra node feature flagging the seed nodes of each batch (downstream training only)
    cfg.gnn.ego = False

    cfg.gnn.add_encodings = False
    cfg.gnn.encodings_file = ""
    cfg.gnn.encodings_random_feats = False
    # Architecture of the pretrained encoder that is loaded from encodings_file
    cfg.gnn.encodings_reverse_mp = False  # Encoder was pretrained with reverse message passing
    cfg.gnn.encodings_ports = False  # Encoder was pretrained with port numbers as extra edge features
    cfg.gnn.encodings_layers = -1  # Number of encoder layers, -1 to use gnn.layers_mp
    cfg.gnn.encodings_dim = 64  # Hidden dimension of the encoder
    cfg.gnn.encodings_edge_dim = 2  # Leading edge features the encoder takes (Timestamp, Amount)
