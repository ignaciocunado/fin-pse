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

    # Precomputed node encodings (GPSE-style): the frozen encoder runs once per split graph with its own inputs
    # (daily-USD amounts, days, pretraining normalisation) and sampling, every account as a seed. The baseline model,
    # its features and batches stay unchanged; the encodings are concatenated to its node embeddings.
    cfg.gnn.encodings_precompute = False
    # What is added to the baseline: pretrained (the encoder in encodings_file), random_init (same architecture,
    # untrained weights) or exact (the pretraining targets computed exactly on each split graph, etl/aml/targets.py)
    cfg.gnn.encodings_source = "pretrained"
    cfg.gnn.encodings_draws = 4  # Random node feature draws averaged per account
    cfg.gnn.encodings_batch_size = 512  # Seed accounts per batch when precomputing
    cfg.gnn.exact_target_groups = ["ego", "temporal", "flow", "motifs"]  # Targets used by encodings_source=exact
    cfg.gnn.exact_gfp = True  # Include the GFP motif targets (needs snapml) in encodings_source=exact
