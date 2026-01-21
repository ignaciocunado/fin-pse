from torch_geometric.graphgym.register import register_config


@register_config('extend_model_config')
def extend_model_config(cfg):
    # Cross-entropy config
    cfg.model.w_ce1 = 1.0
    cfg.model.w_ce2 = 6.0