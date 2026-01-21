from torch_geometric.graphgym.register import register_config


@register_config("extend_general_config")
def extend_general_config(cfg):
    cfg.save_model = False
    cfg.root_dir = ""
