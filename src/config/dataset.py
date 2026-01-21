from torch_geometric.graphgym import register_config


@register_config('extend_dataset_config')
def extend_dataset_config(cfg):
    cfg.dataset.table = None