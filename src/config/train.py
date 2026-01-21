from torch_geometric.graphgym.register import register_config


@register_config('extend_train_config')
def extend_train(cfg):
    cfg.train.mode = 'aml_train'
    cfg.train.num_neighs = [100, 100]