import logging

from torch_geometric.graphgym import parse_args, cfg, set_cfg, load_cfg, dump_cfg, create_model, create_scheduler, \
    auto_select_device
from torch_geometric.graphgym.register import train_dict

from dataloader import get_loaders, get_dataset, AMLData
from src.util import set_seed, get_optimizer

import random
import os
import sys


def logger_setup(log_dir: str):
    """Setup logging to file and stdout"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "logs.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    dump_cfg(cfg)

    auto_select_device()

    if args.seed == 1:
        args.seed = random.randint(2, 256000)
    set_seed(args.seed)

    model = create_model(dim_in=cfg.dim_in, dim_out=cfg.dim_out)
    optim = get_optimizer(cfg.optimizer, model)
    loaders = get_loaders()
    dataset = get_dataset()
    scheduler = create_scheduler(optim, cfg.optim)

    extra_dataset_info = {}
    if isinstance(dataset, AMLData):
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = dataset.get_data()
        extra_dataset_info['tr_data'] = tr_data
        extra_dataset_info['val_data'] = val_data
        extra_dataset_info['te_data'] = te_data
        extra_dataset_info['tr_inds'] = tr_inds
        extra_dataset_info['val_inds']= val_inds
        extra_dataset_info['te_inds']= te_inds

    logging.info(f"Running Training")
    train_dict[cfg.train.mode](loaders, model, optim, scheduler, **extra_dataset_info)


if __name__ == "__main__":
    main()
