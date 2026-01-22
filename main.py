import logging

import torch
import wandb
from torch_geometric.utils import degree

import src  # for custom configs

from torch_geometric.graphgym import (
    parse_args,
    cfg,
    set_cfg,
    load_cfg,
    dump_cfg,
    create_model,
    create_scheduler,
    auto_select_device,
)
from torch_geometric.graphgym.loader import load_dataset
from torch_geometric.graphgym.register import train_dict, dataset_dict

from src.data.aml_data import AMLData
from src.util import set_seed, get_optimizer

import os
import sys


def logger_setup(log_dir: str):
    """Setup logging to file and stdout"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "logs.log")), logging.StreamHandler(sys.stdout)],
    )


def run_loop_settings(cfg, args):
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    num_iterations = args.repeat
    seeds = [cfg.seed + x for x in range(num_iterations)]
    run_ids = seeds
    return run_ids, seeds


def get_deg_data(dataset: AMLData):
    d = degree(dataset[0].edge_index[1], num_nodes=len(dataset[0].x), dtype=torch.long)
    cfg.gnn.pna_deg = torch.bincount(d, minlength=1).tolist()


def main():
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    dump_cfg(cfg)

    logger_setup(os.path.join(cfg.out_dir, "logs"))

    for run_id, seed in zip(*run_loop_settings(cfg, args)):
        cfg.seed = seed
        cfg.run_id = run_id

        set_seed(cfg.seed)
        auto_select_device()

        dataset = load_dataset()

        get_deg_data(dataset)

        model = create_model()
        optim = get_optimizer(cfg.optim.optimizer, model)
        scheduler = create_scheduler(optim, cfg.optim)

        wandb.init(config=cfg)

        train_dict[cfg.train.mode](dataset, model, optim, scheduler)

        wandb.finish()


if __name__ == "__main__":
    main()
