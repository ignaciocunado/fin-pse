import logging

import random

import os.path as osp

import torch
from torch_geometric.graphgym.register import loss_dict
from torch_geometric.loader import NeighborLoader

import wandb
from torch.optim import Optimizer
from torch_geometric.config_store import LRScheduler
from torch_geometric.graphgym import register_train, cfg
from tqdm import tqdm

from src.data.ssl_data import AMLSSL
from src.util import save_model


def get_loaders(data):
    # if 'cumulative' in cfg.dataset.nodes or not cfg.ssl.windowed_features:
    return NeighborLoader(data, num_neighbors=cfg.train.num_neighs,batch_size=cfg.train.batch_size,shuffle=True)
    # TODO: Check which loader to use
    return data

def pretrain(
        dataset,
        model,
        optimizer,
        loss_fn,
):
    idxs = list(range(len(dataset)))

    step = 0
    for epoch in tqdm(range(1, cfg.optim.max_epoch + 1)):
        random.shuffle(idxs)
        model.train()

        epoch_loss = 0.0
        epoch_count = 0

        for i in idxs:
            data = dataset[i]

            loader = get_loaders(data)

            if loader.__class__ == NeighborLoader:
                for batch in loader:
                    batch.to(cfg.accelerator)
                    pred = model(batch)
                    loss = loss_fn(pred, batch.y)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    optimizer.step()

                    epoch_loss += float(loss.item())
            else:
                data.to(cfg.accelerator)
                pred = model(data)
                loss = loss_fn(pred, data.y_ego, data.y_ego_mask)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                optimizer.step()

                epoch_loss += float(loss.item())

            epoch_count += 1
            step += 1

        logging.info(f"Epoch {epoch:03d} | avg_loss={epoch_loss / max(1, epoch_count):.6f}")
        wandb.log({
            'epoch': epoch,
            'loss': epoch_loss / max(1, epoch_count)
        })

    if cfg.save_model:
        table = cfg.dataset.nodes.replace('.csv', '').replace('_Nodes', '')
        filename = f"{table}_epoch_101.tar"

        save_model(model.encoder, optimizer, 100)
        artifact = wandb.Artifact('model_checkpoint', type='model')
        artifact.add_file(osp.join(cfg.checkpoint_dir,filename))
        wandb.log_artifact(artifact)

    return model

@register_train("ssl")
def pretrain_model(dataset: AMLSSL, model: torch.nn.Module, optimizer: Optimizer, scheduler: LRScheduler):
    model = pretrain(
        dataset,
        model,
        optimizer,
        loss_dict[cfg.model.loss_fun],
    )

    return model

