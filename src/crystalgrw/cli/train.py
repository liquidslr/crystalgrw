import time
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import os
import json
import copy
import argparse
import random

from ..common.model_utils import get_model


def run_train(cfg):
    if cfg.train.deterministic:
        torch.manual_seed(cfg.train.random_seed)
        torch.cuda.manual_seed(cfg.train.random_seed)
        torch.cuda.manual_seed_all(cfg.train.random_seed)
        np.random.seed(cfg.train.random_seed)
        random.seed(cfg.train.random_seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    model = get_model(cfg)
    model.init()
    model.train_start()
    print(model)
    print('\nModel parameters:')
    print(f'{round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)}M')

    for e in range(model.current_epoch, cfg.train.max_epochs):
        tick = time.time()
        model.train()

        model.train_epoch_start(e)
        for batch_idx, batch in enumerate(model.train_dataloader):
            loss = model.training_step(batch.to(model.device), batch_idx)
            model.optimizer.zero_grad()
            loss.backward()
            model.clip_grad_value_()
            model.optimizer.step()
            if cfg.optim.lr_scheduler._target_ != "ReduceLROnPlateau":
                model.scheduler.step()
            model.train_step_end(e)

        model.train_epoch_end(e)

        if e % cfg.logging.check_val_every_n_epoch == 0:

            model.eval()

            model.val_epoch_start(e)

            with torch.no_grad():
                outs = []
                for val_batch_idx, val_batch in enumerate(model.val_dataloader):
                    val_out = model.validation_step(val_batch.to(model.device), val_batch_idx)
                    outs.append(val_out.detach())
                    model.val_step_end(e)

            model.val_epoch_end(e)

            if cfg.optim.lr_scheduler._target_ == "ReduceLROnPlateau":
                model.scheduler.step(torch.mean(torch.stack([x for x in outs])))

        model.train_val_epoch_end(e)
        print(f"\tTraining time: {time.time() - tick} s")

        if model.early_stopping(e):
            break

    model.train_end(e)
