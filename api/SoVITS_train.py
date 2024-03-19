# -*- coding: utf-8 -*-
# coding=utf-8
from random import randint
import os
from GPT_SoVITS.s2_train import run
from GPT_SoVITS.utils import get_hparams
hps = get_hparams(stage=2)
import torch.multiprocessing as mp
from pathlib import Path
import logging
import argparse

import torch, platform
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")


# SoVIT_train
def main(project_id):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available() or torch.backends.mps.is_available(), "Only GPU training is allowed."

    if torch.backends.mps.is_available():
        n_gpus = 1
    else:
        n_gpus = torch.cuda.device_count()


    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    my_exp_dir = f"./work_dir/data_process/{project_id}"
    pretrained_s2G= f"./GPT_SoVITS/pretrained_models/s2G488k.pth"
    pretrained_s2D = f"./GPT_SoVITS/pretrained_models/s2D488k.pth"
    save_every_epoch = 10
    if_save_latest = 0
    if_save_every_weights = True
    save_weight_dir = Path(f"./work_dir/train/{project_id}/logs2_weight")
    save_weight_dir.mkdir(parents=True, exist_ok=True)
    name = project_id
    hps['data']['exp_dir'] = my_exp_dir
    hps['train']['pretrained_s2G'] = pretrained_s2G
    hps['train']['pretrained_s2D'] = pretrained_s2D
    hps['train']['save_every_epoch'] = save_every_epoch
    hps['train']['if_save_latest'] = if_save_latest
    hps['train']['if_save_every_weights'] = if_save_every_weights
    hps['save_weight_dir'] = save_weight_dir
    hps['name']= name

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )