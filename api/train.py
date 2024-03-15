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
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from GPT_SoVITS.AR.data.data_module import Text2SemanticDataModule
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.AR.utils.io import load_yaml_config
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")
from GPT_SoVITS.AR.utils import get_newest_ckpt
from GPT_SoVITS.s1_train import my_model_ckpt



# SoVITS训练
def main(project_id):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available() or torch.backends.mps.is_available(), "Only GPU training is allowed."

    if torch.backends.mps.is_available():
        n_gpus = 1
    else:
        n_gpus = torch.cuda.device_count()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    my_exp_dir = f"./work_dir/train/{project_id}"
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

    print(hps)

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )

# GPT训练
def main_GPT(project_id,args):
    config = load_yaml_config(args.config_file)
    config['output_dir'] = f"./work_dir/train/{project_id}/log_s1"
    config['train_semantic_path'] = f"./work_dir/data_process/{project_id}/6-name2semantic.tsv"
    config['train_phoneme_path'] = f"./work_dir/data_process/{project_id}/2-name2text.txt"
    output_dir = Path(f"./work_dir/train/{project_id}/log_s1")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"]="localhost"
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu",
        # val_check_interval=9999999999999999999999,###不要验证
        # check_val_every_n_epoch=None,
        limit_val_batches=0,
        devices=-1,
        benchmark=False,
        fast_dev_run=False,
        strategy = "auto" if torch.backends.mps.is_available() else DDPStrategy(
            process_group_backend="nccl" if platform.system() != "Windows" else "gloo"
        ),  # mps 不支持多节点训练
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        config, output_dir
    )

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
        # dev_semantic_path=args.dev_semantic_path,
        # dev_phoneme_path=args.dev_phoneme_path
    )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


