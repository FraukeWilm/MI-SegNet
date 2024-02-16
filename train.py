import torch
import os
os.environ["WANDB_MODE"]="offline"
import random
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy
from data.datamodule import BaseDataModule
from module import MISegModule
from module_var_autoencoder import VarModule
from baselines.module_baseline_unet import UnetModule
from baselines.module_baseline_densenet import DensenetModule
from baselines.module_baseline_segnet import SegnetModule
import wandb
import yaml
import argparse 


def random_seed(seed_value, use_cuda):
  '''
  Sets the random seed for numpy, pytorch, python.random and pytorch GPU vars.
  '''
  np.random.seed(seed_value) # Numpy vars
  torch.manual_seed(seed_value) # PyTorch vars
  random.seed(seed_value) # Python
  if use_cuda: # GPU vars
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  print(f'Random state set:{seed_value}, cuda used: {use_cuda}')

def main(args):
    # calculate batch_size
    cfg = DictConfig(yaml.safe_load(open("configs/{}.yaml".format(args.config_name))))
    cfg.data.batch_size = cfg.data.batch_base * cfg.cluster.batch_mul
    if not cfg.files.image_path:
        cfg.files.image_path = args.datadir
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    logger = pl_loggers.WandbLogger(project=cfg.wandb.project)
    data_module = BaseDataModule(cfg)

    match args.network:
        case 'misegnet':
            module = MISegModule(cfg, device)
        case 'variational':
            module = VarModule(cfg, device)
        case 'unet':
            module = UnetModule(cfg, device)
        case 'densenet':
            module = DensenetModule(cfg, device)
        case 'segnet':
            module = SegnetModule(cfg, device)
    random_seed(cfg.training.seed, torch.cuda.is_available())

    metric_checkpoint = ModelCheckpoint(dirpath=logger.experiment.dir, verbose=True, monitor='Source mIoU', mode='max')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = TQDMProgressBar(refresh_rate=int(256 / cfg.data.batch_size))
    callbacks = [lr_monitor, progress_bar, metric_checkpoint]  # more callbacks can be added

    trainer = pl.Trainer(max_epochs=cfg.training.epochs,
                         callbacks=callbacks, logger=logger,
                         accelerator='gpu',
                         #strategy=DDPStrategy(find_unused_parameters=True, process_group_backend=cfg.cluster.backend),
                         devices=cfg.cluster.n_gpus,
                         log_every_n_steps = 20)

    trainer.fit(module, data_module)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Define config file.")
    parser.add_argument("--datadir", help="Set data dir.")
    parser.add_argument("--network", help="Set network.")
    args = parser.parse_args()
    main(args)


