import torch
import hydra
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
from module_vqgan import VQModule, AutoencoderKL
import wandb


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

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig):
    # calculate batch_size
    cfg.data.batch_size = cfg.data.batch_base * cfg.cluster.batch_mul
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = pl_loggers.WandbLogger(project=cfg.wandb.project)
    data_module = BaseDataModule(cfg)
    module = MISegModule(cfg, device)
    #module = VQModule(cfg, device)
    #module = AutoencoderKL(cfg, device)
    random_seed(cfg.training.seed, torch.cuda.is_available())

    metric_checkpoint = ModelCheckpoint(dirpath=logger.experiment.dir, verbose=True, monitor='Val Loss', mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = TQDMProgressBar(refresh_rate=int(256 / cfg.data.batch_size))
    callbacks = [lr_monitor, progress_bar, metric_checkpoint]  # more callbacks can be added

    trainer = pl.Trainer(max_epochs=cfg.training.epochs,
                         callbacks=callbacks, logger=logger,
                         accelerator='gpu',
                         strategy=DDPStrategy(find_unused_parameters=True, process_group_backend=cfg.cluster.backend),
                         devices=cfg.cluster.n_gpus)

    trainer.fit(module, data_module)
    wandb.finish()

if __name__ == "__main__":
    main()



