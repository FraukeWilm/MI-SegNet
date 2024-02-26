from segmentation_models_pytorch import Unet, DeepLabV3Plus
from torchvision.models import resnet18, resnet34
from data.datamodule import BaseDataModule
from MI_SegNet import Seg_encoder_LM
from omegaconf import DictConfig
from torch_cka import CKA
import numpy as np
import argparse
import random
import torch
import json
import yaml
import os
import re


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

class ModelCKA():
    def __init__(self, cfgA, cfgB) -> None:
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda") 
        else:
            self.device = torch.device("cpu")
    
        modelA, layersA = self.configure_model(cfgA)
        modelB, layersB = self.configure_model(cfgB)
        self.load_model_checkpoint(modelA, cfgA)
        self.load_model_checkpoint(modelB, cfgB)
        self.dataloader = self.setup_dataloader(cfgA)
        random_seed(cfgA.training.seed, torch.cuda.is_available())
        nameA = "-".join(re.split('-', re.split('/', cfgA.model.path)[1])[:-2])
        nameB = "-".join(re.split('-', re.split('/', cfgB.model.path)[1])[:-2])
        self.run(modelA, modelB, layersA, layersB, nameA= nameA, nameB = nameB)
        
        
    def configure_model(self, cfg):
        if cfg.model.name.__contains__ ('segnet'):
            encoder = Seg_encoder_LM(cfg.model.input_channel, init_features=64, num_blocks=2)
            layers = list(np.unique(re.findall('encoder.\d.res_layer.\d', str([name for name, _ in encoder.named_modules()]))))
        elif cfg.model.name.__contains__ ('unet'):
            unet = Unet(encoder_name='resnet34', classes=3)
            encoder = unet.encoder
            layers = list(np.unique(re.findall('layer\d.\d', str([name for name, _ in encoder.named_modules()]))))
        elif cfg.model.name == 'densenet':
            densenet = DeepLabV3Plus(encoder_name='resnet34', classes=3)
            encoder = densenet.encoder
            layers = list(np.unique(re.findall('layer\d.\d', str([name for name, _ in encoder.named_modules()]))))
        else:
            encoder, layers = None, []
        return encoder, layers
    
    def load_model_checkpoint(self, encoder, cfg):
        encoder_ckpts = {}
        ckpt_temp = torch.load(cfg.model.path, map_location=self.device)['state_dict']
        for (key, value) in ckpt_temp.items():
            if key.startswith('seg_encoder'):
                encoder_ckpts[key.split("seg_encoder.")[-1]] = value
        encoder.load_state_dict(encoder_ckpts)

    def setup_dataloader(self, cfg):
        data_module = BaseDataModule(cfg)
        data_module.setup(stage = 'test')
        dataloader = data_module.val_dataloader()
        return dataloader

    def run(self, modelA, modelB, layersA, layersB, nameA, nameB):

        cka = CKA(modelA, modelB,
                model1_name=nameA,   # good idea to provide names to avoid confusion
                model2_name=nameB,   
                model1_layers=layersA, # List of layers to extract features from
                model2_layers=layersB, # extracts all layer features by default
                device=self.device)

        cka.compare(self.dataloader) # secondary dataloader is optional
        cka.plot_results(save_path = 'cka_{}_{}.png'.format(nameA, nameB), title = "Centered Kernel Alignment")
        results = cka.export()  # returns a dict that contains model names, layer names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", help="Set data dir.")
    parser.add_argument("--annotation_path", help="Set annotation path.")
    parser.add_argument("--rundirA", help="Define experiment directory A.")
    parser.add_argument("--rundirB", help="Define experiment directory B.")
    args = parser.parse_args()
    with open(os.path.join(args.rundirA, 'files', "config.yaml"), 'r') as stream:
        configA = DictConfig(eval(yaml.safe_load(stream)['cfg']['value']))
    with open(os.path.join(args.rundirA, 'files', "wandb-metadata.json"), 'r') as stream:
        wandb_configA = json.load(stream)
    with open(os.path.join(args.rundirB, 'files', "config.yaml"), 'r') as stream:
        configB = DictConfig(eval(yaml.safe_load(stream)['cfg']['value']))
    with open(os.path.join(args.rundirB, 'files', "wandb-metadata.json"), 'r') as stream:
        wandb_configB = json.load(stream)
    configA['model']['name'] = wandb_configA['args'][5]
    configA['model']['path'] = os.path.join(args.rundirA, 'files', list(filter(lambda file: file.endswith('.ckpt'), os.listdir(os.path.join(args.rundirA, 'files'))))[0])
    configB['model']['name'] = wandb_configB['args'][5]
    configB['model']['path'] = os.path.join(args.rundirB, 'files', list(filter(lambda file: file.endswith('.ckpt'), os.listdir(os.path.join(args.rundirB, 'files'))))[0])
    configA.files.image_path = args.datadir
    configA.files.annotation_file = args.annotation_path
    cka_module = ModelCKA(configA, configB)