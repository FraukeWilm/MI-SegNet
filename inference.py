import torch
from tqdm import tqdm
import cv2
import os
import json
import numpy as np
import yaml
import albumentations.augmentations.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torchvision import transforms
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.functional.classification.jaccard import _jaccard_index_reduce
from MI_SegNet import Seg_encoder_LM, Seg_decoder_LM
import argparse 
from omegaconf import DictConfig
from segmentation_models_pytorch import Unet, DeepLabV3Plus
from torch import nn

class Inference:
    def __init__(self, dir_path, image_path, annotation_path, model, config, num_classes):
        self.config = config
        self.model = model
        self.dir_path = dir_path
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.num_classes = num_classes
        self.mode_keys = {'scanner': ["cs2", "nz20", "nz210", "gt450", "p1000"]}
        self.n_domains = len(self.mode_keys['scanner']) 
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda") 
        else:
            self.device = torch.device("cpu")
        self.patch_size = self.config['data']['patch_size']
        self.test_images = os.listdir(self.image_path)
        self.label_dict = {'background': 0, 'non-tumor': 1, 'tumor': 2}
        with open("data/whites.yaml", 'r') as stream:
            self.whites = yaml.safe_load(stream)
        with open(self.annotation_path) as f:
            self.annotation_dict = json.load(f)
        self.cm = [ConfusionMatrix(task='multiclass', num_classes=self.num_classes, ignore_index=-1).to(self.device) for _ in range(self.n_domains)]
        self.transform_image = nn.Identity() #transforms.Normalize(0.5, 0.5)


    def configure_model(self):
        if self.model.__contains__ ('segnet'):
            self.encoder = Seg_encoder_LM(self.config['model']['input_channel'], init_features=64, num_blocks=2).to(self.device)
            self.decoder = Seg_decoder_LM(self.config['model']['output_channel'], init_features=64, num_blocks=2).to(self.device)
        elif self.model.__contains__ ('unet'):
            unet = Unet(encoder_name='resnet34', classes=3)
            self.encoder = unet.encoder.to(self.device)
            self.decoder = unet.decoder.to(self.device) 
            self.segmentation_head = unet.segmentation_head.to(self.device)
        elif self.model == 'densenet':
            densenet = DeepLabV3Plus(encoder_name='resnet34', classes=3)
            self.seg_encoder = densenet.encoder.to(self.device)
            self.seg_decoder = densenet.decoder.to(self.device)
            self.segmentation_head = densenet.segmentation_head.to(self.device)

    def load_model_checkpoint(self):
        encoder_ckpts = {}
        decoder_ckpts = {}
        head_ckpts = {}
        ckpt_name = list(filter(lambda file: file.endswith('.ckpt'), os.listdir(os.path.join(self.dir_path, 'files'))))[0]
        ckpt_temp = torch.load(os.path.join(self.dir_path,'files', ckpt_name), map_location=self.device)['state_dict']
        for (key, value) in ckpt_temp.items():
            if key.startswith('seg_encoder'):
                encoder_ckpts[key.split("seg_encoder.")[-1]] = value
            elif key.startswith('seg_decoder'):
                decoder_ckpts[key.split("seg_decoder.")[-1]] = value
            elif key.startswith('segmentation_head'):
                head_ckpts[key.split("segmentation_head.")[-1]] = value
        self.encoder.load_state_dict(encoder_ckpts)
        self.encoder.eval()
        self.decoder.load_state_dict(decoder_ckpts)
        self.decoder.eval()
        if not self.model.__contains__ ('segnet'):
            self.segmentation_head.load_state_dict(head_ckpts)
            self.segmentation_head.eval()

    def process(self):
        # start extracting patches
        with torch.inference_mode():
            for sample in tqdm(self.test_images):
                tensors, gt = self.get_batch(sample)
                tensors = [self.transform_image(t) for t in tensors]
                tensors = self.chunk(tensors, n_chunks=gt.shape[1]//self.patch_size)
                gt = self.chunk(gt.unsqueeze(1), n_chunks=gt.shape[1]//self.patch_size)
                input_batch = torch.stack(tensors).to(self.device)
                features = self.encoder(input_batch)
                if not self.model.__contains__ ('segnet'):
                    outputs = self.decoder(*features)
                    outputs = self.segmentation_head(outputs)
                else:
                    outputs = self.decoder(features)
                outputs = torch.max(outputs, dim=1)[1]
                self.update_cm(outputs, gt)

    def get_batch(self, sample):
        patches = cv2.imread(os.path.join(self.image_path, sample))
        patches = cv2.cvtColor(patches, cv2.COLOR_BGR2RGB)
        patches[(patches == [0, 0, 0]).all(axis=-1)] = [255, 255, 255]
        tensors = [ToTensorV2().apply(patches[:, i * patches.shape[0]:(i + 1) * patches.shape[0], :]) / 255. for i in range(self.n_domains)]
        gt = torch.stack([ToTensorV2().apply(patch) for patch in
                          self.get_y_patch(sample, patches, self.annotation_dict,
                                           whites=self.whites[sample[:6]])]).squeeze(1)
        return tensors, gt

    def get_y_patch(self, sample_name, patches, annotation_dict, down_factor=4, whites={}):
        whites = [whites[s] for s in self.mode_keys['scanner']]
        x, y = sample_name[:-4].split("_")[2:]
        image_id = [i["id"] for i in annotation_dict["images"] if i["file_name"] == sample_name[:6] + "_cs2.svs"][0]
        polygons = [anno for anno in annotation_dict['annotations'] if anno["image_id"] == image_id]
        y_patch = -1 * np.ones(shape=(self.n_domains, patches.shape[0], patches.shape[0]), dtype=np.int8)

        for poly in polygons:
            coordinates = np.array(poly['segmentation']).reshape((-1, 2))
            coordinates = (coordinates - (int(x) * 100, int(y) * 100)) / down_factor
            label = 1 if poly["category_id"] < 7 else 2
            for i in range(self.n_domains):
                cv2.drawContours(y_patch[i], [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)
        white_mask = np.array(
            [cv2.cvtColor(patches[:, i * patches.shape[0]:(i + 1) * patches.shape[0], :], cv2.COLOR_RGB2GRAY) > whites[i] for i in
             range(self.n_domains)])
        excluded = (y_patch == -1)
        y_patch[np.logical_and(white_mask, excluded)] = 0
        return y_patch
    
    def chunk(self, tensors, n_chunks=2):
        x_chunks = [torch.chunk(t, n_chunks, dim=1) for t in tensors]
        x_chunks = [ch for chunk in x_chunks for ch in chunk]
        y_chunks = [torch.chunk(t, n_chunks, dim=2) for t in x_chunks]
        y_chunks = [ch for chunk in y_chunks for ch in chunk]
        return y_chunks

    
    def update_cm(self, outputs, gt):
        for o in range(outputs.shape[0]):
            idx = o//(outputs.shape[0]//self.n_domains)
            self.cm[idx].update(outputs[o], gt[o].squeeze().to(self.device))

    def get_ious(self):
        ious = [_jaccard_index_reduce(mat.compute(), average='none').cpu().numpy() for mat in self.cm]
        ious = {str(self.mode_keys['scanner'][idx]): {**{key: str(iou[value]) for key, value in self.label_dict.items()}, **{'mIoU': str(iou.mean())}}
                  for idx, iou in enumerate(ious)}
        
        with open(os.path.join(self.dir_path, 'files',"iou.json"),'w') as f:
            json.dump(ious, f)

    def run(self):
        self.configure_model()
        self.load_model_checkpoint()
        self.process()
        self.get_ious()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", help="Define experiment directory.")
    parser.add_argument("--datadir", help="Set data dir.")
    parser.add_argument("--annotation_path", help="Set annotation path.")
    args = parser.parse_args()
    runs = filter(lambda file: file.__contains__('fold'), os.listdir(args.experiment_dir))
    for run in list(runs):
        with open(os.path.join(args.experiment_dir, run, 'files', "config.yaml"), 'r') as stream:
            config = eval(yaml.safe_load(stream)['cfg']['value'])
        with open(os.path.join(args.experiment_dir, run, 'files', "wandb-metadata.json"), 'r') as stream:
            wandb_config = json.load(stream)
        print("Evaluating", run)
        model = wandb_config['args'][5]
        inference_module = Inference(dir_path = os.path.join(args.experiment_dir, run), image_path=args.datadir, annotation_path=args.annotation_path, model=model, config=config, num_classes=3)
        inference_module.run()


