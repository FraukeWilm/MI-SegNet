import torch
from tqdm import tqdm
import cv2
import os
import json
import numpy as np
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.functional.classification.jaccard import _jaccard_index_reduce
from MI_SegNet import Seg_encoder_LM, Seg_decoder_LM
import argparse 
from segmentation_models_pytorch import Unet, DeepLabV3Plus
import matplotlib.pyplot as plt
import openslide
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import shutil
from PIL import Image

class Inference:
    def __init__(self, dir_path, image_path, annotation_path, config, num_classes):
        self.config = config
        self.model = config.model.name
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
        self.patch_size = self.config.data.patch_size
        self.down_factor = self.config.data.ds_factor
        self.label_dict = {'background': 0, 'non-tumor': 1, 'tumor': 2}
        with open("data/whites.yaml", 'r') as stream:
            self.whites = yaml.safe_load(stream)
        with open(self.annotation_path) as f:
            self.annotation_dict = json.load(f)
        self.cm = [ConfusionMatrix(task='multiclass', num_classes=self.num_classes, ignore_index=-1).to(self.device) for _ in range(self.n_domains)]

    def configure_model(self):
        if self.model.__contains__ ('segnet'):
            self.encoder = Seg_encoder_LM(self.config.model.input_channel, init_features=64, num_blocks=2).to(self.device)
            self.decoder = Seg_decoder_LM(self.config.model.output_channel, init_features=64, num_blocks=2).to(self.device)
        elif self.model.__contains__ ('unet'):
            unet = Unet(encoder_name='resnet34', classes=3)
            self.encoder = unet.encoder.to(self.device)
            self.decoder = unet.decoder.to(self.device) 
            self.segmentation_head = unet.segmentation_head.to(self.device)
        elif self.model == 'densenet':
            densenet = DeepLabV3Plus(encoder_name='resnet34', classes=3)
            self.encoder = densenet.encoder.to(self.device)
            self.decoder = densenet.decoder.to(self.device)
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

    def get_active_map(self, slide, xsteps, ysteps):
        downsamples_int = [int(x) for x in slide.level_downsamples]
        ds = 32 if 32 in downsamples_int else 16
        level = np.where(np.abs(np.array(slide.level_downsamples)-ds)<0.1)[0][0]
        overview = slide.read_region(level=level, location=(0,0), size=slide.level_dimensions[level])
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(overview)[:,:,0:3],cv2.COLOR_BGR2GRAY)
        # OTSU thresholding
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # dilate
        dil = cv2.dilate(thresh, kernel = np.ones((7,7),np.uint8))
        # erode
        activeMap = cv2.erode(dil, kernel = np.ones((7,7),np.uint8))
        overlay = np.zeros(np.array(overview).shape, np.uint8)[:,:,0:3]
        overlay[:,:,0] = activeMap
        factor = ds/self.down_factor
        step_ds = int(np.ceil(float(self.patch_size)/factor))
        coordlist = []
        for y in ysteps:
            for x in xsteps:
                x_ds = int(np.floor(float(x)/factor))
                y_ds = int(np.floor(float(y)/factor))
                needCalculation = np.sum(activeMap[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds])>0.9*step_ds*step_ds
                if (needCalculation):
                    coordlist.append([x,y])
        return coordlist

    
    def process(self):
        # start extracting patches
        with torch.inference_mode():
            for idx, scanner in enumerate(self.mode_keys['scanner']):
                wsis = [file for file in os.listdir(os.path.join(self.image_path, scanner)) if not os.path.isdir(file)]
                for wsi in tqdm(wsis):
                    os.mkdir(os.path.join(self.dir_path, "temp"))
                    slide = openslide.open_slide(os.path.join(self.image_path, scanner, wsi))
                    level = (np.abs(np.array(slide.level_downsamples) - self.down_factor)).argmin()
                    shape = slide.level_dimensions[level]
                    x_indices = np.arange(0, int((shape[0] // (self.patch_size//2)) + 1)) * (self.patch_size//2)
                    y_indices = np.arange(0, int((shape[1] // (self.patch_size//2)) + 1)) * (self.patch_size//2)
                    coordlist = self.get_active_map(slide, x_indices, y_indices)
                    dataloader = DataLoader(coordlist, batch_size=32)
                    for xs, ys in tqdm(dataloader,desc='Processing %s' % wsi):
                        patches = [self.get_patch(slide, level, xs[i], ys[i]) for i in range(len(xs))]
                        gt = torch.stack([torch.Tensor(self.get_y_patch(wsi, patches[j], xs[j], ys[j])) for j in range(len(xs))])
                        tensors = [torch.Tensor(patch/255.).permute((2,0,1)) for patch in patches]
                        input_batch = torch.stack(tensors).to(self.device)
                        features = self.encoder(input_batch)
                        if not self.model.__contains__ ('segnet'):
                            outputs = self.decoder(*features)
                            outputs = self.segmentation_head(outputs)
                        else:
                            outputs = self.decoder(features)
                        outputs = torch.max(outputs, dim=1)[1]
                        start, stop = int(self.patch_size*(1/4)), int(self.patch_size*(3/4))
                        self.cm[idx].update(outputs[:, start:stop, start:stop], gt[:, start:stop, start:stop].squeeze().to(self.device))
                        for o in range(outputs.shape[0]):
                            plt.imsave(os.path.join(self.dir_path, "temp", "{}_{}_{}_{}.png".format(wsi.split(".")[0], xs[o], ys[o], scanner)),outputs[o][start:stop, start:stop].cpu().numpy(), vmin=0, vmax=2)
                    self.stitch_output_mask(wsi.split(".")[0])

    def stitch_output_mask(self, filename):
        tile_size = (32, 32)
        max_x, max_y = 0, 0
        tile_paths = []
        for file in os.listdir(os.path.join(self.dir_path, 'temp')):
            parts = file.split("_")
            x, y = parts[-3:-1]
            tile_paths.append((os.path.join(self.dir_path, 'temp', file), int(x), int(y)))
            max_x = max(max_x, int(x))
            max_y = max(max_y, int(y))

        new_size = (max_x // 4, max_y // 4)
        # Create an output image
        output = Image.new('RGB', new_size)

        for path, x, y in tile_paths:
            tile = Image.open(path)
            tile = tile.resize(tile_size)
            output.paste(tile, (x // 4, y // 4))
        output.save(os.path.join(self.dir_path, "{}.png".format(filename)))
        shutil.rmtree(os.path.join(self.dir_path, 'temp'))
    
    def get_patch(self, slide, level, x, y):
        patch = np.array(slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=level, size=(self.patch_size, self.patch_size)))
        patch[patch[:, :, -1] == 0] = [255, 255, 255, 0]
        return patch[:,:,:3]

    def get_y_patch(self, filename, patch, x, y):
        image_id = [i["id"] for i in self.annotation_dict["images"] if i["file_name"] == filename[:6] + "_cs2.svs"][0]
        polygons = [anno for anno in self.annotation_dict['annotations'] if anno["image_id"] == image_id]
        y_patch = -1*np.ones(shape=(self.patch_size, self.patch_size), dtype=np.int8)

        for poly in polygons:
            coordinates = np.array(poly['segmentation']).reshape((-1,2))/ self.down_factor
            coordinates = coordinates - (x, y)
            label = 1 if poly["category_id"] < 7 else 2
            cv2.drawContours(y_patch, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)
        white = self.whites[filename[:6]][filename.split('.')[0][7:]]
        white_mask = cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY) > white
        excluded = (y_patch == -1)
        y_patch[np.logical_and(white_mask, excluded)] = 0
        return y_patch


    def get_ious(self):
        ious = [_jaccard_index_reduce(mat.compute(), average='none').cpu().numpy() for mat in self.cm]
        ious = {str(self.mode_keys['scanner'][idx]): {**{key: str(iou[value]) for key, value in self.label_dict.items()}, **{'mIoU': str(iou.mean())}}
                  for idx, iou in enumerate(ious)}
        
        with open(os.path.join(self.dir_path, 'files',"iou_wsi.json"),'w') as f:
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
            config = DictConfig(eval(yaml.safe_load(stream)['cfg']['value']))
        with open(os.path.join(args.experiment_dir, run, 'files', "wandb-metadata.json"), 'r') as stream:
            wandb_config = json.load(stream)
        print("Evaluating", run)
        config['model']['name'] = wandb_config['args'][5]
        inference_module = Inference(dir_path = os.path.join(args.experiment_dir, run), image_path=args.datadir, annotation_path=args.annotation_path, config=config, num_classes=3)
        inference_module.run()


