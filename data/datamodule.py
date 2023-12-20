import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
import glob
from tqdm import tqdm
import albumentations as A
from data.dataset import ScannerDataset
from data.augmentation import album_transforms
from albumentations.pytorch import ToTensorV2
import numpy as np

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self._cfg = cfg
        # location based parameters
        self._data_dir = cfg.files.image_path
        self._anno_path = cfg.files.annotation_file
        self._n_workers = cfg.cluster.n_workers
        # basic data parameters
        self._batch_size = cfg.data.batch_size
        self._ds_factor = cfg.data.ds_factor
        self._patch_size = cfg.data.patch_size
        # sample number for this dataloader
        self._samples = int(cfg.data.patches_per_slide)
        # task specific parameters
        self._sampler = None
        self._sampler_probs = None
        self._scanner_A = cfg.training.scanner_A
        self._scanner_B = cfg.training.scanner_B
        self.aug_train, self.aug_unmod = None, None
        self.setup_augmentations()

        self._label_dict = {'Bg': 0, 'Bone': 1, 'Cartilage': 1, 'Dermis': 1, 'Epidermis': 1, 'Subcutis': 1,
                            'Inflamm/Necrosis': 1, 'Melanoma': 2, 'Plasmacytoma': 2, 'Mast Cell Tumor': 2, 'PNST': 2,
                            'SCC': 2, 'Trichoblastoma': 2, 'Histiocytoma': 2}
        self.valid_idxs = []

    def setup_augmentations(self):
        # setup transforms
        aug = {
            "randcrop": self._patch_size,
            "scale": [0.8, 1.2],
            "flip": True,
            "jitter_d": 1.0,
            "jitter_p": 0.0,
            "grayscale": False,
            "gaussian_blur": 0.0,
            "rotation": True,
            "contrast": 0.0,
            "contrast_p": 0.0,
            "grid_distort": 0.0,
            "grid_shuffle": 0.0,
        }

        self.aug_train = A.Compose(album_transforms(eval=False, aug=aug), additional_targets={'image_2': 'image', 'image_3': 'image', 'image_4': 'image', 'mask_2': 'mask'})
        self.aug_unmod = A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()], additional_targets={'image_2': 'image', 'image_3': 'image', 'image_4': 'image', 'mask_2': 'mask'})

    def setup(self, stage):
        files = []
        suffixes = {"cs2": ".svs", "nz20": ".ndpi", "nz210": ".ndpi", "p1000": ".mrxs", "gt450": ".svs"}
        slides = pd.read_csv('data/scc_dataset.csv', delimiter=";")
        idx = 0
        for index, row in tqdm(slides.iterrows()):
            image_file_A = Path(glob.glob("{}/**/{}_{}{}".format(str(self._data_dir), row["Slide"], self._scanner_A, suffixes[self._scanner_A]),recursive=True)[0])
            image_file_B = Path(glob.glob("{}/**/{}_{}{}".format(str(self._data_dir), row["Slide"], self._scanner_B, suffixes[self._scanner_B]),recursive=True)[0])

            if row["Dataset"] == "val":
                self.valid_idxs.append(idx)
            if row["Dataset"] != "test":
                files.append((image_file_A, image_file_B))
                idx += 1
            else:
                continue

        # create datasets
        train_files = [files[i] for i in range(len(files)) if i not in self.valid_idxs]
        valid_files = [files[i] for i in range(len(files)) if i in self.valid_idxs]
        self._ds_train = ScannerDataset(train_files, self._anno_path, self._samples, self._patch_size, self._ds_factor, self._label_dict, self.aug_train)
        self._ds_val = ScannerDataset(valid_files, self._anno_path, self._samples, self._patch_size, self._ds_factor, self._label_dict, self.aug_unmod)


    def prepare_data(self):
        # This method is called once to prepare the dataset
        pass

    def train_dataloader(self):
        persistent_workers = True if self._n_workers > 0 else False
        return DataLoader(self._ds_train, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True,
                          prefetch_factor=2, shuffle=True, persistent_workers=persistent_workers)

    def val_dataloader(self):
        persistent_workers = True if self._n_workers > 0 else False
        return DataLoader(self._ds_val, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True,
                          prefetch_factor=2, shuffle=False, persistent_workers=persistent_workers)

    def train_dataset(self):
        return self._ds_train

    def val_dataset(self):
        return self._ds_val

    def get_vis_img(self):
        idx = np.random.randint(0, len(self._ds_val))
        return self._ds_val[idx]