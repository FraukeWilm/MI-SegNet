import torch
from data.slide_container import SlideContainer
import yaml
from torch.utils.data import Dataset


class ScannerDataset(torch.utils.data.Dataset):
    def __init__(self, img_l, anno_file_path, num_per_slide, patch_size, ds_level, label_dict, transforms=None):
        self._img_l = img_l
        self._anno_file_path = anno_file_path
        self.num_per_slide = int(num_per_slide)
        self._samples = int(num_per_slide)*len(img_l)
        self._patch_size = patch_size
        self._label_dict = label_dict
        self._ds_level = ds_level
        with open("data/whites.yaml", 'r') as stream:
            self._whites = yaml.safe_load(stream)
        self._class_num = max(label_dict.values()) + 1
        self._transforms = transforms
        self.slide_objs = {}
        self.anno_objs = {}

    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0

    def __getitem__(self, idx):
        # sample wsi
        wsi_idx = idx//self.num_per_slide
        slide_path_A, slide_path_B = self._img_l[wsi_idx]
        # create openslide object
        if self.slide_objs.get(slide_path_A.name) is None:
            white = self._whites[slide_path_A.stem[:6]][slide_path_A.stem[7:]]
            self.slide_objs[slide_path_A.name] = SlideContainer(slide_path_A, slide_path_B, self._anno_file_path, self._ds_level, self._patch_size, self._label_dict, white)
        slide_obj = self.slide_objs[slide_path_A.name]
        # while True needed because of JPEG error on Leica slides
        while True:
            try:
                x1, y1 = slide_obj.get_new_train_coordinates()
                x2, y2 = slide_obj.get_new_train_coordinates()
                patch1_A = slide_obj.get_patch(x1, y1)
                patch2_A = slide_obj.get_patch(x2, y2)
                patch1_B = slide_obj.get_registered_patch(x1, y1)
                patch2_B = slide_obj.get_registered_patch(x2, y2)
                y_patch1 = slide_obj.get_y_patch(x1, y1)
                y_patch2 = slide_obj.get_y_patch(x2, y2)
                break
            except Exception as e:
                print(e)
                white = self._whites[slide_path_A.stem[:6]][slide_path_A.stem[7:]]
                self.slide_objs[slide_path_A.name] = SlideContainer(slide_path_A, slide_path_B, self._anno_file_path, self._ds_level,
                                                                  self._patch_size, self._label_dict,white)
                slide_obj = self.slide_objs[slide_path_A.name]
        transformed = self._transforms(image=patch1_A, image_2=patch2_A, image_3=patch1_B, image_4= patch2_B, mask=y_patch1, mask_2=y_patch2)
        return transformed['image'], transformed['image_2'], transformed['image_3'], transformed['image_4'],transformed['mask'], transformed['mask_2']