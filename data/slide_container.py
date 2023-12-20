import openslide
import cv2
from shapely import geometry
from pathlib import Path
import json
import numpy as np
import random
from qt_wsi_reg.registration_tree import RegistrationQuadTree

parameters = {
    # feature extractor parameters
    "point_extractor": "orb",  # orb , sift
    "maxFeatures": 512,
    "crossCheck": False,
    "flann": False,
    "ratio": 0.6,
    "use_gray": False,

    # QTree parameter
    "homography": True,
    "filter_outliner": False,
    "debug": True,
    "target_depth": 1,
    "run_async": True,
    "num_workers": 2
}

class SlideContainer:

    def __init__(self, file_A: Path,
                 file_B: Path,
                 annotation_file,
                 ds_factor: int = 1,
                 patch_size: int = 256,
                 label_dict=None, white: float = 255.):
        with open(annotation_file) as f:
            data = json.load(f)
            self.tissue_classes = dict(zip([cat["name"] for cat in data["categories"]],[cat["id"] for cat in data["categories"]]))
            image_id = [i["id"] for i in data["images"] if i["file_name"] == file_A.name][0]
            self.polygons = [anno for anno in data['annotations'] if anno["image_id"] == image_id]
        self.labels = set([poly["category_id"] for poly in self.polygons])
        self.labels.discard(self.tissue_classes["Bone"])
        self.labels.discard(self.tissue_classes["Cartilage"])
        self.labels.discard(self.tissue_classes["Inflamm/Necrosis"])
        self.probabilities = dict.fromkeys(list(self.labels), np.nan_to_num(np.true_divide(0.45, len(list(self.labels)) - 1), posinf=0))
        self.probabilities.update({0:0.1})
        self.probabilities.update({max(self.labels): 0.45})
        self.qtree = RegistrationQuadTree(file_A, file_B, **parameters)
        self.slide_A = openslide.open_slide(str(file_A))
        self.slide_B = openslide.open_slide(str(file_B))
        self.white = white
        self.x_min, self.y_min = np.min(np.array([np.min(np.array(polygon['segmentation']).reshape((-1, 2)), axis=0) for polygon in self.polygons]),axis=0)
        self.x_max, self.y_max = np.max(np.array([np.max(np.array(polygon['segmentation']).reshape((-1, 2)), axis=0) for polygon in self.polygons]),axis=0)
        self.patch_size = patch_size
        self.down_factor = ds_factor
        self._level = (np.abs(np.array(self.slide_A.level_downsamples) - ds_factor)).argmin()
        self.label_dict = label_dict


    def get_patch(self, x: int = 0, y: int = 0):
        patch = np.array(self.slide_A.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self._level, size=(self.patch_size, self.patch_size)))
        patch[patch[:, :, -1] == 0] = [255, 255, 255, 0]
        return patch[:,:,:3]


    def get_y_patch(self, x: int = 0, y: int = 0):
        y_patch = -1*np.ones(shape=(self.patch_size, self.patch_size), dtype=np.int8)
        inv_map = {v: k for k, v in self.tissue_classes.items()}

        for poly in self.polygons:
            coordinates = np.array(poly['segmentation']).reshape((-1,2))/ self.down_factor
            coordinates = coordinates - (x, y)
            label = self.label_dict[inv_map[poly["category_id"]]]
            cv2.drawContours(y_patch, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        white_mask = cv2.cvtColor(self.get_patch(x,y),cv2.COLOR_RGB2GRAY) > self.white
        excluded = (y_patch == -1)
        y_patch[np.logical_and(white_mask, excluded)] = 0
        return y_patch

    def get_registered_patch(self, x, y):
        box = self.down_factor * np.array([x + self.patch_size // 2, y + self.patch_size // 2, self.patch_size, self.patch_size])
        xc, yc, w, h = self.qtree.transform_boxes([box])[0]
        level_B = (np.abs(np.array(self.slide_B.level_downsamples) - self.down_factor)).argmin()
        xmin, ymin, xmax, ymax = int(xc - w // 2), int(yc - h // 2), int(xc + w // 2), int(yc + h // 2)
        cropped_B = np.array(
            self.slide_B.read_region(location=(xmin, ymin), level=level_B, size=(int(w // self.down_factor), int(h // self.down_factor))))
        cropped_B[cropped_B[:, :, -1] == 0] = [255, 255, 255, 0]
        cropped_B = cropped_B[:,:,:3]
        w, h = int(w // self.down_factor), int(h // self.down_factor)
        if xmin < 0 or ymin < 0 or xmax > self.slide_B.dimensions[0] or ymax > self.slide_B.dimensions[1]:
            raise ValueError("Image tile out of bounds")
        M = self.qtree.get_inv_rotation_matrix
        T_c1 = np.vstack((np.array([[1, 0, -int(w // 2)], [0, 1, -int(h // 2)]]), np.array([0, 0, 1])))
        center_t = M @ [int(w // 2), int(h // 2), 1]
        T_c2 = np.vstack((np.array([[1, 0, abs(center_t[0])], [0, 1, abs(center_t[1])]]), np.array([0, 0, 1])))
        final_M = T_c2 @ (M @ T_c1)
        transformed_B = cv2.warpAffine(cropped_B, final_M[0:2, :], (w, h), borderMode=1)
        transformed_B = cv2.resize(transformed_B, (self.patch_size, self.patch_size))
        return transformed_B


    def get_new_train_coordinates(self):
        # sampling method
        label = random.choices(list(self.probabilities.keys()), list(self.probabilities.values()))[0]
        if label == 0:
            if random.choice([True, False]):
                xmin = random.choice([self.x_min - self.patch_size * self.down_factor, self.x_max]) // self.down_factor
                ymin = random.randint(0, self.slide_A.dimensions[1]) // self.down_factor
            else:
                xmin = random.randint(0, self.slide_A.dimensions[0]) // self.down_factor
                ymin = random.choice([self.y_min - self.patch_size * self.down_factor, self.y_max]) // self.down_factor
        else:
            # default sampling method
            xmin, ymin = 0, 0
            found = False
            while not found:
                iter = 0
                polygon = np.random.choice([poly for poly in self.polygons if poly["category_id"] == label])
                coordinates = np.array(polygon['segmentation']).reshape((-1, 2))
                minx, miny, xrange, yrange = polygon["bbox"]
                while iter < 25 and not found:
                    iter += 1
                    pnt = geometry.Point(np.random.uniform(minx, minx + xrange), np.random.uniform(miny, miny + yrange))
                    if geometry.Polygon(coordinates).contains(pnt):
                        xmin = pnt.x // self.down_factor - self.patch_size / 2
                        ymin = pnt.y // self.down_factor - self.patch_size / 2
                        found = True
        return xmin, ymin


    def __str__(self):
        return str(self.file)
