from albumentations.core.transforms_interface import ImageOnlyTransform
from data.custom_hed_transform import rgb2hed, hed2rgb
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import ImageFilter
import albumentations as A
import numpy as np
import numbers
import random
import torch

class HEAugment(ImageOnlyTransform):
    def __init__(
        self,
        haematoxylin_sigma_range=(-0.05, 0.05),
        haematoxylin_bias_range=(-0.05, 0.05), 
        eosin_sigma_range=(-0.05, 0.05), 
        eosin_bias_range=(-0.05, 0.05), 
        dab_sigma_range=(-0.05, 0.05), 
        dab_bias_range=(-0.05, 0.05), 
        cutoff_range=(0, 1),
        always_apply=False,
        p=0.5,
    ):
        super(HEAugment, self).__init__(always_apply=always_apply, p=p)

        haematoxylin_sigma_range = self.__check_values(haematoxylin_sigma_range, "haematoxylin_sigma")
        haematoxylin_bias_range = self.__check_values(haematoxylin_bias_range, "haematoxylin_bias")
        eosin_sigma_range = self.__check_values(eosin_sigma_range, "eosin_sigma")
        eosin_bias_range = self.__check_values(eosin_bias_range, "eosin_bias")
        dab_sigma_range = self.__check_values(dab_sigma_range, "dab_sigma")
        dab_bias_range = self.__check_values(dab_bias_range, "dab_bias")

        self.__cutoff_range = self.__check_values(cutoff_range, "cutoff", bounds=(0, 1))

        self.__sigma_ranges = [haematoxylin_sigma_range, eosin_sigma_range, dab_sigma_range]
        self.__sigmas = [haematoxylin_sigma_range[0] if haematoxylin_sigma_range is not None else 0.0,
                         eosin_sigma_range[0] if eosin_sigma_range is not None else 0.0,
                         dab_sigma_range[0] if dab_sigma_range is not None else 0.0]
        self.__bias_ranges = [haematoxylin_bias_range, eosin_bias_range, dab_bias_range]
        self.__biases = [haematoxylin_bias_range[0] if haematoxylin_bias_range is not None else 0.0,
                         eosin_bias_range[0] if eosin_bias_range is not None else 0.0,
                         dab_bias_range[0] if dab_bias_range is not None else 0.0]

    @staticmethod
    def __check_values(value, name, bounds=(-1, 1)):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError("{} values should be between {}".format(name, bounds))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        return value
    
    def randomize(self):
        # Randomize sigma and bias for each channel.
        self.__sigmas = [np.random.uniform(low=sigma_range[0], high=sigma_range[1], size=None) if sigma_range is not None else 1.0 for sigma_range in self.__sigma_ranges]
        self.__biases = [np.random.uniform(low=bias_range[0], high=bias_range[1], size=None) if bias_range is not None else 0.0 for bias_range in self.__bias_ranges]

    def apply(self, img,  **params):
        self.randomize()
        # Check if the patch is inside the cutoff values.
        #
        patch_mean = np.mean(a=img) / 255.0
        if self.__cutoff_range[0] <= patch_mean <= self.__cutoff_range[1]:
            # Reorder the patch to channel last format and convert the image patch to HED color coding.
            #
            patch_hed = rgb2hed(rgb=img)

            # Augment the Haematoxylin channel.
            #
            if self.__sigmas[0] != 0.0:
                patch_hed[:, :, 0] *= (1.0 + self.__sigmas[0])

            if self.__biases[0] != 0.0:
                patch_hed[:, :, 0] += self.__biases[0]

            # Augment the Eosin channel.
            #
            if self.__sigmas[1] != 0.0:
                patch_hed[:, :, 1] *= (1.0 + self.__sigmas[1])

            if self.__biases[1] != 0.0:
                patch_hed[:, :, 1] += self.__biases[1]

            # Augment the DAB channel.
            #
            if self.__sigmas[2] != 0.0:
                patch_hed[:, :, 2] *= (1.0 + self.__sigmas[2])

            if self.__biases[2] != 0.0:
                patch_hed[:, :, 2] += self.__biases[2]

            # Convert back to RGB color coding and order back to channels first order.
            #
            patch_rgb = hed2rgb(hed=patch_hed)
            patch_rgb = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)
            patch_rgb *= 255.0
            patch_rgb = patch_rgb.astype(dtype=np.uint8)

            return patch_rgb

        else:
            # The image patch is outside the cutoff interval.
            #
            return img

    def get_transform_init_args_names(self):
        return ("haematoxylin_sigma_range",
        "haematoxylin_bias_range", 
        "eosin_sigma_range", 
        "eosin_bias_range", 
        "dab_sigma_range", 
        "dab_bias_range", 
        "cutoff_range")


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    """Borrowed from MoCo implementation"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class FixedRandomRotation:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = F.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)

class AlbumentationsTransform:
    """Wrapper for Albumnetation transforms"""
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        aug_img = self.aug(image=np.array(img))['image']
        return aug_img



def torchvision_transforms(eval=False, aug=None):

    trans = []

    if aug["randcrop"] and aug["scale"] and not eval:
        trans.append(transforms.RandomResizedCrop(aug["randcrop"], scale=aug["scale"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip(p=0.5))
        trans.append(transforms.RandomVerticalFlip(p=0.5))

    if aug["jitter_d"] and not eval:
        trans.append(transforms.RandomApply(
            [transforms.ColorJitter(brightness = 0.2*aug["jitter_d"], contrast = 0.4*aug["jitter_d"], saturation = 0.8*aug["jitter_d"], hue = 0.1*aug["jitter_d"])],
             p=aug["jitter_p"]))

    if aug["gaussian_blur"] and not eval:
        trans.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=aug["gaussian_blur"]))

    if aug["rotation"] and not eval:
        # rotation_transform = FixedRandomRotation(angles=[0, 90, 180, 270])
        trans.append(FixedRandomRotation(angles=[0, 90, 180, 270]))

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())

    else:
        trans.append(transforms.ToTensor())

    # trans = transforms.Compose(trans)
    return trans

def album_transforms(eval=False, aug=None):
    trans = []

    if aug["randcrop"] and not eval:
        #trans.append(A.PadIfNeeded(min_height=aug["randcrop"], min_width=aug["randcrop"]))
        trans.append(A.RandomResizedCrop(width=aug["randcrop"], height=aug["randcrop"], scale=aug["scale"]))

    if aug["randcrop"] and eval:
        #trans.append(A.PadIfNeeded(min_height=aug["randcrop"], min_width=aug["randcrop"]))
        trans.append(A.CenterCrop(width=aug["randcrop"], height=aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(A.Flip(p=0.5))
        #trans.append(A.HorizontalFlip(p=0.5))

    if aug["jitter_d"] and not eval:
        trans.append(A.ColorJitter(brightness = 0.2*aug["jitter_d"], contrast = 0.4*aug["jitter_d"], saturation = 0.8*aug["jitter_d"], hue = 0.1*aug["jitter_d"],
                                   p=aug["jitter_p"]))
        
    if aug["hed"] and not eval:
        trans.append(HEAugment(p=0.5))

    if aug["gaussian_blur"] and not eval:
        trans.append(A.GaussianBlur(blur_limit=(3,7), sigma_limit=(0.1, 2), p=aug["gaussian_blur"]))

    if aug["rotation"] and not eval:
        trans.append(A.RandomRotate90(p=0.5))

    # Pathology specific augmentation
    if aug["grid_distort"] and not eval:
        trans.append(A.GridDistortion(num_steps=9, distort_limit=0.2, interpolation=1, border_mode=2, p=aug["grid_distort"]))
    if aug["contrast"] and not eval:
        trans.append(A.RandomContrast(limit=aug["contrast"], p=aug["contrast_p"]))
    if aug["grid_shuffle"] and not eval:
        trans.append(A.RandomGridShuffle(grid=(3, 3), p=aug["grid_shuffle"]))

    trans.append(A.ToFloat(max_value=255.0))
    trans.append(ToTensorV2())
    return trans