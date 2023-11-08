# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Sequence

import torch
from torchvision import transforms
from torchvision.transforms import functional as F



logger = logging.getLogger("dinov2")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

class CASSLECompose(transforms.Compose):
    def __call__(self, img):
        augmentations_dict = dict()
        for i, t in enumerate(self.transforms):
            img = t(img)
            if isinstance(img, tuple) and len(img) == 2 and isinstance(img[1], dict):
                img, aug_params = img
                augmentations_dict.update(aug_params)
        return img, augmentations_dict



class CASSLERandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        _, height, width = F.get_dimensions(img)
        aug_params = {
            "crop": torch.tensor([i / height, j / width, h / height, w / width])
        }
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias), aug_params


class CASSLERandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        did_flip = 0.0
        if torch.rand(1) < self.p:
            did_flip = 1.0
            img = F.hflip(img)
        return img, {"flip": torch.tensor([did_flip])}


class CASSLEColorJitter(transforms.ColorJitter):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        img0 = img
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        x_means = F.to_tensor(img0).mean(dim=[1, 2])
        x_new_means = F.to_tensor(img).mean(dim=[1, 2])
        color_diff = (x_means - x_new_means).cpu()

        return img, {
            "jitter": torch.tensor([
                (brightness_factor or 1.0),
                (contrast_factor or 1.0),
                (saturation_factor or 1.0),
                (hue_factor or 0.0),
            ]),
            "diff": color_diff
        }


class CASSLERandomGrayscale(transforms.RandomGrayscale):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        did_grayscale = 0.0
        num_output_channels, _, _ = F.get_dimensions(img)
        if torch.rand(1) < self.p:
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            did_grayscale = 1.0
        return img, {"grayscale": torch.tensor([did_grayscale])}


class CASSLEGaussianBlur(transforms.GaussianBlur):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma]), {"blur": torch.tensor([sigma])}

class CASSLERandomApply(transforms.RandomApply):
    def __init__(self, transforms, p=0.5, defaults=None):
        super().__init__(transforms, p=p)
        self.defaults = defaults or dict()

    def forward(self, img):
        applied = torch.rand(1)
        if self.p <= applied:
            return img, self.defaults
        augmentations_dict = dict()
        for i, t in enumerate(self.transforms):
            img = t(img)
            if isinstance(img, tuple) and len(img) == 2:

                img, aug_params = img
                assert isinstance(aug_params, dict)
                augmentations_dict.update(aug_params)

        assert set(augmentations_dict.keys()) == set(self.defaults.keys()), (
            set(augmentations_dict.keys()),
            set(self.defaults.keys()),
            self.transforms,
            augmentations_dict,
            self.p,
            applied
        )
        return img, augmentations_dict

class CASSLEGaussianBlur2(CASSLERandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = CASSLEGaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p, defaults={"blur": torch.tensor([0.0])})





class CASSLERandomSolarize(transforms.RandomSolarize):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be solarized.

        Returns:
            PIL Image or Tensor: Randomly solarized image.
        """
        did_solarize = 0.0
        if torch.rand(1).item() < self.p:
            did_solarize = 1.0
            img = F.solarize(img, self.threshold)
        return img, {"solarization": torch.tensor([did_solarize])}


class CASSLEDataAugmentationDINO(object):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = CASSLECompose(
            [
                CASSLERandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                CASSLERandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = CASSLECompose(
            [
                CASSLERandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                CASSLERandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = CASSLECompose(
            [
                CASSLERandomApply(
                    [CASSLEColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                    defaults={
                        "jitter": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                        "diff": torch.tensor([0.0, 0.0, 0.0])
                    }
                ),
                CASSLERandomGrayscale(p=0.2),
            ]
        )

        solarization_noop = CASSLERandomApply([], p=0, defaults={"solarization": torch.tensor([0.0])})
        global_transfo1_extra = CASSLECompose(
            [
                CASSLEGaussianBlur2(p=1.0),
                solarization_noop
            ]
        )
        global_transfo2_extra = CASSLECompose(
            [
                CASSLEGaussianBlur2(p=0.1),
                CASSLERandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = CASSLEGaussianBlur2(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = CASSLECompose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = CASSLECompose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = CASSLECompose([
            color_jittering, local_transfo_extra, solarization_noop, self.normalize
        ])

    def __call__(self, image):
        output = {}

        im1_base, im1_base_ap = self.geometric_augmentation_global(image)
        global_crop_1, gc1_ap = self.global_transfo1(im1_base)
        gc1_ap.update(im1_base_ap)

        im2_base, im2_base_ap = self.geometric_augmentation_global(image)
        global_crop_2, gc2_ap = self.global_transfo2(im2_base)
        gc2_ap.update(im2_base_ap)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_ap"] = [gc1_ap, gc2_ap]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher_ap"] = [gc1_ap, gc2_ap]

        local_crops = []
        local_crops_ap = []
        for _ in range(self.local_crops_number):
            gal, gal_ap = self.geometric_augmentation_local(image)
            lt, lt_ap = self.local_transfo(gal)
            lt_ap.update(gal_ap)

            local_crops.append(lt)
            local_crops_ap.append(lt_ap)

        output["local_crops"] = local_crops
        output["local_crops_ap"] = local_crops_ap

        output["offsets"] = ()

        return output

# if __name__ == "__main__":
#     aug = CASSLEDataAugmentationDINO(
#         global_crops_scale=[0.32, 1.0],
#         local_crops_scale=[0.05, 0.32],
#         local_crops_number=8,
#         global_crops_size=224,
#         local_crops_size=96
#     )
#
#     from PIL import Image
#
#     img = Image.open("/home/mprzewie/Downloads/NeurIPS_logo_stacked.png")
#
#     img_trans = aug(img)
#
#     print(img_trans.keys())
#
#     for k, v in img_trans.items():
#         if k.endswith("ap"):
#             print(k)
#             # print(v)
#             for ap in v:
#                 print(sorted(ap.keys()))
#             print("----")