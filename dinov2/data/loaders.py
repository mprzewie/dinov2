# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np
import torch
from torch.nn.functional import embedding
from torch.utils.data import Sampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from .datasets import ImageNet, ImageNet22k
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
from ..ctrlo import ocl_transforms, ocl_preprocessing
from ..ctrlo.datasets import WebdatasetDataModule

logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        # assert key in ("root", "extra", "split")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    elif name == "ImageFolder":
        class_ = ImageFolder
    elif name == "ctrlo":
        class_ = None
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def _numpycopy(np_array: np.ndarray):
    return np_array.copy()

def _pad_with_fake_target(image_dino, target_transform: Optional[Callable]):
    fake_target = -1
    if target_transform is not None:
        fake_target = target_transform(fake_target)
    return image_dino, fake_target

def make_dataset(
    *,
    dataset_str: str,
    transform_dino: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform_dino: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    if dataset_str.startswith("Image"):
        dataset = class_(transform=transform_dino, target_transform=target_transform, **kwargs)
        logger.info(f"# of dataset samples: {len(dataset):,d}")
    elif dataset_str.startswith("ctrlo"):
        ds_root = Path(kwargs["root"])
        ds_name = kwargs["ds_name"]
        ds_split = kwargs["ds_split"]
        ds_size = int(kwargs["ds_size"])
        preprocessing_transform_03a = ocl_transforms.Map(
            transform=transforms.Compose([
                ocl_preprocessing.SelectConditioningInfoVG(
                    embeddings_path=str(ds_root / "category_name_to_llama3_emb.pkl"),
                    num_max_binds=3,
                    num_slots=3
                ),  # Replace with `experiment.num_slots`
                ocl_preprocessing.CopyFields(mapping={"instance_mask": "instance_mask_v2"})
            ]),
            fields=("image", "instance_mask", "instance_bbox", "name", "bbox_centroids", "name_embedding", "selected_indices", "contrastive_loss_mask", "all_bbox_centroids"),
            batch_transform=False
        )

        train_transform_03b = ocl_transforms.SimpleTransform(
            transforms={
                "image": transforms.Compose([
                    # transforms.Lambda(lambda image: image.copy()),
                    transforms.Lambda(_numpycopy),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                "image_dino": transforms.Compose([
                    transforms.ToPILImage(),
                    transform_dino,
                    transforms.Lambda(_pad_with_fake_target),
                ]),
                "name_embedding": transforms.Compose([
                    # transforms.Lambda(lambda name_embedding: name_embedding.copy()),
                    transforms.Lambda(_numpycopy),
                    ocl_preprocessing.ToTensor()
                ]),
                "bbox_centroids": transforms.Compose([
                    # transforms.Lambda(lambda bbox_centroids: bbox_centroids.copy()),
                    transforms.Lambda(_numpycopy),
                    ocl_preprocessing.ToTensor()
                ]),
                "all_bbox_centroids": transforms.Compose([
                    # transforms.Lambda(lambda all_bbox_centroids: all_bbox_centroids.copy()),
                    transforms.Lambda(_numpycopy),
                    ocl_preprocessing.ToTensor()
                ]),
                "selected_indices": transforms.Compose([
                    # transforms.Lambda(lambda selected_indices: selected_indices.copy()),
                    transforms.Lambda(_numpycopy),
                    ocl_preprocessing.ToTensor()
                ]),
                "contrastive_loss_mask": transforms.Compose([
                    # transforms.Lambda(lambda contrastive_loss_mask: contrastive_loss_mask.copy()),
                    transforms.Lambda(_numpycopy),
                    ocl_preprocessing.ToTensor()
                ]),
                "instance_mask": transforms.Compose([
                    ocl_preprocessing.IntegerToOneHotMask(output_axis=-3),
                    ocl_preprocessing.AddEmptyMasksVG(),
                    ocl_preprocessing.DenseMaskToTensor()
                ]),
                "instance_mask_v2": transforms.Compose([
                    ocl_preprocessing.IntegerToOneHotMask(output_axis=-3),
                    ocl_preprocessing.AddEmptyMasksVG(),
                    ocl_preprocessing.DenseMaskToTensor()
                ])
            },
            batch_transform=False
        )

        eval_transforms_03c = ocl_transforms.SimpleTransform(
            transforms={
                "image": transforms.Compose([
                    # transforms.Lambda(lambda image: image.copy()),
                    transforms.Lambda(_numpycopy),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                "instance_mask": transforms.Compose([
                    ocl_preprocessing.IntegerToOneHotMask(output_axis=-3),
                    ocl_preprocessing.AddEmptyMasksVG(),
                    ocl_preprocessing.DenseMaskToTensor()
                ]),
                "instance_mask_v2": transforms.Compose([
                    ocl_preprocessing.IntegerToOneHotMask(output_axis=-3),
                    ocl_preprocessing.AddEmptyMasksVG(),
                    ocl_preprocessing.DenseMaskToTensor()
                ])
            },
            batch_transform=False
        )

        eval_transforms = {
            "03a_preprocessing": preprocessing_transform_03a,
            "03c_preprocessing": eval_transforms_03c,
        }

        train_transforms = {
            "03a_preprocessing": preprocessing_transform_03a,
            "03ab_image_duplicate": ocl_transforms.DuplicateFields({"image": "image_dino"}, batch_transform=False),
            "03b_preprocessing": train_transform_03b,
        }

        dataset = WebdatasetDataModule(
            num_workers=1,
            batch_size=1,
            train_shards=f"{ds_root}/{ds_name}/{ds_split}/shard-{{000000..001100}}.tar",
            val_shards=f"{ds_root}/{ds_name}/{ds_split}/shard-{{000000..001100}}.tar",
            test_shards=f"{ds_root}/{ds_name}/{ds_split}/shard-{{000000..001100}}.tar",
            train_size=ds_size,
            val_size=ds_size,
            test_size=ds_size,
            use_autopadding=True,
            train_transforms=train_transforms,
            eval_transforms=eval_transforms,
            shuffle_train=True,
            use_epochs=False,
        )

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform_dino)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    wd_train: bool = True,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    if isinstance(dataset, torch.utils.data.Dataset):
        sampler = _make_sampler(
            dataset=dataset,
            type=sampler_type,
            shuffle=shuffle,
            seed=seed,
            size=sampler_size,
            advance=sampler_advance,
        )

        def worker_init_fn(worker_id):
            os.sched_setaffinity(0, range(os.cpu_count()))

        logger.info("using PyTorch data loader")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn
        )



        try:
            logger.info(f"# of batches: {len(data_loader):,d}")
        except TypeError:  # data loader has no length
            logger.info("infinite data loader")
        return data_loader

    elif isinstance(dataset, WebdatasetDataModule):
        dataset.batch_size = batch_size
        dataset.num_workers = num_workers
        dataset.shuffle_train = shuffle

        if wd_train:
            return dataset.train_dataloader()
        else:
            return dataset.val_dataloader()

