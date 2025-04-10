# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import logging
from random import random, randint, shuffle
from typing import Any, Tuple, List

import torch
from torch.utils.data import Dataset


logger = logging.getLogger("dinov2")

class DatasetWithEnumeratedTargets(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def get_image_data(self, index: int) -> bytes:
        return self._dataset.get_image_data(index)

    def get_target(self, index: int) -> Tuple[Any, int]:
        target = self._dataset.get_target(index)
        return (index, target)

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        image, target = self._dataset[index]
        target = index if target is None else target
        return image, (index, target)

    def __len__(self) -> int:
        return len(self._dataset)


class TargetEncoder:
    def __init__(self, encoding_size: int=1000, num_negatives: int = 0):
        self.encoding_size = encoding_size
        self.warned = False
        self.num_negatives = num_negatives

    def __call__(self, target) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(target, int):
            positive = torch.zeros(self.encoding_size)
            if target >= 0 and target < self.encoding_size:
                positive[target] = 1
            elif not self.warned:
                logger.warning(
                    f"Target {target} is out of bounds for encoding size {self.encoding_size}"
                )
                self.warned = True
            assert self.num_negatives < self.encoding_size, f"{self.num_negatives=} >= {self.encoding_size=} doesn't make sense"
            target_pool = [t for t in range(self.encoding_size) if t != target]
            shuffle(target_pool)
            negatives = []
            for t in target_pool[:self.num_negatives]:
                n = torch.zeros(self.encoding_size)
                n[t] = 1
                negatives.append(n)

        else:
            raise NotImplementedError((target, type(target)))

        return positive, (torch.stack(negatives) if len(negatives) > 0 else torch.zeros(0, self.encoding_size))

class RandomEncoder(TargetEncoder):
    def __call__(self, target) -> torch.Tensor:
        target = randint(0, self.encoding_size-1)
        encoding = torch.zeros(self.encoding_size)
        encoding[target] = 1
        return encoding

class TargetKeeperAndEncoder:
    def __init__(self, target_encoder: TargetEncoder):
        self.target_encoder = target_encoder

    def __call__(self, target):
        return target, self.target_encoder(target)