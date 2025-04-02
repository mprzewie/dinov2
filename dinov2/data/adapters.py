# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Tuple

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
    def __init__(self, encoding_size: int=1000):
        self.encoding_size = encoding_size
        self.warned = False

    def __call__(self, target) -> torch.Tensor:
        if isinstance(target, int):
            encoding = torch.zeros(self.encoding_size)
            if target >= 0 and target < self.encoding_size:
                encoding[target] = 1
            elif not self.warned:
                logger.warning(
                    f"Target {target} is out of bounds for encoding size {self.encoding_size}"
                )
                self.warned = True
        else:
            raise NotImplementedError((target, type(target)))

        return encoding

class TargetKeeperAndEncoder:
    def __init__(self, target_encoder: TargetEncoder):
        self.target_encoder = target_encoder

    def __call__(self, target):
        return target, self.target_encoder(target)