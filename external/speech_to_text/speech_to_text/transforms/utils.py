# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import typing
import string
import torch
import numpy as np


def tokens_to_tensor(
        tokens: typing.List[int],
        pad_id: int = 0,
        target_length: typing.Optional[int] = None,
) -> torch.LongTensor:
    if target_length is not None:
        pad_size = target_length - len(tokens)
        if pad_size > 0:
            tokens = np.hstack((tokens, np.full(pad_size, pad_id)))
        if len(tokens) > target_length:
            tokens = tokens[:target_length]
    return torch.tensor(tokens).long()


class AudioCompose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, data, sample_rate):
        for t in self.transforms:
            data, sample_rate = t(data, sample_rate)
        return data, sample_rate


class ToNumpy:
    def __init__(self, flatten=False):
        self.flatten = flatten

    def __call__(self, data, sample_rate):
        if self.flatten:
            data = data.flatten()
        return data.numpy(), sample_rate


class ToTensor:
    def __call__(self, data, sample_rate):
        return torch.from_numpy(data), sample_rate
