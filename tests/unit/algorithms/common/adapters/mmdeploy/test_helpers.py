# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.utils import Config


class MockModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, img, img_metas=None, **kwargs):
        if isinstance(img, list):
            img = img[0]
        if img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)

        x = self.conv1(img)
        x = self.conv2(img)
        x = torch.mean(x, dim=(2, 3))
        x = self.linear(x)
        return x

    def forward_dummy(self, img, img_metas=None, **kwargs):
        if isinstance(img, list):
            img = img[0]
        if img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)

        x = self.conv1(img)
        x = self.conv2(img)
        x = torch.mean(x, dim=(2, 3))
        x = self.linear(x)
        return x

    def train_step(self, *args, **kwargs):
        return dict()

    def init_weights(self, *args, **kwargs):
        pass

    def set_step_params(self, *args, **kwargs):
        pass


def create_model(lib="mmcls"):
    if lib == "mmcls":
        from mmcls.models import CLASSIFIERS

        CLASSIFIERS.register_module(MockModel, force=True)
    elif lib == "mmdet":
        from mmdet.models import DETECTORS

        DETECTORS.register_module(MockModel, force=True)
    elif lib == "mmseg":
        from mmseg.models import SEGMENTORS

        SEGMENTORS.register_module(MockModel, force=True)
    else:
        raise ValueError()

    return MockModel()


def create_config():
    config = Config(
        {
            "model": {
                "type": "MockModel",
            },
            "data": {
                "test": {
                    "pipeline": [
                        {"type": "LoadImageFromFile"},
                        {"type": "Normalize", "mean": [0, 0, 0], "std": [1, 1, 1]},
                        {"type": "ImageToTensor", "keys": ["img"]},
                    ]
                }
            },
        }
    )

    return config
