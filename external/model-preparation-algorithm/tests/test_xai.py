# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch

from mmcls.models import build_classifier

from mpa.modules.hooks.auxiliary_hooks import ReciproCAMHook


torch.manual_seed(0)


class TestExplainMethods:
    @staticmethod
    def get_model():
        model_cfg = dict(
            type='ImageClassifier',
            backbone=dict(
                type='ResNet',
                depth=18,
                num_stages=4,
                out_indices=(3,),
                style='pytorch'),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=2,
                in_channels=512,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5),
            ))

        model = build_classifier(model_cfg)
        return model.eval()

    def test_recipro_cam(self):
        model = self.get_model()
        img = torch.rand(2, 3, 224, 224) - 0.5
        data = {'img_metas': {}, 'img': img}

        with ReciproCAMHook(model) as rcam_hook:
            _ = model(return_loss=False, **data)
        saliency_maps = rcam_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == (2, 7, 7)

        sal_class0_reference = np.array([[ 27, 112, 105,  92, 114,  84,  73],
                                   [  0,  88, 144,  91,  58, 103,  96],
                                   [ 54, 155, 121, 105,   1, 111, 146],
                                   [ 34, 199, 122, 159, 126, 164,  83],
                                   [ 59,  70, 237, 149, 127, 164, 148],
                                   [ 14, 213, 150, 135, 124, 215,  43],
                                   [ 36,  98, 114,  61,  99, 255,  30]], dtype=np.uint8)
        sal_class1_reference = np.array([[227, 142, 149, 162, 140, 170, 181],
                                   [255, 166, 110, 163, 196, 151, 158],
                                   [200,  99, 133, 149, 253, 142, 108],
                                   [220,  55, 132,  95, 128,  90, 171],
                                   [195, 184,  17, 105, 127,  90, 106],
                                   [240,  41, 104, 119, 130,  39, 211],
                                   [218, 156, 140, 193, 155,   0, 224]], dtype=np.uint8)
        assert (saliency_maps[0][0] == sal_class0_reference).all()
        assert (saliency_maps[0][1] == sal_class1_reference).all()
