# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch

from mmcls.models import build_classifier
from mmdet.models import build_detector

from mpa.modules.hooks.auxiliary_hooks import ReciproCAMHook, DetSaliencyMapHook
from mpa.det.stage import DetectionStage


torch.manual_seed(0)


class TestExplainMethods:
    @staticmethod
    def get_classification_model():
        model_cfg = dict(
            type="ImageClassifier",
            backbone=dict(type="ResNet", depth=18, num_stages=4, out_indices=(3,), style="pytorch"),
            neck=dict(type="GlobalAveragePooling"),
            head=dict(
                type="LinearClsHead",
                num_classes=2,
                in_channels=512,
                loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
                topk=(1, 5),
            ),
        )

        model = build_classifier(model_cfg)
        return model.eval()

    @staticmethod
    def get_detection_model():
        model_cfg = dict(
            type='CustomSingleStageDetector',
            backbone=dict(type='mobilenetv2_w1',
                          out_indices=(4, 5),
                          frozen_stages=-1,
                          norm_eval=False,
                          pretrained=True),
            bbox_head=dict(
                type='CustomSSDHead',
                in_channels=(96, 320),
                num_classes=20,
                anchor_generator=dict(
                    type='SSDAnchorGeneratorClustered',
                    reclustering_anchors=True,
                    strides=[16, 32],
                    widths=[np.array([70.93408016, 132.06659281, 189.56180207, 349.90057837]),
                            np.array([272.31733885, 448.52200666, 740.63350023, 530.78990182,
                                      790.99297377])],
                    heights=[np.array([93.83759764, 235.21261441, 432.6029996, 250.08979657]),
                             np.array([672.8829653, 474.84783528, 420.18291446, 741.02592293,
                                       766.45636125])]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
            )
        )
        model = build_detector(model_cfg)
        return model.eval()

    def test_recipro_cam(self):
        model = self.get_classification_model()
        img = torch.rand(2, 3, 224, 224) - 0.5
        data = {"img_metas": {}, "img": img}

        with ReciproCAMHook(model) as rcam_hook:
            _ = model(return_loss=False, **data)
        saliency_maps = rcam_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == (2, 7, 7)

        sal_class0_reference = np.array(
            [
                [27, 112, 105, 92, 114, 84, 73],
                [0, 88, 144, 91, 58, 103, 96],
                [54, 155, 121, 105, 1, 111, 146],
                [34, 199, 122, 159, 126, 164, 83],
                [59, 70, 237, 149, 127, 164, 148],
                [14, 213, 150, 135, 124, 215, 43],
                [36, 98, 114, 61, 99, 255, 30],
            ],
            dtype=np.uint8,
        )
        sal_class1_reference = np.array(
            [
                [227, 142, 149, 162, 140, 170, 181],
                [255, 166, 110, 163, 196, 151, 158],
                [200, 99, 133, 149, 253, 142, 108],
                [220, 55, 132, 95, 128, 90, 171],
                [195, 184, 17, 105, 127, 90, 106],
                [240, 41, 104, 119, 130, 39, 211],
                [218, 156, 140, 193, 155, 0, 224],
            ],
            dtype=np.uint8,
        )
        assert (saliency_maps[0][0] == sal_class0_reference).all()
        assert (saliency_maps[0][1] == sal_class1_reference).all()

    def test_saliency_map_detection(self):
        model = self.get_detection_model()
        img = torch.rand(2, 3, 224, 224) - 0.5
        data = {"img_metas": [{}], "img": [img]}

        with DetSaliencyMapHook(model) as det_hook:
            _ = model(return_loss=False, rescale=True, **data)
        saliency_maps = det_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == (21, 7, 7)

        sal_class0_reference = np.array(
            [
                [189, 224, 89, 115, 133, 135, 176],
                [54, 134, 148, 82, 135, 177, 154],
                [0, 103, 39, 66, 156, 206, 73],
                [82, 126, 142, 123, 210, 147, 167],
                [115, 129, 108, 128, 174, 185, 121],
                [90, 131, 118, 113, 89, 150, 105],
                [106, 189, 148, 180, 206, 255, 145],
            ],
            dtype=np.uint8,
        )
        sal_class1_reference = np.array(
            [
                [230, 138, 133, 92, 127, 101, 77],
                [100, 129, 141, 156, 0, 87, 136],
                [99, 51, 147, 218, 123, 50, 75],
                [111, 98, 70, 142, 172, 110, 73],
                [124, 69, 34, 97, 157, 78, 171],
                [214, 153, 56, 93, 128, 139, 148],
                [189, 255, 208, 224, 169, 167, 202],
            ],
            dtype=np.uint8,
        )
        assert (saliency_maps[0][0] == sal_class0_reference).all()
        assert (saliency_maps[0][1] == sal_class1_reference).all()
