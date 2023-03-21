# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import numpy as np
import torch
from mmcv.runner import auto_fp16
from mmdet.models.builder import DETECTORS
from torch import nn

from otx.mpa.deploy.utils import is_mmdeploy_enabled
from otx.mpa.utils.logger import get_logger

from .custom_maskrcnn_detector import CustomMaskRCNN

logger = get_logger()


class TileClassifier(torch.nn.Module):
    def __init__(self):
        """Tile classifier model"""
        super().__init__()
        self.fp16_enabled = False
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 6 * 6, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),
        )

        self.loss_fun = torch.nn.BCEWithLogitsLoss()
        self.sigmoid = torch.nn.Sigmoid()

    @auto_fp16()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            img (torch.Tensor): input image

        Returns:
            torch.Tensor: logits
        """
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y

    @auto_fp16()
    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate BCE loss.

        Args:
            pred (torch.Tensor): _description_
            target (torch.Tensor): binary target

        Returns:
            torch.Tensor: BCE loss
        """
        loss = self.loss_fun(pred, target)
        return loss

    @auto_fp16()
    def simple_test(self, img: torch.Tensor) -> torch.Tensor:
        """Simple test.

        Args:
            img (torch.Tensor): input image

        Returns:
            torch.Tensor: objectness score
        """
        return self.sigmoid(self.forward(img))[0][0]


@DETECTORS.register_module()
class CustomMaskRCNNTileOptimised(CustomMaskRCNN):
    """Custom MaskRCNN detector with tile classifier."""

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_classifier = TileClassifier()

    def forward_train(
        self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs
    ):
        """Forward pass during training.

        Joint training of tile classifier and MaskRCNN.

        Args:
            img (torch.Tensor): input image
            img_metas (list): image meta data
            gt_bboxes (list): ground truth bounding boxes
            gt_labels (list): ground truth labels
            gt_bboxes_ignore (list, optional): ground truth bounding boxes to be ignored. Defaults to None.
            gt_masks (list, optional): ground truth masks. Defaults to None.
            proposals (list, optional): proposals. Defaults to None.
        """

        losses = dict()
        targets = [len(gt_bbox) > 0 for gt_bbox in gt_bboxes]
        pred = self.tile_classifier(img)
        target_labels = torch.tensor(targets, device=pred.device)
        loss_tile = self.tile_classifier.loss(pred, target_labels.unsqueeze(1).float())

        losses.update(dict(loss_tile=loss_tile))

        if not any(targets):
            return losses

        img = img[targets]
        img_metas = [item for keep, item in zip(targets, img_metas) if keep]
        gt_labels = [item for keep, item in zip(targets, gt_labels) if keep]
        gt_bboxes = [item for keep, item in zip(targets, gt_bboxes) if keep]
        gt_masks = [item for keep, item in zip(targets, gt_masks) if keep]
        if gt_bboxes_ignore:
            gt_bboxes_ignore = [item for keep, item in zip(targets, gt_bboxes_ignore) if keep]
        rcnn_loss = super().forward_train(
            img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, proposals, **kwargs
        )
        losses.update(rcnn_loss)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Simple test.

        Tile classifier is used to filter out images without any objects.
        If no objects are present, empty results are returned. Otherwise, MaskRCNN is used to detect objects.

        Args:
            img (torch.Tensor): input image
            img_metas (list): image meta data
            proposals (list, optional): proposals. Defaults to None.
            rescale (bool, optional): rescale. Defaults to False.

        Returns:
            tuple: MaskRCNN output
        """
        keep = self.tile_classifier.simple_test(img) > 0.45

        if not keep:
            tmp_results = []
            num_classes = 1
            bbox_results = []
            mask_results = []
            for _ in range(num_classes):
                bbox_results.append(np.empty((0, 5), dtype=np.float32))
                mask_results.append([])
            tmp_results.append((bbox_results, mask_results))
            return tmp_results

        assert self.with_bbox, "Bbox head must be implemented."
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER, mark
    from mmdeploy.utils import is_dynamic_shape

    from otx.mpa.modules.hooks.recording_forward_hooks import (
        ActivationMapHook,
        FeatureVectorHook,
    )

    @mark("tile_classifier", inputs=["image"], outputs=["tile_prob"])
    def tile_classifier__simple_test_impl(ctx, self, img):
        """Tile Classifier Simple Test Impl with added mmdeploy marking for model partitioning

        Partition tile classifier by marking tile classifier in tracing.

        Args:
            ctx (object): object context
            img (torch.Tensor): input image

        Returns:
            torch.Tensor: objectness score
        """
        return self.sigmoid(self.forward(img))[0][0]

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_maskrcnn_tile_optimised.TileClassifier.simple_test"
    )
    def tile_classifier__simple_test(ctx, self, img, **kwargs):
        """Tile Classifier Simple Test Rewriter.

        Partition tile classifier by rewriting tile classifier simple test.

        Args:
            ctx (object): object context
            img (torch.Tensor): input image

        Returns:
            torch.Tensor: objectness score
        """
        return tile_classifier__simple_test_impl(ctx, self, img)

    @mark("custom_maskrcnn_forward", inputs=["input"], outputs=["dets", "labels", "masks", "feats", "saliencies"])
    def __forward_impl(ctx, self, img, img_metas, **kwargs):
        assert isinstance(img, torch.Tensor)

        deploy_cfg = ctx.cfg
        is_dynamic_flag = is_dynamic_shape(deploy_cfg)
        # get origin input shape as tensor to support onnx dynamic shape
        img_shape = torch._shape_as_tensor(img)[2:]
        if not is_dynamic_flag:
            img_shape = [int(val) for val in img_shape]
        img_metas[0]["img_shape"] = img_shape
        return self.simple_test(img, img_metas, **kwargs)

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_maskrcnn_tile_optimised.CustomMaskRCNNTileOptimised.forward"
    )
    def custom_maskrcnn__forward(ctx, self, img, img_metas=None, return_loss=False, **kwargs):
        if img_metas is None:
            img_metas = [{}]
        else:
            assert len(img_metas) == 1, "do not support aug_test"
            img_metas = img_metas[0]

        if isinstance(img, list):
            img = img[0]

        return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_maskrcnn_tile_optimised.CustomMaskRCNNTileOptimised.simple_test"
    )
    def custom_mask_rcnn__simple_test(ctx, self, img, img_metas, proposals=None, **kwargs):
        """Custom Mask RCNN Simple Test Rewriter for ONNX tracing

        Tile classifier is added to ONNX tracing in order to partition the model.

        Args:
            ctx (object): object context
            img (torch.Tensor): input image
            img_metas (list): image meta data
            proposals (list, optional): proposals. Defaults to None.

        Returns:
            tuple: MaskRCNN output with tile classifier output
        """
        assert self.with_bbox, "Bbox head must be implemented."
        tile_prob = self.tile_classifier.simple_test(img)

        x = self.extract_feat(img)
        if proposals is None:
            proposals, _ = self.rpn_head.simple_test_rpn(x, img_metas)
        out = self.roi_head.simple_test(x, proposals, img_metas, rescale=False)

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(x)
            saliency_map = ActivationMapHook.func(x[-1])
            return (*out, feature_vector, saliency_map)

        return (*out, tile_prob)
