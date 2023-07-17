"""Custom MaskRCNN detector with tile classifier."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import numpy as np
import torch
from mmcls.models.necks.gap import GlobalAveragePooling
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from mmdet.models.builder import DETECTORS
from torch import nn

from otx.algorithms.common.adapters.mmdeploy import is_mmdeploy_enabled
from otx.algorithms.common.adapters.nncf import no_nncf_trace

from .custom_maskrcnn_detector import CustomMaskRCNN


class TileClassifier(torch.nn.Module):
    """Tile classifier for the tile optimised MaskRCNN model."""

    def __init__(self):
        super().__init__()
        self.fp16_enabled = False
        self.features = nn.Sequential(
            ConvModule(3, 64, 11, stride=4, padding=2, act_cfg=dict(type="ReLU")),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvModule(64, 192, 5, padding=2, act_cfg=dict(type="ReLU")),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvModule(192, 256, 3, padding=1, act_cfg=dict(type="ReLU")),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvModule(256, 256, 3, padding=1, act_cfg=dict(type="ReLU")),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # NOTE: Original Adaptive Avg Pooling is replaced with Global Avg Pooling
        # due to ONNX tracing issues: https://github.com/openvinotoolkit/training_extensions/pull/2337

        self.gap = GlobalAveragePooling()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),
        )

        self.loss_func = torch.nn.BCEWithLogitsLoss()
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
        x = self.gap(x)
        y = self.classifier(x)
        return y

    @auto_fp16()
    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate BCE loss.

        Args:
            pred (torch.Tensor): logits
            target (torch.Tensor): binary target

        Returns:
            torch.Tensor: BCE loss
        """
        loss = self.loss_func(pred, target)
        return loss

    @auto_fp16()
    def simple_test(self, img: torch.Tensor) -> torch.Tensor:
        """Simple test.

        Args:
            img (torch.Tensor): input image

        Returns:
            torch.Tensor: objectness score
        """

        out = self.forward(img)
        with no_nncf_trace():
            return self.sigmoid(out).flatten()


# pylint: disable=too-many-ancestors
@DETECTORS.register_module()
class CustomMaskRCNNTileOptimized(CustomMaskRCNN):
    """Custom MaskRCNN detector with tile classifier.

    Args:
        *args: args
        **kwargs: kwargs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_classifier = TileClassifier()

    # pylint: disable=too-many-arguments
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
            kwargs: kwargs
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

    @staticmethod
    def make_fake_results(num_classes):
        """Make fake results.

        Returns:
            tuple: MaskRCNN output
        """
        bbox_results = []
        mask_results = []
        for _ in range(num_classes):
            bbox_results.append(np.empty((0, 5), dtype=np.float32))
            mask_results.append([])
        return bbox_results, mask_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False, full_res_image=False):
        """Simple test.

        Tile classifier is used to filter out images without any objects.
        If no objects are present, empty results are returned. Otherwise, MaskRCNN is used to detect objects.

        Args:
            img (torch.Tensor): input image
            img_metas (list): image meta data
            proposals (list, optional): proposals. Defaults to None.
            rescale (bool, optional): rescale. Defaults to False.
            full_res_image (bool, optional): if the image is full resolution or not. Defaults to False.

        Returns:
            tuple: MaskRCNN output
        """
        keep = self.tile_classifier.simple_test(img) > 0.45
        if isinstance(full_res_image, bool):
            full_res_image = [full_res_image]
        keep = full_res_image[0] | keep

        results = []
        for _ in range(len(img)):
            fake_result = CustomMaskRCNNTileOptimized.make_fake_results(self.roi_head.bbox_head.num_classes)
            results.append(fake_result)

        if any(keep):
            img = img[keep]
            img_metas = [item for keep, item in zip(keep, img_metas) if keep]
            assert self.with_bbox, "Bbox head must be implemented."
            x = self.extract_feat(img)

            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            maskrcnn_results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
            for i, keep_flag in enumerate(keep):
                if keep_flag:
                    results[i] = maskrcnn_results.pop(0)
        return results


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER, mark
    from mmdeploy.utils import is_dynamic_shape

    # pylint: disable=ungrouped-imports
    from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
        FeatureVectorHook,
    )

    @mark("tile_classifier", inputs=["image"], outputs=["tile_prob"])
    def tile_classifier__simple_test_impl(self, img):
        """Tile Classifier Simple Test Impl with added mmdeploy marking for model partitioning.

        Partition tile classifier by marking tile classifier in tracing.

        Args:
            self (object): object
            img: input image

        Returns:
            torch.Tensor: objectness score
        """
        return self.sigmoid(self.forward(img))[0][0]

    # pylint: disable=line-too-long, unused-argument
    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_maskrcnn_tile_optimized.TileClassifier.simple_test"  # noqa: E501
    )
    def tile_classifier__simple_test(ctx, self, img):
        """Tile Classifier Simple Test Rewriter.

        Partition tile classifier by rewriting tile classifier simple test.

        Args:
            ctx (object): object context
            self (object): object
            img (torch.Tensor): input image

        Returns:
            torch.Tensor: objectness score
        """
        return tile_classifier__simple_test_impl(self, img)

    # pylint: disable=protected-access
    @mark(
        "custom_maskrcnn_forward",
        inputs=["input"],
        outputs=["dets", "labels", "masks", "tile_prob", "feats", "saliencies"],
    )
    def __forward_impl(ctx, self, img, img_metas, **kwargs):
        """Custom MaskRCNN Forward Impl with added mmdeploy marking for model partitioning.

        Args:
            ctx (object): object context
            self (object): object
            img (torch.Tensor): input image
            img_metas (dict): image meta data
            **kwargs: kwargs

        Returns:
            simple test: MaskRCNN output
        """
        assert isinstance(img, torch.Tensor)

        deploy_cfg = ctx.cfg
        is_dynamic_flag = is_dynamic_shape(deploy_cfg)
        # get origin input shape as tensor to support onnx dynamic shape
        img_shape = torch._shape_as_tensor(img)[2:]
        if not is_dynamic_flag:
            img_shape = [int(val) for val in img_shape]
        img_metas[0]["img_shape"] = img_shape
        return self.simple_test(img, img_metas, **kwargs)

    # pylint: disable=line-too-long
    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_maskrcnn_tile_optimized.CustomMaskRCNNTileOptimized.forward"  # noqa: E501
    )
    def custom_maskrcnn__forward(ctx, self, img, img_metas=None, **kwargs):
        """Custom MaskRCNN Forward Rewriter.

        Args:
            ctx (object): object context
            self (object): object
            img (torch.Tensor): input image
            img_metas (dict, optional): image meta data. Defaults to None.
            **kwargs: kwargs

        Returns:
            MaskRCNN output
        """
        if img_metas is None:
            img_metas = [{}]
        else:
            assert len(img_metas) == 1, "do not support aug_test"
            img_metas = img_metas[0]

        if isinstance(img, list):
            img = img[0]

        return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)

    # pylint: disable=line-too-long
    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_maskrcnn_tile_optimized.CustomMaskRCNNTileOptimized.simple_test"  # noqa: E501
    )
    def custom_mask_rcnn__simple_test(ctx, self, img, img_metas, proposals=None):
        """Custom Mask RCNN Simple Test Rewriter for ONNX tracing.

        Tile classifier is added to ONNX tracing in order to partition the model.

        Args:
            ctx (object): object context
            self (object): object
            img (torch.Tensor): input image
            img_metas (list): image meta data
            proposals (list, optional): proposals. Defaults to None.

        Returns:
            tuple: MaskRCNN output with tile classifier output
        """
        assert self.with_bbox, "Bbox head must be implemented."
        tile_prob = self.tile_classifier.simple_test(img)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        if proposals is None:
            proposals, _ = self.rpn_head.simple_test_rpn(x, img_metas)
        out = self.roi_head.simple_test(x, proposals, img_metas, rescale=False)

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(x)
            # Saliency map will be generated from predictions. Generate dummy saliency_map.
            saliency_map = torch.empty(1, dtype=torch.uint8)
            return (*out, tile_prob, feature_vector, saliency_map)

        return (*out, tile_prob)
