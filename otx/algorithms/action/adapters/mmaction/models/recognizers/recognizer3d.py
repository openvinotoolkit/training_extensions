# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import OrderedDict

import torch
from mmaction.models.builder import RECOGNIZERS
from mmaction.models.recognizers.base import BaseRecognizer
from torch import nn


@RECOGNIZERS.register_module(force=True)
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect model as output state_dict for OTX model compatibility"""
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["MoViNetBase"]:
            return

        output = OrderedDict()
        if backbone_type == "MoViNetBase":
            for k, v in state_dict.items():
                if k.startswith("cls_head"):
                    k = k.replace("cls_head.", "")
                else:
                    k = k.replace("backbone.", "")
                output[k] = v
        return output

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):
        """Redirect input state_dict to model for OTX model compatibility"""
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["MoViNetBase"]:
            return

        if backbone_type == "MoViNetBase":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith("classifier"):
                    k = k.replace("classifier", "cls_head.classifier")
                else:
                    k = "backbone." + k
                state_dict[k] = v

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, "max_testing_views is only compatible " "with batch_size == 1"
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr : view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [torch.cat([x[i] for x in feats]) for i in range(len_tuple)]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(feat.size())
            assert feat_dim in [5, 2], (
                "Got feature of unknown architecture, "
                "only 3D-CNN-like ([N, in_channels, T, H, W]), and "
                "transformer-like ([N, in_channels]) features are supported."
            )
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs,)

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
