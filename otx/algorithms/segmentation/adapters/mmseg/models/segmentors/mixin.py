"""Modules for decode and loss reweighting/mix."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from mmseg.core import add_prefix
from mmseg.models.builder import build_loss
from mmseg.ops import resize
from torch import nn

from otx.algorithms.segmentation.adapters.mmseg.models.utils import LossEqualizer

# pylint: disable=too-many-locals


class PixelWeightsMixin:
    """PixelWeightsMixin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_train_components(self.train_cfg)
        self.feature_maps = None

    def _init_train_components(self, train_cfg):
        if train_cfg is None:
            self.mutual_losses = None
            self.loss_equalizer = None
            return

        mutual_loss_configs = train_cfg.get("mutual_loss")
        if mutual_loss_configs:
            if isinstance(mutual_loss_configs, dict):
                mutual_loss_configs = [mutual_loss_configs]

            self.mutual_losses = nn.ModuleList()
            for mutual_loss_config in mutual_loss_configs:
                self.mutual_losses.append(build_loss(mutual_loss_config))
        else:
            self.mutual_losses = None

        loss_reweighting_config = train_cfg.get("loss_reweighting")
        if loss_reweighting_config:
            self.loss_equalizer = LossEqualizer(**loss_reweighting_config)
        else:
            self.loss_equalizer = None

    @staticmethod
    def _get_argument_by_name(trg_name, arguments):
        assert trg_name in arguments.keys()
        return arguments[trg_name]

    def set_step_params(self, init_iter, epoch_size):
        """Sets the step params for the current object's decode head."""
        self.decode_head.set_step_params(init_iter, epoch_size)

        if getattr(self, "auxiliary_head", None) is not None:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.set_step_params(init_iter, epoch_size)
            else:
                self.auxiliary_head.set_step_params(init_iter, epoch_size)

    def _decode_head_forward_train(self, x, img_metas, pixel_weights=None, **kwargs):
        """Run forward train in decode head."""
        trg_map = self._get_argument_by_name(self.decode_head.loss_target_name, kwargs)
        loss_decode, logits_decode = self.decode_head.forward_train(
            x,
            img_metas,
            trg_map,
            train_cfg=self.train_cfg,
            pixel_weights=pixel_weights,
            return_logits=True,
        )

        scale = self.decode_head.last_scale
        scaled_logits_decode = scale * logits_decode

        name_prefix = "decode"

        losses, meta = dict(), dict()
        losses.update(add_prefix(loss_decode, name_prefix))
        meta[f"{name_prefix}_scaled_logits"] = scaled_logits_decode

        return losses, meta

    def _auxiliary_head_forward_train(self, x, img_metas, **kwargs):

        losses, meta = dict(), dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                trg_map = self._get_argument_by_name(aux_head.loss_target_name, kwargs)
                loss_aux, logits_aux = aux_head.forward_train(
                    x,
                    img_metas,
                    trg_map,
                    train_cfg=self.train_cfg,
                    return_logits=True,
                )

                scale = aux_head.last_scale
                scaled_logits_aux = scale * logits_aux

                name_prefix = f"aux_{idx}"
                losses.update(add_prefix(loss_aux, name_prefix))
                meta[f"{name_prefix}_scaled_logits"] = scaled_logits_aux
        else:
            trg_map = self._get_argument_by_name(self.auxiliary_head.loss_target_name, kwargs)
            loss_aux, logits_aux = self.auxiliary_head.forward_train(
                x,
                img_metas,
                trg_map,
                train_cfg=self.train_cfg,
                return_logits=True,
            )

            scale = self.auxiliary_head.last_scale
            scaled_logits_aux = scale * logits_aux

            name_prefix = "aux"
            losses.update(add_prefix(loss_aux, name_prefix))
            meta[f"{name_prefix}_scaled_logits"] = scaled_logits_aux

        return losses, meta

    def forward_train(self, img, img_metas, gt_semantic_seg, pixel_weights=None, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()

        features = self.extract_feat(img)

        loss_decode, meta_decode = self._decode_head_forward_train(
            features, img_metas, pixel_weights, gt_semantic_seg=gt_semantic_seg, **kwargs
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux, meta_aux = self._auxiliary_head_forward_train(
                features, img_metas, gt_semantic_seg=gt_semantic_seg, **kwargs
            )
            losses.update(loss_aux)

        if self.mutual_losses is not None and self.with_auxiliary_head:
            meta = dict()
            meta.update(meta_decode)
            meta.update(meta_aux)

            out_mutual_losses = dict()
            for mutual_loss_idx, mutual_loss in enumerate(self.mutual_losses):
                logits_a = self._get_argument_by_name(mutual_loss.trg_a_name, meta)
                logits_b = self._get_argument_by_name(mutual_loss.trg_b_name, meta)

                logits_a = resize(
                    input=logits_a, size=gt_semantic_seg.shape[2:], mode="bilinear", align_corners=self.align_corners
                )
                logits_b = resize(
                    input=logits_b, size=gt_semantic_seg.shape[2:], mode="bilinear", align_corners=self.align_corners
                )

                mutual_labels = gt_semantic_seg.squeeze(1)
                mutual_loss_value, mutual_loss_meta = mutual_loss(logits_a, logits_b, mutual_labels)

                mutual_loss_name = mutual_loss.name + f"-{mutual_loss_idx}"
                out_mutual_losses[mutual_loss_name] = mutual_loss_value
                losses[mutual_loss_name] = mutual_loss_value
                losses.update(add_prefix(mutual_loss_meta, mutual_loss_name))

            losses["loss_mutual"] = sum(out_mutual_losses.values())

        if self.loss_equalizer is not None:
            unweighted_losses = {loss_name: loss for loss_name, loss in losses.items() if "loss" in loss_name}
            weighted_losses = self.loss_equalizer.reweight(unweighted_losses)

            for loss_name, loss_value in weighted_losses.items():
                losses[loss_name] = loss_value

        return losses
