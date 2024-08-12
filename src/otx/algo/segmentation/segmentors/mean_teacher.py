# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base mean teacher algorithm for semi-supervised semantic segmentation learning."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from otx.algo.common.utils.utils import cut_mixer

if TYPE_CHECKING:
    from otx.core.data.entity.base import ImageInfo


class MeanTeacher(nn.Module):
    """MeanTeacher for Semi-supervised learning.

    Args:
        model (nn.Module): model
        unsup_weight (float, optional): unsupervised weight. Defaults to 1.0.
        drop_unrel_pixels_percent (int, optional): drop unrel pixels percent. Defaults to 20.
        semisl_start_epoch (int, optional): semisl start epoch. Defaults to 0.
        filter_pixels_epochs (int, optional): filter pixels epochs. Defaults to 100.
    """

    def __init__(
        self,
        model: nn.Module,
        proto_head: nn.Module | None = None,
        unsup_weight: float = 1.0,
        drop_unrel_pixels_percent: int = 20,
        semisl_start_epoch: int = 0,
        proto_pretrain_epochs: int = 0,
        filter_pixels_epochs: int = 100,
    ) -> None:
        super().__init__()

        self.teacher_model = model
        self.student_model = copy.deepcopy(model)
        # no grads for teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.unsup_weight = unsup_weight
        self.drop_unrel_pixels_percent = drop_unrel_pixels_percent
        # filter unreliable pixels during first X epochs
        self.filter_pixels_epochs = filter_pixels_epochs
        self.semisl_start_epoch = semisl_start_epoch
        self.proto_net = proto_head
        self.proto_pretrain_epochs = proto_pretrain_epochs

    def forward(
        self,
        inputs: Tensor,
        unlabeled_weak_images: Tensor | None = None,
        unlabeled_strong_images: Tensor | None = None,
        global_step: int | None = None,
        steps_per_epoch: int | None = None,
        img_metas: list[ImageInfo] | None = None,
        unlabeled_img_metas: list[ImageInfo] | None = None,
        masks: Tensor | None = None,
        mode: str = "tensor",
    ) -> Tensor:
        """Step for model training.

        Args:
            inputs (Tensor): input labeled images
            unlabeled_weak_images (Tensor, optional): unlabeled images with weak augmentations. Defaults to None.
            unlabeled_strong_images (Tensor, optional): unlabeled images with strong augmentations. Defaults to None.
            global_step (int, optional): global step. Defaults to None.
            steps_per_epoch (int, optional): steps per epoch. Defaults to None.
            img_metas (list[ImageInfo], optional): image meta information. Defaults to None.
            unlabeled_img_metas (list[ImageInfo], optional): unlabeled image meta information. Defaults to None.
            masks (Tensor, optional): ground truth masks for training. Defaults to None.
            mode (str, optional): mode of forward. Defaults to "tensor".
        """
        if mode != "loss":
            # only labeled images for validation and testing
            # teacher_lbld_backbone_feats = self.teacher_model.backbone(inputs)
            return self.teacher_model(inputs, img_metas, masks, mode=mode)
            # return self.proto_net(teacher_lbld_backbone_feats, gt_semantic_seg=masks, pretrain_prototype=False)

        if global_step is None or steps_per_epoch is None:
            msg = "global_step and steps_per_epoch should be provided"
            raise ValueError(msg)

        run_semisl_training = global_step > self.semisl_start_epoch * steps_per_epoch
        # extract features from labeled and unlabeled augmented images
        # we need to separate features from backbone and decode head for proto network
        student_lbld_backbone_feats = self.student_model.backbone(inputs)
        student_labeled_logits = self.student_model.decode_head(student_lbld_backbone_feats)
        loss_decode = self.student_model.calculate_loss(
                student_labeled_logits,
                img_metas,
                masks=masks,
                interpolate=True,
            )

        if run_semisl_training:
            # generate pseudo labels, filter high entropy pixels, compute loss reweight
            percent_unreliable = self.drop_unrel_pixels_percent * (
                1 - global_step / self.filter_pixels_epochs * steps_per_epoch
            )
            pl_from_teacher, reweight_unsup = self._generate_pseudo_labels(
                unlabeled_weak_images,
                percent_unreliable=percent_unreliable,
            )
            unlabeled_strong_images_aug, pl_from_teacher_aug = cut_mixer(unlabeled_strong_images, pl_from_teacher)

            student_unlbld_backbone_feats = self.student_model.backbone(unlabeled_strong_images_aug)
            student_unlabeled_logits = self.student_model.decode_head(student_unlbld_backbone_feats)

            # unsupervised loss
            loss_decode_u = self.student_model.calculate_loss(
                student_unlabeled_logits,
                unlabeled_img_metas,
                masks=pl_from_teacher_aug,
                interpolate=True,
            )
            loss_decode_u = {f"{k}_unlabeled": v * reweight_unsup * self.unsup_weight for k, v in loss_decode_u.items()}
            loss_decode.update(loss_decode_u)

        else:
            student_unlbld_backbone_feats, pl_from_teacher_aug, reweight_unsup = None, None, 1.0


        if self.proto_net:
            pretrain_proto = global_step < self.proto_pretrain_epochs * steps_per_epoch
            loss_decode.update(self.decode_proto_network(features_sup=student_lbld_backbone_feats,
                                                        gt_semantic_seg=masks,
                                                        img_metas=img_metas,
                                                        unlabeled_img_metas=unlabeled_img_metas,
                                                        pretrain_proto=pretrain_proto,
                                                        features_unsup=student_unlbld_backbone_feats,
                                                        pl_from_teacher=pl_from_teacher_aug,
                                                        reweight_unsup=reweight_unsup))

            return loss_decode

        return loss_decode

    def _generate_pseudo_labels(self, ul_w_img: Tensor, percent_unreliable: float) -> tuple[Tensor, Tensor]:
        """Generate pseudo labels from teacher model, apply filter loss method.

        Args:
            ul_w_img (torch.Tensor): weakly augmented unlabeled images
            ul_img_metas (list[ImageInfo]): unlabeled images meta data
            percent_unreliable (float): percent of unreliable pixels

        """
        with torch.no_grad():
            teacher_out = self.teacher_model(ul_w_img, mode="tensor")
            teacher_out = torch.nn.functional.interpolate(
                teacher_out,
                size=ul_w_img.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            teacher_prob_unsup = torch.softmax(teacher_out, axis=1)
            _, pl_from_teacher = torch.max(teacher_prob_unsup, axis=1, keepdim=True)

        # drop pixels with high entropy
        reweight_unsup = 1.0
        if percent_unreliable > 0:
            keep_percent = 100 - percent_unreliable
            batch_size, _, h, w = teacher_out.shape

            entropy = -torch.sum(teacher_prob_unsup * torch.log(teacher_prob_unsup + 1e-10), dim=1, keepdim=True)

            thresh = np.percentile(entropy[pl_from_teacher != 255].detach().cpu().numpy().flatten(), keep_percent)
            thresh_mask = entropy.ge(thresh).bool() * (pl_from_teacher != 255).bool()

            # mark as ignore index
            pl_from_teacher[thresh_mask] = 255
            # reweight unsupervised loss
            reweight_unsup = batch_size * h * w / torch.sum(pl_from_teacher != 255)

        return pl_from_teacher, reweight_unsup

    def decode_proto_network(
        self, features_sup, gt_semantic_seg, img_metas, unlabeled_img_metas=None, pretrain_proto=False, features_unsup=None, pl_from_teacher=None, reweight_unsup=1.0
    ):
        """Forward prototype network, compute proto loss.

        If there is no unsupervised part, only supervised loss will be computed.

        Args:
            features_sup (torch.Tensor): student backbone features from labeled images
            gt_semantic_seg (torch.Tensor): ground truth semantic segmentation label maps
            features_unsup (torch.Tensor): student backbone features from unlabeled images. Default: None
            pl_from_teacher (torch.Tensor): teacher generated pseudo labels. Default: None
            reweight_unsup (float): reweighting coefficient for unsupervised part after
                filtering high entropy pixels. Default: 1.0
        """
        # supervised branch
        proto_out_supervised = self.proto_net(inputs=features_sup, gt_semantic_seg=gt_semantic_seg, pretrain_prototype=pretrain_proto)
        valid_label_mask = self.student_model.get_valid_label_mask(img_metas)
        loss_proto = self.proto_net.calculate_loss(**proto_out_supervised, seg_label=gt_semantic_seg, valid_label_mask=valid_label_mask, interpolate=True)
        proto_losses = {"loss_proto": loss_proto}

        # unsupervised branch
        if not pretrain_proto and features_unsup is not None and pl_from_teacher is not None:
            proto_out_unsupervised = self.proto_net(features_unsup, pl_from_teacher, pretrain_proto)
            unlbld_valid_label_mask = self.student_model.get_valid_label_mask(unlabeled_img_metas)
            loss_proto_u = self.proto_net.calculate_loss(**proto_out_unsupervised, seg_label=pl_from_teacher, valid_label_mask=unlbld_valid_label_mask, interpolate=True)
            proto_losses.update({"loss_proto_unlabeled": loss_proto_u * reweight_unsup * self.unsup_weight})

        return proto_losses
