"""DetCon implementation for self-supervised learning.

Original papers:
- 'Efficient Visual Pretraining with Contrastive Detection', https://arxiv.org/abs/2103.10957
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmseg.models.builder import SEGMENTORS, build_backbone, build_loss, build_neck
from mmseg.ops import resize
from mpa.utils.logger import get_logger

logger = get_logger()


class MaskPooling(nn.Module):
    """
    Args:
        num_classes:
        num_samples:
        downsample:
        replacement (bool): whether samples are drawn with replacement or not. Default is True.
    """

    def __init__(
        self,
        num_classes: int,
        num_samples: int = 16,
        downsample: int = 32,
        replacement: bool = True,
    ):

        super().__init__()

        self.num_classes = num_classes
        self.num_samples = num_samples
        self.replacement = replacement

        self.mask_ids = torch.arange(num_classes)
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)

    def pool_masks(self, masks: torch.Tensor):
        """Create binary masks and performs mask pooling
        Args:
            masks: (b, 1, h, w)
        Returns:
            masks: (b, num_classes, d)
        """

        if masks.ndim < 4:
            masks = masks.unsqueeze(dim=1)

        masks = masks == self.mask_ids[None, :, None, None].to(masks.device)
        masks = self.pool(masks.to(torch.float))

        b, c, h, w = masks.shape
        masks = torch.reshape(masks, (b, c, h * w))
        masks = torch.argmax(masks, dim=1)
        masks = torch.eye(self.num_classes).to(masks.device)[masks]
        masks = torch.transpose(masks, 1, 2)

        return masks

    def sample_masks(self, masks: torch.Tensor):
        """Samples which binary masks to use in the loss.
        Args:
            masks: (b, num_classes, d)
        Returns:
            masks: (b, num_samples, d)
        """

        batch_size = masks.shape[0]
        mask_exists = torch.greater(masks.sum(dim=-1), 1e-3)
        sel_masks = mask_exists.to(torch.float) + 1e-11

        mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples, replacement=self.replacement)
        sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(batch_size)])

        return sampled_masks, mask_ids

    def forward(self, masks: List[torch.Tensor]):
        """
        Args:
            masks: [mask1, mask2]
        Returns:
            sampled_masks: [sampled_mask1, sampled_mask2]
            sampled_mask_ids: [sampled_mask_ids1, sampled_mask_ids2]
        """

        binary_masks = self.pool_masks(masks)
        sampled_masks, sampled_mask_ids = self.sample_masks(binary_masks)
        areas = sampled_masks.sum(dim=-1, keepdim=True)
        sampled_masks = sampled_masks / torch.maximum(areas, torch.tensor(1.0, device=areas.device))

        return sampled_masks, sampled_mask_ids


@SEGMENTORS.register_module()
class DetConB(nn.Module):
    def __init__(
        self,
        backbone,
        neck=None,
        head=None,
        pretrained=None,
        base_momentum: float = 0.996,
        num_classes: int = 256,
        num_samples: int = 16,
        downsample: int = 32,
        input_transform: str = "resize_concat",
        in_index: List[int] = [],
        align_corners: bool = False,
        loss_cfg: Dict[str, Union[str, float]] = {},
        **kwargs,
    ):
        super(DetConB, self).__init__()

        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.downsample = downsample
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        # build backbone
        self.online_backbone = build_backbone(backbone)
        self.target_backbone = build_backbone(backbone)

        # build projector
        self.online_projector = build_neck(neck)
        self.target_projector = build_neck(neck)

        # build head with predictor
        self.predictor = build_neck(head)

        # set maskpooling
        self.mask_pool = MaskPooling(num_classes, num_samples, downsample)

        self.init_weights(pretrained=pretrained)

        # build detcon loss
        self.detcon_loss = build_loss(loss_cfg)

        # Hooks for super_type transparent weight save
        self._register_state_dict_hook(self.state_dict_hook)

    def init_weights(self, pretrained: Optional[str] = None):
        """Initialize the weights of model.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """

        if pretrained is not None:
            logger.info(f"load model from: {pretrained}")
            load_checkpoint(
                self.online_backbone,
                pretrained,
                strict=False,
                map_location=None,
                logger=logger,
                revise_keys=[(r"^backbone\.", "")],
            )

        # init backbone
        for param_ol, param_tgt in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False
            param_ol.requires_grad = True

        # init projector
        self.online_projector.init_weights(init_linear="kaiming")
        for param_ol, param_tgt in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False
            param_ol.requires_grad = True

        # init the predictor
        self.predictor.init_weights()

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""

        for param_ol, param_tgt in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + param_ol.data * (1.0 - self.momentum)

        for param_ol, param_tgt in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + param_ol.data * (1.0 - self.momentum)

    def get_transformed_features(self, x):
        if self.input_transform:
            return self.transform_inputs(x)
        elif isinstance(self.in_index, int):
            return x[self.in_index]
        else:
            raise ValueError()

    def transform_inputs(self, inputs: List[torch.Tensor]):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def extract_feat(self, img: torch.Tensor):
        """Extract features from images."""

        x = self.backbone(img)

        return x

    def sample_masked_feats(self, feats: Union[torch.Tensor, List, Tuple], masks: torch.Tensor, projector: nn.Module):
        """Sampled features from mask.

        Args:
            feats (Tensor):
            masks (Tensor):
            projector (nn.Module):
        """
        if isinstance(feats, (list, tuple)) and len(feats) > 1:
            feats = self.get_transformed_features(feats)

        sampled_masks, sampled_mask_ids = self.mask_pool(masks)

        b, c, h, w = feats.shape
        feats = feats.reshape((b, c, h * w)).transpose(1, 2)
        sampled_feats = sampled_masks @ feats
        sampled_feats = sampled_feats.reshape((-1, c))

        proj = projector(sampled_feats)

        return proj, sampled_mask_ids

    def forward(self, img: torch.Tensor, img_metas: List[Dict], gt_semantic_seg: torch.Tensor):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert img.ndim == 5 and gt_semantic_seg.ndim == 4
        img1, img2 = img[:, 0], img[:, 1]
        mask1, mask2 = gt_semantic_seg[:, 0], gt_semantic_seg[:, 1]

        proj1, id1 = self.sample_masked_feats(self.online_backbone(img1), mask1, self.online_projector)
        proj2, id2 = self.sample_masked_feats(self.online_backbone(img2), mask2, self.online_projector)

        with torch.no_grad():
            self._momentum_update()
            proj1_tgt, id1_tgt = self.sample_masked_feats(self.target_backbone(img1), mask1, self.target_projector)
            proj2_tgt, id2_tgt = self.sample_masked_feats(self.target_backbone(img2), mask2, self.target_projector)

        # predictor
        pred1, pred2 = self.predictor(proj1), self.predictor(proj2)
        pred1 = pred1.reshape((-1, self.num_samples, pred1.shape[-1]))
        pred2 = pred2.reshape((-1, self.num_samples, pred2.shape[-1]))
        proj1_tgt = proj1_tgt.reshape((-1, self.num_samples, proj1_tgt.shape[-1]))
        proj2_tgt = proj2_tgt.reshape((-1, self.num_samples, proj2_tgt.shape[-1]))

        # decon loss
        loss = self.detcon_loss(
            pred1=pred1,
            pred2=pred2,
            target1=proj1_tgt,
            target2=proj2_tgt,
            pind1=id1,
            pind2=id2,
            tind1=id1_tgt,
            tind2=id2_tgt,
        )

        return loss

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch["img_metas"]))

        return outputs

    def val_step(self, data_batch, **kwargs):
        pass

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for var_name, var_value in losses.items():
            if isinstance(var_value, torch.Tensor):
                log_vars[var_name] = var_value.mean()
            elif isinstance(var_value, list):
                log_vars[var_name] = sum(_loss.mean() for _loss in var_value)
            elif isinstance(var_value, (int, float)):
                log_vars[var_name] = var_value
            else:
                raise TypeError(f"{var_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for var_name, var_value in log_vars.items():
            if isinstance(var_value, (int, float)):
                continue

            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                var_value = var_value.data.clone()
                dist.all_reduce(var_value.div_(dist.get_world_size()))

            log_vars[var_name] = var_value.item()

        return loss, log_vars

    def set_step_params(self, init_iter, epoch_size):
        pass

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Save only online backbone as output state_dict."""
        logger.info("----------------- BYOL.state_dict_hook() called")
        output = OrderedDict()
        for k, v in state_dict.items():
            if "online_backbone." in k:
                k = k.replace("online_backbone.", "")
                output[k] = v
        return output
