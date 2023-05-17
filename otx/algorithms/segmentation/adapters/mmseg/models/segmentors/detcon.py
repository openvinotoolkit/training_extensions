"""DetCon implementation for self-supervised learning.

Original papers:
- 'Efficient Visual Pretraining with Contrastive Detection', https://arxiv.org/abs/2103.10957
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=unused-argument, invalid-name, unnecessary-pass, not-callable

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint
from mmseg.models.builder import (  # pylint: disable=no-name-in-module
    SEGMENTORS,
    build_backbone,
    build_head,
    build_neck,
)
from mmseg.ops import resize
from torch import nn

from otx.algorithms.common.utils.logger import get_logger

from .otx_encoder_decoder import OTXEncoderDecoder

logger = get_logger()


class MaskPooling(nn.Module):
    """Mask pooling module to filter each class with the same class.

    Args:
        num_classes (int): The number of classes to be considered as pseudo classes. Default: 256.
        num_samples (int): The number of samples to be sampled. Default: 16.
        downsample (int): The ratio of the mask size to the feature size. Default: 32.
        replacement (bool): Whether samples are drawn with replacement or not.
            It can be used when `num_classes` is small rather than `num_samples`. Default: True.
    """

    def __init__(
        self,
        num_classes: int = 256,
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
        """Perform mask pooling and create binary masks.

        Args:
            masks (Tensor): Ground truth masks.

        Returns:
            Tensor: Pooled binary masks.
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
        """Samples of which binary masks to use in the loss.

        Args:
            masks (Tensor): Pooled binary masks from `self.pool_masks`.

        Returns:
            tuple[Tensor, Tensor]: (sampled_masks, mask_ids),
                sampled binary masks and ids used to sample masks.
        """
        assert masks.ndim == 3

        batch_size = masks.shape[0]
        mask_exists = torch.greater(masks.sum(dim=-1), 1e-3)
        sel_masks = mask_exists.to(torch.float) + 1e-11

        mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples, replacement=self.replacement)
        sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(batch_size)])

        return sampled_masks, mask_ids

    def forward(self, masks: torch.Tensor):
        """Forward function for mask pooling.

        Args:
            masks (Tensor): Ground truth masks to be sampled.

        Returns:
            tuple[Tensor, Tensor]: (sampled_masks, sampled_mask_ids),
                normalized sampled binary masks and ids used to sample masks.
        """
        binary_masks = self.pool_masks(masks)
        sampled_masks, sampled_mask_ids = self.sample_masks(binary_masks)
        areas = sampled_masks.sum(dim=-1, keepdim=True)
        sampled_masks = sampled_masks / torch.maximum(areas, torch.tensor(1.0, device=areas.device))

        return sampled_masks, sampled_mask_ids


# pylint: disable=too-many-arguments, dangerous-default-value, too-many-instance-attributes
@SEGMENTORS.register_module()
class DetConB(nn.Module):
    """DetCon Implementation.

    Implementation of 'Efficient Visual Pretraining with Contrastive Detection'
        (https://arxiv.org/abs/2103.10957).

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict, optional): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict, optional): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target network.
            Default: 0.996.
        num_classes (int): The number of classes to be considered as pseudo classes. Default: 256.
        num_samples (int): The number of samples to be sampled. Default: 16.
        downsample (int): The ratio of the mask size to the feature size. Default: 32.
        input_transform (str): Input transform of features from backbone. Default: "resize_concat".
        in_index (list): Feature index to be used for DetCon if the backbone outputs
            multi-scale features wrapped by list or tuple. Default: [0].
        align_corners (bool): Whether apply `align_corners` during resize. Default: False.
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        pretrained: Optional[str] = None,
        base_momentum: float = 0.996,
        num_classes: int = 256,
        num_samples: int = 16,
        downsample: int = 32,
        input_transform: str = "resize_concat",
        in_index: Union[List[int], int] = [0],
        align_corners: bool = False,
        **kwargs,
    ):
        super().__init__()

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
        self.predictor = build_head(head)

        # set maskpooling
        self.mask_pool = MaskPooling(num_classes, num_samples, downsample)

        self.init_weights(pretrained=pretrained)

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

    def transform_inputs(self, inputs: Union[List, Tuple]):
        """Transform inputs for decoder.

        Args:
            inputs (list, tuple): List (or tuple) of multi-level img features.

        Returns:
            Tensor: The transformed inputs.
        """
        # TODO (sungchul): consider tensor component, too
        if self.input_transform == "resize_concat" and isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select" and isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        else:
            if isinstance(self.in_index, (list, tuple)):
                self.in_index = self.in_index[0]
            inputs = inputs[self.in_index]  # type: ignore

        return inputs

    def extract_feat(self, img: torch.Tensor):
        """Extract features from images.

        Args:
            img (Tensor): Input image.

        Return:
            Tensor: Features from the online_backbone.
        """
        x = self.online_backbone(img)
        return x

    def sample_masked_feats(
        self,
        feats: Union[torch.Tensor, List, Tuple],
        masks: torch.Tensor,
        projector: nn.Module,
    ):
        """Sampled features from mask.

        Args:
            feats (list, tuple, Tensor): Features from the backbone.
            masks (Tensor): Ground truth masks to be sampled and to be used to filter `feats`.
            projector (nn.Module): Projector MLP.

        Returns:
            tuple[Tensor, Tensor]: (proj, sampled_mask_ids), features from the projector and ids used to sample masks.
        """
        if isinstance(feats, (list, tuple)) and len(feats) > 1:
            feats = self.transform_inputs(feats)

        # TODO (sungchul): consider self.input_transform == "multiple_select"
        sampled_masks, sampled_mask_ids = self.mask_pool(masks)

        b, c, h, w = feats.shape  # type: ignore
        feats = feats.reshape((b, c, h * w)).transpose(1, 2)  # type: ignore
        sampled_feats = sampled_masks @ feats
        sampled_feats = sampled_feats.reshape((-1, c))

        proj = projector(sampled_feats)

        return proj, sampled_mask_ids

    # pylint: disable=too-many-locals
    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        gt_semantic_seg: torch.Tensor,
        return_embedding: bool = False,
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): Input information.
            gt_semantic_seg (Tensor): Pseudo masks.
                It is used to organize features among the same classes.
            return_embedding (bool): Whether returning embeddings from the online backbone.
                It can be used for SupCon. Default: False.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.ndim == 5 and gt_semantic_seg.ndim == 5
        batch_size = img.shape[0]
        imgs = torch.cat((img[:, 0], img[:, 1]), dim=0)
        masks = torch.cat((gt_semantic_seg[:, :, 0], gt_semantic_seg[:, :, 1]), dim=0)

        embds = self.online_backbone(imgs)
        projs, ids = self.sample_masked_feats(embds, masks, self.online_projector)

        with torch.no_grad():
            self._momentum_update()
            projs_tgt, ids_tgt = self.sample_masked_feats(self.target_backbone(imgs), masks, self.target_projector)

        # predictor
        loss = self.predictor(projs, projs_tgt, ids, ids_tgt, batch_size, self.num_samples)

        if return_embedding:
            return loss, embds, masks
        return loss

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        raise AttributeError("Self-SL doesn't support `forward_test` for evaluation.")

    def train_step(
        self,
        data_batch: Dict[str, Any],
        optimizer: Union[torch.optim.Optimizer, Dict],
        **kwargs,
    ):
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
            **kwargs (Any): Addition keyword arguments.

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

    def val_step(self, **kwargs):
        """Disenable validation step during self-supervised learning."""
        pass

    def _parse_losses(self, losses: Dict[str, Any]):
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
        """`set_step_params` to be skipped."""
        pass

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Save only online backbone as output state_dict."""
        logger.info("----------------- BYOL.state_dict_hook() called")
        output = OrderedDict()
        for k, v in state_dict.items():
            if "online_backbone." in k:
                k = k.replace("online_backbone.", "backbone.")
                output[k] = v
        return output


# pylint: disable=too-many-locals
@SEGMENTORS.register_module()
class SupConDetConB(OTXEncoderDecoder):  # pylint: disable=too-many-ancestors
    """Apply DetConB as a contrastive part of `Supervised Contrastive Learning` (https://arxiv.org/abs/2004.11362).

    SupCon with DetConB uses ground truth masks instead of pseudo masks to organize features among the same classes.

    Args:
        decode_head (dict, optional): Config dict for module of decode head. Default: None.
        train_cfg (dict, optional): Config dict for training. Default: None.
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        decode_head: Optional[Dict[str, Any]] = None,
        neck: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        pretrained: Optional[str] = None,
        base_momentum: float = 0.996,
        num_classes: int = 256,
        num_samples: int = 16,
        downsample: int = 32,
        input_transform: str = "resize_concat",
        in_index: Union[List[int], int] = [0],
        align_corners: bool = False,
        train_cfg: Optional[Dict[str, Any]] = None,
        test_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs,
        )

        self.detconb = DetConB(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            base_momentum=base_momentum,
            num_classes=num_classes,
            num_samples=num_samples,
            downsample=downsample,
            input_transform=input_transform,
            in_index=in_index,
            align_corners=align_corners,
            **kwargs,
        )
        self.backbone = self.detconb.online_backbone
        # TODO (sungchul): Is state_dict_hook needed to save segmentor only?
        # 1. use state_dict_hook : we can save memory as only saving backbone + decode_head.
        # 2. save all : we can use additional training with the whole weights (backbone + decode_head + detcon).

    # pylint: disable=arguments-renamed
    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg,
        **kwargs,
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): Input information.
            gt_semantic_seg (Tensor): Ground truth masks.
                It is used to organize features among the same classes.
            **kwargs (Any): Addition keyword arguments.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = {}
        if img.ndim == 4:
            # supervised learning with interval
            embds = self.detconb.online_backbone(img)
            masks = gt_semantic_seg
        else:
            # supcon training
            loss_detcon, embds, masks = self.detconb.forward_train(
                img=img,
                img_metas=img_metas,
                gt_semantic_seg=gt_semantic_seg,
                return_embedding=True,
            )
            losses.update(dict(loss_detcon=loss_detcon["loss"]))
            img_metas += img_metas

        # decode head
        loss_decode = self._decode_head_forward_train(embds, img_metas, gt_semantic_seg=masks)
        losses.update(loss_decode)

        return losses
