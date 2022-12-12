# pylint: disable=missing-module-docstring, too-many-instance-attributes, unused-argument
from collections import OrderedDict
from typing import Dict, Optional, Any

import torch
import torch.distributed as dist
from torch import nn
from mmcls.models.builder import CLASSIFIERS, build_backbone, build_head, build_neck
from mpa.utils.logger import get_logger

logger = get_logger()


@CLASSIFIERS.register_module()
class BYOL(nn.Module):
    """
    Implementation of "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning (https://arxiv.org/abs/2006.07733)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target network.
            Default: 0.996.
    """
    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        pretrained: Optional[str] = None,
        base_momentum: float = 0.996,
        **kwargs # pylint: disable=unused-argument
    ):

        super().__init__()

        # build backbone
        self.online_backbone = self.backbone = build_backbone(backbone)
        self.target_backbone = build_backbone(backbone)

        # build projector
        self.online_projector = build_neck(neck)
        self.target_projector = build_neck(neck)

        # build head with predictor
        self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self, pretrained: Optional[str] = None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            logger.info(f"load model from: {pretrained}")

        # init backbone
        self.online_backbone.init_weights(pretrained=pretrained)
        for param_ol, param_tgt in zip(self.online_backbone.parameters(),
                                       self.target_backbone.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False
            param_ol.requires_grad = True

        # init projector
        self.online_projector.init_weights(init_linear="kaiming")
        for param_ol, param_tgt in zip(self.online_projector.parameters(),
                                       self.target_projector.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False
            param_ol.requires_grad = True

        # init the predictor
        self.head.init_weights()

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_backbone.parameters(),
                                       self.target_backbone.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

        for param_ol, param_tgt in zip(self.online_projector.parameters(),
                                       self.target_projector.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network."""
        self._momentum_update()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, **kwargs):
        """Forward computation during training.

        Args:
            img1 (Tensor): Input of two concatenated images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img2 (Tensor): Input of two concatenated images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        proj_1 = self.online_projector(self.online_backbone(img1))
        proj_2 = self.online_projector(self.online_backbone(img2))
        with torch.no_grad():
            proj_1_tgt = self.target_projector(self.target_backbone(img1)).clone().detach()
            proj_2_tgt = self.target_projector(self.target_backbone(img2)).clone().detach()

        loss = self.head(proj_1, proj_2_tgt)["loss"] + self.head(proj_2, proj_1_tgt)["loss"]
        return dict(loss=loss)

    def train_step(self, data: Dict[str, Any], optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
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
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data["img1"].data))

        return outputs

    # pylint: disable=missing-function-docstring
    def val_step(self, *args):
        pass

    def _parse_losses(self, losses: Dict[str, Any]):
        """Parse loss dictionary.

        Args:
            losses (dict): ...

        Returns:
            tuple: ...
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items()
                   if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
