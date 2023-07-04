"""L2SP loss for mmdetection adapters."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner.checkpoint import _load_checkpoint
from mmdet.models import LOSSES
from torch import nn

# TODO: Need to fix pylint issues
# pylint: disable=unused-argument


@LOSSES.register_module()
class L2SPLoss(nn.Module):
    """L2-SP regularization Class for mmdetection adapter."""

    def __init__(self, model, model_ckpt, loss_weight=0.0001):
        """L2-SP regularization loss.

        Args:
            model (nn.Module): Input module to regularize
            model_ckpt (str): Starting-point model checkpoint
                Matched params in model would be regularized to be close to starting-point params
            loss_weight (float, optional): Weight of the loss. Defaults to 0.0001
        """
        super().__init__()
        self.model_ckpt = model_ckpt
        self.loss_weight = loss_weight
        print("L2SP loss init!")
        print(f" - starting-point: {model_ckpt}")

        if model_ckpt is None:
            raise ValueError("Model checkpoint path should be provided to enable L2-SP loss!")
        if loss_weight <= 0.0:
            raise ValueError("Loss weight should be a positive value!")

        # Load weights
        src_weights = _load_checkpoint(self.model_ckpt)
        if "state_dict" in src_weights:
            src_weights = src_weights["state_dict"]
        dst_weights = model.named_parameters()

        # Strip 'module.' from weight names if any
        src_weights = {k.replace("module.", ""): v for k, v in src_weights.items()}
        src_weights = {k.replace("model_s.", ""): v for k, v in src_weights.items()}
        # for name in src_weights:
        #    print(name)

        # Match weight name & shape
        self.l2_weights = []
        self.l2sp_weights = []
        for name, p in dst_weights:
            name = name.replace("module.", "")
            if not p.requires_grad:
                print(f"{name}: skip - no grad")
                continue

            if name in src_weights:
                src_w = src_weights[name]
            elif name.replace("backbone.", "") in src_weights:
                # In case of backbone only weights like ImageNet pretrained
                # -> names are stating w/o 'backbone.'
                src_w = src_weights[name.replace("backbone.", "")]
            else:
                self.l2_weights.append(p)
                print(f"{name}: l2 - new param")
                continue

            if p.shape != src_w.shape:
                # Same name but w/ different shape
                self.l2_weights.append(p)
                print(f"{name}: l2 - diff shape ({p.shape} vs {src_w.shape}")
                continue

            src_w.requires_grad = False
            self.l2sp_weights.append((p, src_w))
            print(f"{name}: l2sp")

    def forward(self, **kwargs):
        """Forward function.

        Returns:
            torch.Tensor: The calculated loss
        """

        # loss = torch.tensor(0.0, requires_grad=True)
        loss = 0.0
        # L2 loss
        for weight in self.l2_weights:
            loss += (weight**2).sum()
        # L2-SP loss
        for weight, weight_0 in self.l2sp_weights:
            loss += ((weight - weight_0.to(weight)) ** 2).sum()

        return self.loss_weight * loss
