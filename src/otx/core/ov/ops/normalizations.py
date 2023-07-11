"""Normalization-related modules for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.op import Attribute, Operation
from otx.core.ov.ops.poolings import AvgPoolV1


@dataclass
class BatchNormalizationV0Attribute(Attribute):
    """BatchNormalizationV0Attribute class."""

    epsilon: float
    max_init_iter: int = field(default=2)


@OPS.register()
class BatchNormalizationV0(Operation[BatchNormalizationV0Attribute]):
    """BatchNormalizationV0 class."""

    TYPE = "BatchNormInference"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = BatchNormalizationV0Attribute

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_num_init_iter", torch.tensor(0))

    def forward(self, inputs, gamma, beta, mean, variance):
        """BatchNormalizationV0's forward function."""

        output = F.batch_norm(
            input=inputs,
            running_mean=mean,
            running_var=variance,
            weight=gamma,
            bias=beta,
            training=self.training,
            momentum=0.1,
            eps=self.attrs.epsilon,
        )

        if self.training and self._num_init_iter < self.attrs.max_init_iter:
            # no parameters update for adaptive phase
            with torch.no_grad():
                n_dims = inputs.dim() - 2
                gamma = gamma.unsqueeze(0)
                beta = beta.unsqueeze(0)
                for _ in range(n_dims):
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
                output = inputs * gamma + beta
                self._num_init_iter += 1
                if self._num_init_iter >= self.attrs.max_init_iter:
                    # Adapt weight & bias using the first batch statistics
                    # to undo normalization approximately
                    gamma.data = gamma.data * mean
                    beta.data = beta.data + (mean / (variance + self.attrs.epsilon))

        return output


@dataclass
class LocalResponseNormalizationV0Attribute(Attribute):
    """LocalResponseNormalizationV0Attribute class."""

    alpha: float
    beta: float
    bias: float
    size: int


@OPS.register()
class LocalResponseNormalizationV0(Operation[LocalResponseNormalizationV0Attribute]):
    """LocalResponseNormalizationV0 class."""

    TYPE = "LRN"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = LocalResponseNormalizationV0Attribute

    def forward(self, inputs, axes):
        """LocalResponseNormalizationV0's forward function."""
        dim = inputs.dim()

        axes = axes.detach().cpu().tolist()
        assert all(ax >= 1 for ax in axes)

        axes = [ax - 1 for ax in axes]
        kernel = [1 for _ in range(dim - 1)]
        stride = [1 for _ in range(dim - 1)]
        pads_begin = [0 for _ in range(dim - 1)]
        pads_end = [0 for _ in range(dim - 1)]
        for axe in axes:
            kernel[axe] = self.attrs.size
            pads_begin[axe] = self.attrs.size // 2
            pads_end[axe] = (self.attrs.size - 1) // 2

        avg_attrs = {
            "auto_pad": "explicit",
            "strides": stride,
            "kernel": kernel,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "exclude-pad": True,
            "shape": self.shape,
        }
        avg_pool = AvgPoolV1("temp", **avg_attrs)

        div = inputs.mul(inputs).unsqueeze(1)
        div = avg_pool(div)
        div = div.squeeze(1)
        div = div.mul(self.attrs.alpha).add(self.attrs.bias).pow(self.attrs.beta)
        output = inputs / div
        return output


@dataclass
class NormalizeL2V0Attribute(Attribute):
    """NormalizeL2V0Attribute class."""

    eps: float
    eps_mode: str

    def __post_init__(self):
        """NormalizeL2V0Attribute post-init function."""
        super().__post_init__()
        valid_eps_mode = ["add", "max"]
        if self.eps_mode not in valid_eps_mode:
            raise ValueError(f"Invalid eps_mode {self.eps_mode}. " f"It must be one of {valid_eps_mode}.")


@OPS.register()
class NormalizeL2V0(Operation[NormalizeL2V0Attribute]):
    """NormalizeL2V0 class."""

    TYPE = "NormalizeL2"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = NormalizeL2V0Attribute

    def forward(self, inputs, axes):
        """NormalizeL2V0's forward function."""
        eps = self.attrs.eps
        eps_mode = self.attrs.eps_mode

        if isinstance(axes, torch.Tensor):
            axes = axes.detach().cpu().tolist()
        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        # normalization layer convert to FP32 in FP16 training
        input_float = inputs.float()
        if axes:
            norm = input_float.pow(2).sum(axes, keepdim=True)
        else:
            norm = input_float

        if eps_mode == "add":
            norm = norm + eps
        elif eps_mode == "max":
            norm = torch.clamp(norm, max=eps)

        return (input_float / norm.sqrt()).type_as(inputs)


@dataclass
class MVNV6Attribute(Attribute):
    """MVNV6Attribute class."""

    normalize_variance: bool
    eps: float
    eps_mode: str

    def __post_init__(self):
        """MVNV6Attribute's post-init function."""
        super().__post_init__()
        valid_eps_mode = ["INSIDE_SQRT", "OUTSIDE_SQRT"]
        if self.eps_mode not in valid_eps_mode:
            raise ValueError(f"Invalid eps_mode {self.eps_mode}. " f"It must be one of {valid_eps_mode}.")


@OPS.register()
class MVNV6(Operation[MVNV6Attribute]):
    """MVNV6 class."""

    TYPE = "MVN"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = MVNV6Attribute

    def forward(self, inputs, axes):
        """MVNV6's forward function."""
        output = inputs - inputs.mean(axes.tolist(), keepdim=True)
        if self.attrs.normalize_variance:
            eps_mode = self.attrs.eps_mode
            eps = self.attrs.eps
            var = torch.square(output).mean(axes.tolist(), keepdim=True)
            if eps_mode == "INSIDE_SQRT":
                output = output / torch.sqrt(var + eps)
            elif eps_mode == "OUTSIDE_SQRT":
                output = output / (torch.sqrt(var) + eps)
        return output
