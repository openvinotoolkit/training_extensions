import torch
import torch.nn as nn
import torchvision.ops as ops


class DeformableConvFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, cls, x, offset):
        weight = cls.state_dict()["weight"]
        weight = g.op("Constant", value_t=weight)

        return g.op(
            "DeformableConv2D",
            x,
            offset,
            weight,
            strides_i=(cls.stride, cls.stride),
            pads_i=(cls.padding, cls.padding, cls.padding, cls.padding),
            dilations_i=(cls.dilation, cls.dilation),
            deformable_group_i=cls.groups,
        )

    @staticmethod
    def forward(self, cls, x, offset):
        y = cls.origin_forward(x, offset)
        return y


class DeformableConvolution(ops.DeformConv2d):
    """
    This is a support class which helps export network with SparseConv in ONNX format.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_forward = super().forward
        self.stride = kwargs.get("stride", 1)
        self.padding = kwargs.get("padding", 0)
        self.dilation = kwargs.get("dilation", 1)
        self.groups = kwargs.get("groups", 1)
        self.pad_l = nn.ConstantPad2d((1, 1, 1, 1), 0)

    def forward(self, x, offset):
        """
        Using paddings is a workaround for 2021.4 release.
        """
        x = self.pad_l(x)
        offset = self.pad_l(offset)
        y = DeformableConvFunc.apply(self, x, offset)
        y = y[:, :, 1:-1, 1:-1]
        return y
