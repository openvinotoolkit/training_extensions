import torch
import torch.nn as nn
import torch.nn.functional as F
from open3d.ml.torch.layers import SparseConv, SparseConvTranspose

class SparseConvFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, cls, feat, in_pos, out_pos, voxel_size):
        kernel = cls.state_dict()["kernel"]
        offset = cls.state_dict()["offset"]
        kernel = g.op("Constant", value_t=kernel)
        offset = g.op("Constant", value_t=offset)
        return g.op("SparseConv", feat, in_pos, out_pos, kernel, offset)

    @staticmethod
    def forward(self, cls, feat, in_pos, out_pos, voxel_size):
        return cls.origin_forward(feat, in_pos, out_pos, voxel_size)


class SparseConvONNX(SparseConv):
    """
    This is a support class which helps export network with SparseConv in ONNX format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_forward = super().forward

    def forward(self, feat, in_pos, out_pos, voxel_size):
        return SparseConvFunc.apply(self, feat, in_pos, out_pos, voxel_size)


class SparseConvTransposeFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, cls, feat, in_pos, out_pos, voxel_size):
        kernel = cls.state_dict()["kernel"]
        offset = cls.state_dict()["offset"]
        kernel = g.op("Constant", value_t=kernel)
        offset = g.op("Constant", value_t=offset)
        return g.op("SparseConvTranspose", feat, in_pos, out_pos, kernel, offset)

    @staticmethod
    def forward(self, cls, feat, in_pos, out_pos, voxel_size):
        return cls.origin_forward(feat, in_pos, out_pos, voxel_size)


class SparseConvTransposeONNX(SparseConvTranspose):
    """
    This is a support class which helps export network with SparseConvTranspose in ONNX format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_forward = super().forward

    def forward(self, feat, in_pos, out_pos, voxel_size):
        return SparseConvTransposeFunc.apply(self, feat, in_pos, out_pos, voxel_size)
