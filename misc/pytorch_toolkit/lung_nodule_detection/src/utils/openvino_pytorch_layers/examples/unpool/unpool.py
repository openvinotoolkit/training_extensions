import torch
import torch.nn as nn

class Unpool2d(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, indices, output_size=None):
        if output_size:
            return g.op('Unpooling', x, indices, output_size)
        else:
            return g.op('Unpooling', x, indices)

    @staticmethod
    def forward(self, x, indices, output_size=None):
        if not output_size is None:
            return nn.MaxUnpool2d(2, stride=2)(x, indices, output_size=output_size.size())
        else:
            return nn.MaxUnpool2d(2, stride=2)(x, indices)
