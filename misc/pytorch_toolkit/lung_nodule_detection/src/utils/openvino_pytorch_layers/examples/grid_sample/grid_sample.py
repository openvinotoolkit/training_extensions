import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSample(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, grid):
        return g.op('GridSample', x, grid)

    @staticmethod
    def forward(self, x, grid):
        return F.grid_sample(x, grid, 'bilinear', 'zeros', True)
