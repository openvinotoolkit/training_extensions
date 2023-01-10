import torch
import torch.nn as nn
import torch.nn.functional as F

class CalculateGrid(torch.autograd.Function):
    @staticmethod
    def symbolic(g, in_positions):
        return g.op("CalculateGrid", in_positions)

    @staticmethod
    def forward(self, in_positions):
        filter = torch.Tensor([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1], [-1, 0, 0],
                               [0, -1, -1], [0, -1, 0], [0, 0, -1],
                               [0, 0, 0]]).to(in_positions.device)

        out_pos = in_positions.long().repeat(1, filter.shape[0]).reshape(-1, 3)
        filter = filter.repeat(in_positions.shape[0], 1)

        out_pos = out_pos + filter
        out_pos = out_pos[out_pos.min(1).values >= 0]
        out_pos = out_pos[(~((out_pos.long() % 2).bool()).any(1))]
        out_pos = torch.unique(out_pos, dim=0)

        return out_pos + 0.5
