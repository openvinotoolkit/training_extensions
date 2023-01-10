import torch
import torch.nn as nn

class ComplexMul(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input_tensor, other_tensor, is_conj = True):
        return g.op("ComplexMultiplication", input_tensor, other_tensor, is_conj_i=int(is_conj))

    @staticmethod
    def forward(self, input_tensor, other_tensor):
        complex_index = -1
        real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
        imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]

        multiplication = torch.cat(
            [
                real_part.unsqueeze(dim=complex_index),
                imaginary_part.unsqueeze(dim=complex_index),
            ],
            dim=complex_index,
        )
        return multiplication
