import torch
from packaging import version
from typing import List, Tuple, Union

def roll(
    data: torch.Tensor,
    shift: Union[int, Union[Tuple[int, ...], List[int]]],
    dims: Union[int, Union[Tuple, List]],
) -> torch.Tensor:
    """
    Similar to numpy roll but applies to pytorch tensors.
    Parameters
    ----------
    data : torch.Tensor
    shift: tuple, int
    dims : tuple, list or int

    Returns
    -------
    torch.Tensor
    """
    if isinstance(shift, (tuple, list)) and isinstance(dims, (tuple, list)):
        if len(shift) != len(dims):
            raise ValueError(f"Length of shifts and dimensions should be equal. Got {len(shift)} and {len(dims)}.")
        for curr_shift, curr_dim in zip(shift, dims):
            data = roll(data, curr_shift, curr_dim)
        return data
    dim_index = dims
    shift = shift % data.size(dims)

    if shift == 0:
        return data
    left_part = data.narrow(dim_index, 0, data.size(dims) - shift)
    right_part = data.narrow(dim_index, data.size(dims) - shift, shift)
    return torch.cat([right_part, left_part], dim=dim_index)

def fftshift(data: torch.Tensor, dims) -> torch.Tensor:
    shift = [data.size(curr_dim) // 2 for curr_dim in dims]
    return roll(data, shift, dims)

def ifftshift(data: torch.Tensor, dims) -> torch.Tensor:
    shift = [(data.size(curr_dim) + 1) // 2 for curr_dim in dims]
    return roll(data, shift, dims)

class FFT(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, inverse, centered, dims):
        dims = torch.tensor(dims)
        dims = g.op("Constant", value_t=dims)

        return g.op('FFT', x, dims, inverse_i=inverse, centered_i=centered)

    @staticmethod
    def forward(self, x, inverse, centered, dims):
        # https://pytorch.org/docs/stable/torch.html#torch.fft
        if centered:
            x = ifftshift(x, dims)

        if version.parse(torch.__version__) >= version.parse("1.8.0"):
            func = torch.fft.ifftn if inverse else torch.fft.fftn
            x = torch.view_as_complex(x)
            y = func(x, dim=dims, norm="ortho")
            y = torch.view_as_real(y)
        else:
            signal_ndim = max(dims)
            assert dims == list(range(1, signal_ndim + 1))
            func = torch.ifft if inverse else torch.fft
            y = func(input=x, signal_ndim=signal_ndim, normalized=True)

        if centered:
            y = fftshift(y, dims)

        return y
