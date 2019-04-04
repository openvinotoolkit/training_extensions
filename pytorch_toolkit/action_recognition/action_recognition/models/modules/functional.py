import torch


def squash_dims(tensor, dims):
    """
    Squashes dimension, given in dims into one, which equals to product of given.

    Args:
        tensor (Tensor): input tensor
        dims: dimensions over which tensor should be squashed

    """
    assert len(dims) >= 2, "Expected two or more dims to be squashed"

    size = tensor.size()

    squashed_dim = size[dims[0]]
    for i in range(1, len(dims)):
        assert dims[i] == dims[i - 1] + 1, "Squashed dims should be consecutive"
        squashed_dim *= size[dims[i]]

    result_dims = size[:dims[0]] + (squashed_dim,) + size[dims[-1] + 1:]
    return tensor.contiguous().view(*result_dims)


def unsquash_dim(tensor, dim, res_dim):
    """
    Unsquashes dimension, given in dim into separate dimensions given is res_dim
    Args:
        tensor (Tensor): input tensor
        dim (int): dimension that should be unsquashed
        res_dim (tuple): list of dimensions, that given dim should be unfolded to

    """
    size = tensor.size()
    result_dim = size[:dim] + res_dim + size[dim + 1:]
    return tensor.view(*result_dim)


def reduce_tensor(tensor, dims, reduction=torch.sum):
    """Performs reduction over multiple dimensions at once"""
    permute_idx = [i for i, d in enumerate(tensor.size()) if i not in dims]
    result_dims = [d for i, d in enumerate(tensor.size()) if i not in dims]
    tensor = tensor.permute(*(permute_idx + list(dims))).contiguous()
    return reduction(tensor.view(*result_dims, -1), -1)
