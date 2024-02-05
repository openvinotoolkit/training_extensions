import numpy as np
import torch
from otx.core.utils.mask_util import encode_rle
from pycocotools import mask as mask_utils


def test_encode_rle(num_test_cases=30):
    """Test encode_rle function.

    Args:
        num_test_cases (int, optional): number of test cases. Defaults to 30.
    """
    for _ in range(num_test_cases):
        h, w = torch.randint(low=1, high=800, size=(2,))
        mask = torch.randint(low=0, high=2, size=(h, w)).bool()
        torch_rle = encode_rle(mask)
        torch_rle = mask_utils.frPyObjects(torch_rle, *torch_rle["size"])
        np_rle = mask_utils.encode(np.asfortranarray(mask.numpy()))
        assert torch_rle["counts"] == np_rle["counts"], f"Expected {np_rle['counts']} but got {torch_rle['counts']}"
        assert torch_rle["size"] == np_rle["size"], f"Expected {np_rle['size']} but got {torch_rle['size']}"
