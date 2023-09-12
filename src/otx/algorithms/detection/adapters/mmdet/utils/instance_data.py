"""Data structure for instance-level annnotations or predictions."""

import numpy as np
import torch
from mmdet.core import InstanceData


class CustomInstanceData(InstanceData):
    """Data structure for instance-level annnotations or predictions."""

    def __getitem__(self, item):
        """Get the corresponding values according to item.

        Args:
            item: get the corresponding values according to item.

        Returns:
            obj:`CustomInstanceData`: Corresponding values.
        """
        assert len(self), " This is a empty instance"

        assert isinstance(item, (str, slice, int, torch.LongTensor, torch.BoolTensor, torch.Tensor))

        if isinstance(item, str):
            return getattr(self, item)

        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError(f"Index {item} out of range!")
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.new()
        if isinstance(item, (torch.Tensor)):
            assert item.dim() == 1, "Only support to get the" " values along the first dimension."
            if isinstance(item, torch.BoolTensor):
                assert len(item) == len(self), (
                    f"The shape of the"
                    f" input(BoolTensor)) "
                    f"{len(item)} "
                    f" does not match the shape "
                    f"of the indexed tensor "
                    f"in results_filed "
                    f"{len(self)} at "
                    f"first dimension. "
                )

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, list):
                    r_list = []
                    # convert to indexes from boolTensor
                    if isinstance(item, torch.BoolTensor):
                        indexes = torch.nonzero(item).view(-1)
                    else:
                        indexes = item
                    for index in indexes:
                        r_list.append(v[index])
                    new_data[k] = r_list
        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data
