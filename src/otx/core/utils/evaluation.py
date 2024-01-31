from typing import Any, Dict, Tuple
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pycocotools.mask as mask_utils
import torch



class OTXInstSegMeanAveragePrecision(MeanAveragePrecision):

    def encode_rle(self, mask):
        rle = {"counts": [], "size": list(mask.shape)}
        device = mask.device
        flattened_mask = mask.t().ravel()
        diff_arr = torch.diff(flattened_mask)
        nonzero_indices = torch.where(diff_arr != 0)[0] + 1

        lengths = torch.diff(torch.cat(
            (torch.tensor([0], device=device),
             nonzero_indices,
             torch.tensor([len(flattened_mask)], device=device)),
        ))

        # note that the odd counts are always the numbers of zeros
        if flattened_mask[0] == 1:
            lengths = torch.cat((torch.tensor([0], device=device), lengths))

        rle["counts"] = list(lengths.cpu().numpy())
        return rle

    def _get_safe_item_values(self, item: Dict[str, Any]) -> Tensor | Tuple:
        if self.iou_type == "segm":
            masks = []
            for mask in item["masks"]:
                rle = self.encode_rle(mask)
                rle = mask_utils.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
                masks.append((tuple(rle["size"]), rle["counts"]))
            return tuple(masks)
        else:
            raise Exception(f"IOU type {self.iou_type} is not supported")