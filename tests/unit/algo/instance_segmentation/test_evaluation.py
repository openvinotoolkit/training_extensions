import torch
from otx.core.metrics.mean_ap import MaskRLEMeanAveragePrecision
from otx.core.utils.mask_util import encode_rle
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def test_custom_rle_map_metric(num_masks=50, h=10, w=10):
    """Test custom RLE MAP metric."""
    custom_map_metric = MaskRLEMeanAveragePrecision(iou_type="segm")
    torch_map_metric = MeanAveragePrecision(iou_type="segm")

    # Create random masks
    pred_masks = torch.randint(low=0, high=2, size=(num_masks, h, w)).bool()
    target_masks = torch.randint(low=0, high=2, size=(num_masks, h, w)).bool()
    labels = torch.zeros(num_masks, dtype=torch.long)
    scores = torch.rand(num_masks)

    torch_map_metric.update(
        preds=[{"masks": pred_masks, "labels": labels, "scores": scores}],
        target=[{"masks": target_masks, "labels": labels}],
    )

    custom_map_metric.update(
        preds=[{"masks": [encode_rle(pred) for pred in pred_masks], "labels": labels, "scores": scores}],
        target=[{"masks": [encode_rle(target) for target in target_masks], "labels": labels}],
    )

    # Compare the results
    torch_results = torch_map_metric.compute()
    custom_results = custom_map_metric.compute()

    assert custom_results == torch_results, f"Expected {torch_results} but got {custom_results}"
