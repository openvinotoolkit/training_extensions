"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from otx.algo.instance_segmentation.mmdet.structures.bbox import BaseBoxes, get_box_type

if TYPE_CHECKING:
    from otx.algo.instance_segmentation.mmdet.models.utils import OptInstanceList
    from otx.algo.instance_segmentation.mmdet.structures import SampleList


def stack_boxes(data_list: list[Tensor | BaseBoxes], dim: int = 0) -> Tensor | BaseBoxes:
    """Stack boxes with type of tensor or box type.

    Args:
        data_list (List[Union[Tensor, :obj:`BaseBoxes`]]): A list of tensors
            or box types need to be stacked.
            dim (int): The dimension over which the box are stacked.
                Defaults to 0.

    Returns:
        Union[Tensor, :obj`BaseBoxes`]: Stacked results.
    """
    if data_list and isinstance(data_list[0], BaseBoxes):
        return data_list[0].stack(data_list, dim=dim)
    else:
        return torch.stack(data_list, dim=dim)


def samplelist_boxtype2tensor(batch_data_samples: SampleList):
    """Convert the box type in SampleList to tensor."""
    for data_samples in batch_data_samples:
        if "gt_instances" in data_samples:
            bboxes = data_samples.gt_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.gt_instances.bboxes = bboxes.tensor
        if "pred_instances" in data_samples:
            bboxes = data_samples.pred_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.pred_instances.bboxes = bboxes.tensor
        if "ignored_instances" in data_samples:
            bboxes = data_samples.ignored_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.ignored_instances.bboxes = bboxes.tensor


def images_to_levels(target: list[Tensor | BaseBoxes], num_levels: list[int]):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = stack_boxes(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size count)."""
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f"Only supports dict or list or Tensor, but get {type(results)}.")
    return scores, labels, keep_idxs, filtered_results


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id].detach() for i in range(num_levels)]
    else:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list


def unpack_gt_instances(batch_data_samples: SampleList) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore`` and ``img_metas`` based on ``batch_data_samples.``

    Args:
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        tuple:

            - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_gt_instances_ignore (list[:obj:`InstanceData`]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if "ignored_instances" in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)

    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas


def empty_instances(
    batch_img_metas: list[dict],
    device: torch.device,
    task_type: str,
    instance_results: OptInstanceList = None,
    mask_thr_binary: int | float = 0,
    box_type: str | type = "hbox",
    use_box_type: bool = False,
    num_classes: int = 80,
    score_per_cls: bool = False,
) -> list[InstanceData]:
    """Handle predicted instances when RoI is empty.

    Note: If ``instance_results`` is not None, it will be modified
    in place internally, and then return ``instance_results``

    Args:
        batch_img_metas (list[dict]): List of image information.
        device (torch.device): Device of tensor.
        task_type (str): Expected returned task type. it currently
            supports bbox and mask.
        instance_results (list[:obj:`InstanceData`]): List of instance
            results.
        mask_thr_binary (int, float): mask binarization threshold.
            Defaults to 0.
        box_type (str or type): The empty box type. Defaults to `hbox`.
        use_box_type (bool): Whether to warp boxes with the box type.
            Defaults to False.
        num_classes (int): num_classes of bbox_head. Defaults to 80.
        score_per_cls (bool):  Whether to generate classwise score for
            the empty instance. ``score_per_cls`` will be True when the model
            needs to produce raw results without nms. Defaults to False.

    Returns:
        list[:obj:`InstanceData`]: Detection results of each image
    """
    assert task_type in ("bbox", "mask"), "Only support bbox and mask," f" but got {task_type}"

    if instance_results is not None:
        assert len(instance_results) == len(batch_img_metas)

    results_list = []
    for img_id in range(len(batch_img_metas)):
        if instance_results is not None:
            results = instance_results[img_id]
            assert isinstance(results, InstanceData)
        else:
            results = InstanceData()

        if task_type == "bbox":
            _, box_type = get_box_type(box_type)
            bboxes = torch.zeros(0, box_type.box_dim, device=device)
            if use_box_type:
                bboxes = box_type(bboxes, clone=False)
            results.bboxes = bboxes
            score_shape = (0, num_classes + 1) if score_per_cls else (0,)
            results.scores = torch.zeros(score_shape, device=device)
            results.labels = torch.zeros((0,), device=device, dtype=torch.long)
        else:
            # TODO: Handle the case where rescale is false
            img_h, img_w = batch_img_metas[img_id]["ori_shape"][:2]
            # the type of `im_mask` will be torch.bool or torch.uint8,
            # where uint8 if for visualization and debugging.
            im_mask = torch.zeros(
                0,
                img_h,
                img_w,
                device=device,
                dtype=torch.bool if mask_thr_binary >= 0 else torch.uint8,
            )
            results.masks = im_mask
        results_list.append(results)
    return results_list
