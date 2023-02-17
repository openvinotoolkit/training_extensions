# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch

from otx.mpa.utils.logger import get_logger

logger = get_logger()


def map_class_names(src_classes, dst_classes):
    """Computes src to dst index mapping

    src2dst[src_idx] = dst_idx
    #  according to class name matching, -1 for non-matched ones
    assert(len(src2dst) == len(src_classes))
    ex)
      src_classes = ['person', 'car', 'tree']
      dst_classes = ['tree', 'person', 'sky', 'ball']
      -> Returns src2dst = [1, -1, 0]
    """
    src2dst = []
    for src_class in src_classes:
        if src_class in dst_classes:
            src2dst.append(dst_classes.index(src_class))
        else:
            src2dst.append(-1)
    return src2dst


def refine_results(results):
    if isinstance(results[0], dict):
        tasks = results[0].keys()
        res_refine = {}
        for task in tasks:
            res_refine[task] = np.concatenate([res[task] for res in results])
    else:
        res_refine = np.vstack(results)
    return res_refine


def extract_anchor_ratio(dataset, num_ratios=5):
    ratio_info = []
    if hasattr(dataset, "dataset"):  # to confirm dataset is wrapped.
        dataset = dataset.dataset
    for ds in dataset:
        ori_shape = ds["img_metas"].data["ori_shape"]
        img_shape = ds["img_metas"].data["img_shape"]
        bboxes = ds["gt_bboxes"].data.numpy()
        for bbox in bboxes:
            w_o = bbox[2] - bbox[0]
            h_o = bbox[3] - bbox[1]
            if w_o > 0.04 * ori_shape[1] and h_o > 0.04 * ori_shape[0]:
                w_i = w_o * img_shape[1] / ori_shape[1]
                h_i = h_o * img_shape[0] / ori_shape[0]
                ratio_info.append(w_i / h_i)
    ratio_info = np.sort(np.array(ratio_info))
    ratio_step = int(len(ratio_info) / num_ratios)
    proposal_ratio = []
    for i in range(num_ratios):
        r = np.mean(ratio_info[i * ratio_step : (i + 1) * ratio_step])
        proposal_ratio.append(r)
    return proposal_ratio


def map_cat_and_cls_as_order(classes, cats):
    cat2label = {}
    cat_ids = []
    for i, cls in enumerate(classes):
        for _, cat in cats.items():
            if cls == cat["name"]:
                cat_id = cat["id"]
                cat_ids.append(cat_id)
                cat2label.update({cat_id: i})
    return cat2label, cat_ids


def unwrap_dataset(dataset):
    times = 1
    target_dataset = dataset
    while hasattr(target_dataset, "dataset"):
        if hasattr(target_dataset, "times"):
            times = target_dataset.times
        target_dataset = target_dataset.dataset
    return target_dataset, times
