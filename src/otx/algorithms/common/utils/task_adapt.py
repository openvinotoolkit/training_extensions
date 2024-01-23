"""Module for defining task adapt related utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from otx.utils.logger import get_logger

logger = get_logger()


def map_class_names(src_classes, dst_classes):
    """Computes src to dst index mapping.

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
    """A function that concatenates the results of multiple runs into a single array.

    :param results: list, a list of dictionaries or arrays containing the results.
    :return: numpy.ndarray or dict, the concatenated results.
    """
    if isinstance(results[0], dict):
        tasks = results[0].keys()
        res_refine = {}
        for task in tasks:
            res_refine[task] = np.concatenate([res[task] for res in results])
    else:
        res_refine = np.vstack(results)
    return res_refine


def extract_anchor_ratio(dataset, num_ratios=5):
    """A function that extracts anchor ratios from a given dataset.

    :param dataset: dataset object, an instance of a dataset.
    :param num_ratios: int, the number of anchor ratios to be extracted.
    :return: list, a list of extracted anchor ratios.
    """
    ratio_dict = dict(info=[], step=-1)
    dataset, _ = unwrap_dataset(dataset)
    for item in dataset:
        ori_shape = item["img_metas"].data["ori_shape"]
        img_shape = item["img_metas"].data["img_shape"]
        bboxes = item["gt_bboxes"].data.numpy()
        for bbox in bboxes:
            w_o = bbox[2] - bbox[0]
            h_o = bbox[3] - bbox[1]
            if w_o > 0.04 * ori_shape[1] and h_o > 0.04 * ori_shape[0]:
                w_i = w_o * img_shape[1] / ori_shape[1]
                h_i = h_o * img_shape[0] / ori_shape[0]
                ratio_dict["info"].append(w_i / h_i)
    ratio_dict["info"] = np.sort(np.array(ratio_dict["info"]))
    ratio_dict["step"] = int(len(ratio_dict["info"]) / num_ratios)
    proposal_ratio = []
    for i in range(num_ratios):
        r = np.mean(ratio_dict["info"][i * ratio_dict["step"] : (i + 1) * ratio_dict["step"]])
        proposal_ratio.append(r)
    return proposal_ratio


def map_cat_and_cls_as_order(classes, cats):
    """A function that maps classes and categories to label orders.

    :param classes: list, a list of class names.
    :param cats: dict, a dictionary containing category information.
    :return: tuple of dict and list, a dictionary mapping category IDs to label orders and a list of category IDs.
    """
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
    """A function that unwraps a dataset object to its base dataset.

    :param dataset: dataset object, an instance of a dataset.
    :return: tuple of dataset object and int, the base dataset and the number of times to repeat the dataset.
    """
    times = 1
    target_dataset = dataset
    while hasattr(target_dataset, "dataset"):
        if hasattr(target_dataset, "times"):
            times = target_dataset.times
        target_dataset = target_dataset.dataset
    return target_dataset, times
