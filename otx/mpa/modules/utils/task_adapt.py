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


def class_sensitive_copy_state_dict(src_dict, src_classes, dst_dict, dst_classes, model_type):
    if model_type in ["ImageClassifier", "TaskIncrementalLwF", "ClsIncrementalClassifier", "SAMImageClassifier"]:
        class_sensitive_copy_state_dict_cls(src_dict, src_classes, dst_dict, dst_classes)
    else:
        logger.warning(f"Class-sensitive state_dict copy for {model_type} is not yet supported!")


def class_sensitive_copy_state_dict_cls(src_dict, src_classes, dst_dict, dst_classes):
    logger.info("Class sensitive weight copy called!")

    # Strip 'module.' from state_dicts if any
    dst2src = map_class_names(dst_classes, src_classes)
    logger.info(f"{src_classes} -> {dst_classes}")
    src_dict = {k.replace("module.", ""): v for k, v in src_dict.items()}
    dst_dict = {k.replace("module.", ""): v for k, v in dst_dict.items()}
    # (model, state_dict):
    for k, v_load in src_dict.items():
        if k in dst_dict:
            v = dst_dict[k]
            if torch.equal(torch.tensor(v.shape), torch.tensor(v_load.shape)):
                v.copy_(v_load)
            elif "head" in k:
                if len(v.shape) > 1:
                    v_load = sync_transpose_tensor(v_load, v)
                for dst_idx, src_idx in enumerate(dst2src):
                    v[dst_idx].copy_(v_load[src_idx])
            else:
                raise ValueError("the size of model and checkpoint file are not matched key: " f"{k}")
        else:
            logger.warning(f"WARNING: can not find weight key: {k} in dst_model")
    logger.info("copied state dict completely")


def prob_extractor(model, data_loader):
    if torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        model.eval()
        probs = []
        features = []
        for i, data_batch in enumerate(data_loader):
            img = data_batch["img"]
            if torch.cuda.is_available():
                img = img.cuda()
            p, f = model.extract_prob(img)
            probs.append(p)
            features.append(f.detach().cpu().numpy())
    probs = refine_results(probs)
    features = refine_results(features)
    return probs, features


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


def sync_transpose_tensor(src_w, dst_w):
    src_shape = torch.tensor(src_w.shape)
    dst_shape = torch.tensor(dst_w.shape)
    if src_shape[1] == dst_shape[1]:
        return src_w
    elif src_shape[0] == dst_shape[1]:
        return src_w.t()
    else:
        raise ValueError("the size of model and checkpoint file are not matched")


def unwrap_dataset(dataset):
    times = 1
    target_dataset = dataset
    while hasattr(target_dataset, "dataset"):
        if hasattr(target_dataset, "times"):
            times = target_dataset.times
        target_dataset = target_dataset.dataset
    return target_dataset, times
