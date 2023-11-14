"""Train function for detection task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch
import tqdm
from mmcv.ops.nms import NMSop
import mmcv
from mmcv.ops.roi_align import RoIAlign
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    OptimizerHook,
    build_runner,
    get_dist_info,
)
from mmcv.engine import single_gpu_test
from mmcv.utils import ext_loader
from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import build_ddp, compat_cfg, find_latest_checkpoint, get_root_logger
from mmdet.utils.util_distribution import build_dp, dp_factory
from torchvision.ops import nms as tv_nms
from torchvision.ops import roi_align as tv_roi_align

from otx.algorithms.common.adapters.mmcv.utils import HPUDataParallel, XPUDataParallel
from otx.algorithms.common.adapters.torch.utils.utils import ModelDebugger

ext_module = ext_loader.load_ext("_ext", ["nms", "softnms", "nms_match", "nms_rotated", "nms_quadri"])
dp_factory["xpu"] = XPUDataParallel


def train_detector_debug(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Trains a detector via mmdet."""

    # Prepare configs for training
    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)
    runner_type = "EpochBasedRunner"
    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False,
    )
    train_loader_cfg = {**train_dataloader_default_args, **cfg.data.get("train_dataloader", {})}
    fp16_cfg = cfg.get("fp16_", None)

    eval_cfg = cfg.get("evaluation", {})
    val_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False, persistent_workers=False
        )
    val_dataloader_args = {**val_dataloader_default_args, **cfg.data.get("val_dataloader", {})}
    if val_dataloader_args["samples_per_gpu"] > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)

    # Build dataloaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

    # put model on gpus
    if cfg.device == "xpu":
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids, enable_autocast=bool(fp16_cfg))
        model.to(f"xpu:{cfg.gpu_ids[0]}")
        # patch mmdetection NMS and ROI Align with torchvision
        NMSop.forward = monkey_patched_nms
        RoIAlign.forward = monkey_patched_roi_align
    else:
        # CUDA
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # wrap up model with torch.xpu.optimize
    if cfg.device == "xpu":
        if fp16_cfg is not None:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        model.train()
        model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=dtype)

    best_score = 0
    validate = True
    # debugging tool to save tensors with weights and gradients
    model_debugger = ModelDebugger(model, enabled=True, save_dir="./debug_folder", max_iters=2)

    # Simple training loop
    for epoch in tqdm.tqdm(range(cfg.runner.max_epochs)):
        num_iter_per_epoch = len(data_loaders[-1])
        for i, data in enumerate(data_loaders[-1]):
            optimizer.zero_grad()
            total_loss = 0
            cur_iter = num_iter_per_epoch * epoch + i
            with model_debugger(iter=cur_iter):
                losses = model(return_loss=True, **data)
                for name, loss_ in losses.items():
                    if not name.startswith("loss"):
                        continue
                    if isinstance(loss_, list):
                        for sub_loss in loss_:
                            total_loss += sub_loss
                    else:
                        total_loss += loss_
                total_loss.backward()
            print(f"loss_iter_{cur_iter}: ", total_loss)
            optimizer.step()

        if validate:
            results = single_gpu_test(model, val_dataloader)
            best_score = evaluate(results, val_dataloader, best_score)
            model.train()


def monkey_patched_nms(ctx, bboxes, scores, iou_threshold, offset, score_threshold, max_num):
    """Runs MMCVs NMS with torchvision.nms, or forces NMS from MMCV to run on CPU."""
    is_filtering_by_score = score_threshold > 0
    if is_filtering_by_score:
        valid_mask = scores > score_threshold
        bboxes, scores = bboxes[valid_mask], scores[valid_mask]
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

    if bboxes.dtype == torch.bfloat16:
        bboxes = bboxes.to(torch.float32)
    if scores.dtype == torch.bfloat16:
        scores = scores.to(torch.float32)

    if offset == 0:
        inds = tv_nms(bboxes, scores, float(iou_threshold))
    else:
        device = bboxes.device
        bboxes = bboxes.to("cpu")
        scores = scores.to("cpu")
        inds = ext_module.nms(bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
        bboxes = bboxes.to(device)
        scores = scores.to(device)

    if max_num > 0:
        inds = inds[:max_num]
    if is_filtering_by_score:
        inds = valid_inds[inds]
    return inds


def monkey_patched_roi_align(self, input, rois):
    """Replaces MMCVs roi align with the one from torchvision.

    Args:
        self: patched instance
        input: NCHW images
        rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
    """

    if "aligned" in tv_roi_align.__code__.co_varnames:
        return tv_roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)
    else:
        if self.aligned:
            rois -= rois.new_tensor([0.0] + [0.5 / self.spatial_scale] * 4)
        return tv_roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)


def evaluate(results, dataloader, best_score):
    """Evaluate predictions from model with ground truth."""
    eval_res = dataloader.dataset.evaluate(results, logger=None, metric='mAP', iou_thr=[0.5])
    score = eval_res["mAP"]

    if score >= best_score:
        best_score = score

    return best_score