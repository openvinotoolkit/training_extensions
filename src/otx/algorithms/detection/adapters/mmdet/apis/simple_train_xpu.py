"""Train function for detection task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch
import tqdm
from mmdet.apis import single_gpu_test
from mmcv.ops.nms import NMSop
from mmcv.ops.roi_align import RoIAlign
from mmcv.runner.checkpoint import save_checkpoint as mmcv_save_checkpoint
from mmcv.utils import ext_loader
from mmdet.core import build_optimizer
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import compat_cfg, get_root_logger
from mmdet.utils.util_distribution import build_dp, dp_factory
from torchvision.ops import nms as tv_nms
from torchvision.ops import roi_align as tv_roi_align

from otx.algorithms.common.adapters.mmcv.utils import XPUDataParallel
from otx.algorithms.common.adapters.torch.utils.utils import ModelDebugger
from otx.algorithms.segmentation.adapters.mmseg.apis.simple_train_xpu import ReduceLROnPlateauLrUpdater

ext_module = ext_loader.load_ext("_ext", ["nms", "softnms", "nms_match", "nms_rotated", "nms_quadri"])
dp_factory["xpu"] = XPUDataParallel
logger = get_root_logger(logging.INFO)


def train_detector_debug(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Trains a detector via mmdet."""
    # CHANGE IF REQUIRED
    cfg.device = "cuda" # cuda, cpu, xpu

    # Prepare configs for training
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True,
    )
    # The overall dataloader settings
    loader_cfg.update(
        {
            k: v
            for k, v in cfg.data.items()
            if k not in ["train", "val", "test", "train_dataloader", "val_dataloader", "test_dataloader"]
        }
    )

    # prepare train data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    train_loader_cfg = {**loader_cfg, **cfg.data.get("train_dataloader", {})}
    train_data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    num_iter_per_epoch = len(train_data_loaders[-1])

    # prepare val data loaders
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False, persistent_workers=False
        )

        val_dataloader_args = {**val_dataloader_default_args, **cfg.data.get("val_dataloader", {})}
        # Support batch_size > 1 in validation

        if val_dataloader_args["samples_per_gpu"] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"


    # put model on gpus
    if cfg.device == "xpu":
        is_fp16 = bool(cfg.get("fp16_", False))
        model = XPUDataParallel(model, device_ids=cfg.gpu_ids, enable_autocast=bool(is_fp16))
        model.to(f"xpu:{cfg.gpu_ids[0]}")
        # patch mmdetection NMS and ROI Align with torchvision
        NMSop.forward = monkey_patched_nms
        RoIAlign.forward = monkey_patched_roi_align
    else:
        # CUDA
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # build lr scheduler
    cfg.lr_config.pop("policy")
    lr_scheduler = ReduceLROnPlateauLrUpdater(optimizer=optimizer, iter_per_epoch=num_iter_per_epoch, **cfg.lr_config)
    lr_scheduler.before_run()

    # wrap up model with torch.xpu.optimize
    if cfg.device == "xpu":
        if is_fp16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        model.train()
        model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=dtype)

    print_iter = 10
    best_score = 0
    cur_iter = 0
    # debugging tool to save tensors with weights and gradients
    model_debugger = ModelDebugger(model, enabled=False, save_dir="./debug_folder", max_iters=2)

    # Simple training loop
    for epoch in tqdm.tqdm(range(cfg.runner.max_epochs)):
        lr_scheduler.register_progress(epoch, cur_iter)
        lr_scheduler.before_train_epoch()
        model.train()
        for i, data in enumerate(train_data_loaders[-1]):
            cur_iter = num_iter_per_epoch * epoch + i
            lr_scheduler.register_progress(epoch, cur_iter)
            lr_scheduler.before_train_iter()
            optimizer.zero_grad()
            with model_debugger(iter=cur_iter):
                losses = model(return_loss=True, **data)
                # parse loss (sum up)
                total_loss, loss_log = parse_losses(losses)
                total_loss.backward()

            optimizer.step()

            if (i + 1) % print_iter == 0 or i + 1 == num_iter_per_epoch:  # progress log
                logger.info(
                    f"[{i+1} / {num_iter_per_epoch}] "
                    + " / ".join([f"{key} : {round(val,3)}" for key, val in loss_log.items()])
                )

        # eval
        if validate:
            model.eval()
            logger.info(f"Epoch #{epoch + 1} evaluation starts")
            eval_result = single_gpu_test(model, val_dataloader)
            eval_res = val_dataloader.dataset.evaluate(eval_result, logger=None, metric="mAP", iou_thr=[0.5])
            score = eval_res["mAP"]

            if score >= best_score:
                best_score = score

        save_checkpoint(model, optimizer, cfg.work_dir, epoch + 1)

    print("Training finished. Best score: ", best_score)

def parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f"{loss_name} is not a tensor or list of tensors")

    loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

    log_vars["loss"] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def save_checkpoint(model, optim, out_dir: Union[str, Path], epoch: int, max_keep_ckpts=1):
    """Save the current checkpoint and delete unwanted checkpoint."""
    out_dir: Path = Path(out_dir)
    name = "epoch_{}.pth"

    filename = name.format(epoch)
    filepath = out_dir / filename
    mmcv_save_checkpoint(model, str(filepath), optimizer=optim)
    # in some environments, `os.symlink` is not supported, you may need to
    # set `create_symlink` to False
    dst_file = out_dir / "latest.pth"
    if dst_file.exists():
        dst_file.unlink()
    dst_file.symlink_to(filename)

    # remove other checkpoints
    if max_keep_ckpts > 0:
        redundant_ckpts = range(epoch - max_keep_ckpts, 0, -1)
        for _step in redundant_ckpts:
            ckpt_path = out_dir / name.format(_step)
            if ckpt_path.exists():
                ckpt_path.unlink()
            else:
                break


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
    eval_res = dataloader.dataset.evaluate(results, logger=None, metric="mAP", iou_thr=[0.5])
    score = eval_res["mAP"]

    if score >= best_score:
        best_score = score

    return best_score
