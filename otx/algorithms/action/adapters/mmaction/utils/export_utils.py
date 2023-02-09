"""Utils for Action recognition OpenVINO export task."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import glob
from functools import partial
from subprocess import DEVNULL, CalledProcessError, run  # nosec
from typing import List, Optional

import cv2
import numpy as np
import onnx
import torch
from mmaction.models import Recognizer3D
from mmcv.runner import BaseModule
from mmcv.utils import Config

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError as e:
    raise NotImplementedError("please update mmcv to version>=1.0.4") from e

from otx.algorithms.action.adapters.mmaction.models.detectors import AVAFastRCNN


def _convert_sync_batch_to_normal_batch(module: BaseModule):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_sync_batch_to_normal_batch(child))
    del module
    return module_output


# pylint: disable=too-many-locals
def preprocess(
    clip_len: int,
    width: int,
    height: int,
    interval: int = 1,
    category: int = 1,
):
    """Pre-process for action detection structure.

    To get proper proposals from Faster-RCNN, the action detector needs real data
    """
    frames = []
    frame_dir = f"tests/assets/cvat_dataset/action_detection/train/{category}/images"
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    # TODO: allow only .jpg, .png exts
    rawframes = glob.glob(frame_dir + "/*")  # type: ignore[index]
    rawframes.sort()
    for rawframe in rawframes:
        frame = cv2.imread(rawframe)
        ori_h, ori_w, _ = frame.shape
        resized_frame = cv2.resize(frame, (width, height))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
        resized_frame = (resized_frame - mean) / std
        frames.append(resized_frame)
    np_frames = np.expand_dims(frames, axis=0)  # [1, T, H, W, C]
    np_frames = np_frames.transpose(0, 4, 1, 2, 3)  # [1, C, T, H, W]
    frame_inds = get_frame_inds(np_frames, clip_len, interval)
    np_frames = np_frames[:, :, frame_inds, :, :]
    torch_input = torch.Tensor(np_frames)
    one_meta = {
        "img_shape": torch.Tensor((height, width)),
        "ori_shape": (ori_h, ori_w),
        "pad_shape": (height, width),
        "filename": "demo_vid.png",
        "scale_factor": np.array([width / ori_w, height / ori_h, width / ori_w, height / ori_h]),
        "flip": False,
        "show_img": False,
        "flip_direction": None,
    }
    meta = [[one_meta]]
    return torch_input, meta


def get_frame_inds(np_frames: np.ndarray, clip_len: int, interval: int):
    """Get sampled index for given np_frames."""
    frame_len = np_frames.shape[2]
    ori_clip_len = clip_len * interval
    if frame_len > ori_clip_len - 1:
        start = (frame_len - ori_clip_len + 1) / 2
    else:
        start = 0
    frame_inds = np.arange(clip_len) * interval + int(start)
    frame_inds = np.clip(frame_inds, 0, frame_len - 1)
    frame_inds = frame_inds.astype(np.int)
    return frame_inds


# pylint: disable=too-many-arguments, too-many-locals, protected-access, no-member
def pytorch2onnx(
    model: BaseModule,
    input_shape: List[int],
    opset_version: int = 11,
    show: bool = False,
    output_file: Optional[str] = "tmp.onnx",
    is_localizer: bool = False,
):
    """Convert pytorch model to onnx model.

    Args:
        model (BaseModule): The pytorch model to be exported.
        input_shape (List[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        is_localizer(bool): Determines this model is localizer or not
            Default: False.
    """
    model = _convert_sync_batch_to_normal_batch(model)
    input_tensor = torch.randn(input_shape)

    # onnx.export does not support kwargs
    if not isinstance(model, Recognizer3D):
        if isinstance(model, AVAFastRCNN):
            model.add_detector()
            model.patch_pools()
        input_tensor, meta = preprocess(input_shape[2], input_shape[3], input_shape[4])
        model.forward = partial(model.forward_infer, img_metas=meta)
        onnx_input = [input_tensor]
        input_names = ["data"]
        output_names = ["det_bboxes", "det_labels"]
    else:
        if hasattr(model, "forward_dummy"):
            # TODO Replace model.forward with model.onnx_export
            model.forward = model.forward_dummy
        elif hasattr(model, "_forward") and is_localizer:
            model.forward = model._forward
        else:
            raise NotImplementedError("Please implement the forward method for exporting.")
        onnx_input = input_tensor
        input_names = ["data"]
        output_names = ["logits"]

    model.cpu().eval()

    register_extra_symbolics(opset_version)
    torch.onnx.export(
        model,
        onnx_input,
        output_file,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version,
    )

    print(f"Successfully exported ONNX model: {output_file}")


def _get_mo_cmd():
    for mo_cmd in ("mo", "mo.py"):
        try:
            run([mo_cmd, "-h"], stdout=DEVNULL, stderr=DEVNULL, check=True)
            return mo_cmd
        except CalledProcessError:
            pass
    raise RuntimeError("OpenVINO Model Optimizer is not found or configured improperly")


# pylint: disable=no-member
def onnx2openvino(
    cfg: Config,
    onnx_model_path: Optional[str],
    output_dir_path: Optional[str],
    layout: str,
    input_shape: Optional[List[int]] = None,
    input_format: str = "bgr",
    precision: str = "FP32",
    pruning_transformation: bool = False,
):
    """Convert ONNX model into OpenVINO model."""
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(onnx_model_path)
    _output_names = set(out.name for out in onnx_model.graph.output)
    # Clear names of the nodes that produce network's output blobs.
    for node in onnx_model.graph.node:
        if _output_names.intersection(node.output):
            node.ClearField("name")
    onnx.save(onnx_model, onnx_model_path)
    output_names = ",".join(_output_names)

    mo_cmd = _get_mo_cmd()

    normalize = None
    for pipeline in cfg.data.test.pipeline:
        if pipeline["type"] == "Normalize":
            normalize = pipeline
            break
    assert normalize, "Could not find normalize parameters in data pipeline"

    mean_values = normalize["mean"]
    scale_values = normalize["std"]
    command_line = [
        mo_cmd,
        f"--input_model={onnx_model_path}",
        f"--mean_values={mean_values}",
        f"--scale_values={scale_values}",
        f"--output_dir={output_dir_path}",
        f"--output={output_names}",
        f"--data_type={precision}",
        f"--source_layout={layout}",
    ]

    assert input_format.lower() in ["bgr", "rgb"]

    if input_shape is not None:
        command_line.append(f"--input_shape={input_shape}")
    if normalize["to_bgr"]:
        command_line.append("--reverse_input_channels")
    if pruning_transformation:
        command_line.extend(["--transform", "Pruning"])

    print(" ".join(command_line))

    run(command_line, check=True)


def export_model(
    model: BaseModule,
    config: Config,
    onnx_model_path: Optional[str] = None,
    output_dir_path: Optional[str] = None,
):
    """Export PyTorch model into OpenVINO model."""
    if isinstance(model, Recognizer3D):
        input_shape = [1, 1, 3, 8, 224, 224]
        layout = "??c???"
    else:
        input_shape = [1, 3, 32, 256, 256]
        layout = "bctwh"
    pytorch2onnx(model, input_shape=input_shape, output_file=onnx_model_path)
    onnx2openvino(config, onnx_model_path, output_dir_path, layout, input_shape)
