"""Report Generating for OTX CLI."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Union

import torch

import otx
from otx.algorithms.common.utils import is_xpu_available
from otx.api.entities.model_template import ModelTemplate


def get_otx_report(
    model_template: ModelTemplate,
    task_config: Dict[str, Any],
    data_config: Dict[str, Dict[str, str]],
    results: Dict[str, Any],
    output_path: Union[str, Path],
):
    """Generate CLI reports."""
    dash_line = "-" * 60 + "\n\n"
    # Header
    report_str = get_otx_cli_ascii_banner()
    report_str += dash_line
    report_str += f"Current path: {Path.cwd()}\n"
    report_str += f"sys.argv: {sys.argv}\n"
    report_str += f"OTX: {otx.__version__}\n"
    # 1. Machine Environment
    report_str += sub_title_to_str("Running Environments")
    report_str += env_info_to_str()

    # 2. Task Information (Task, Train-type, Etc.)
    if model_template and task_config:
        report_str += sub_title_to_str("Template Information")
        report_str += template_info_to_str(model_template)

    # 3. Dataset Configuration
    if data_config:
        report_str += sub_title_to_str("Dataset Information")
        report_str += data_config_to_str(data_config)

    # 4. Configurations
    report_str += sub_title_to_str("Configurations")
    report_str += task_config_to_str(task_config)
    # 5. Result Summary
    report_str += sub_title_to_str("Results")
    for key, value in results.items():
        report_str += f"\t{key}: {pformat(value)}\n"

    Path(output_path).write_text(report_str, encoding="UTF-8")


def sub_title_to_str(title: str):
    """Add sub title for report."""
    dash_line = "-" * 60
    report_str = ""
    report_str += dash_line + "\n\n"
    report_str += title + "\n\n"
    report_str += dash_line + "\n"
    return report_str


def env_info_to_str():
    """Get Environments."""
    report_str = ""
    env_info = {}
    try:
        from mmcv.utils.env import collect_env

        env_info = collect_env()
        if "PyTorch compiling details" in env_info:
            env_info.pop("PyTorch compiling details")
    except ModuleNotFoundError:
        env_info["sys.platform"] = sys.platform
        env_info["Python"] = sys.version.replace("\n", "")

        cuda_available = torch.cuda.is_available()
        env_info["CUDA available"] = cuda_available

        if cuda_available:
            devices = defaultdict(list)
            for k in range(torch.cuda.device_count()):
                devices[torch.cuda.get_device_name(k)].append(str(k))
            for name, device_ids in devices.items():
                env_info["GPU " + ",".join(device_ids)] = name
        env_info["PyTorch"] = torch.__version__

    if is_xpu_available():
        devices = defaultdict(list)
        for k in range(torch.xpu.device_count()):
            devices[torch.xpu.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info["GPU " + ",".join(device_ids)] = name

    for key, value in env_info.items():
        report_str += f"\t{key}: {value}\n"
    return report_str


def template_info_to_str(model_template: ModelTemplate):
    """Get Template information."""
    report_str = ""
    for key, value in model_template.__dict__.items():
        report_str += f"\t{key}: {pformat(value)}\n"
    return report_str


def data_config_to_str(data_config: Dict[str, Dict[str, str]]):
    """Get Dataset configuration."""
    report_str = ""
    for subset_key, subset_value in data_config.items():
        report_str += f"{subset_key}:\n"
        for key, value in subset_value.items():
            report_str += f"\t{key}: {value}\n"
    return report_str


def task_config_to_str(task_config: Dict[str, Any]):
    """Get Task configuration."""
    report_str = ""
    not_target = ["log_config"]
    for target, value in task_config.items():
        # Remove otx_dataset from the report as it is unnecessary.
        if target == "data" and isinstance(value, dict):
            for item in value.values():
                if isinstance(item, dict) and "otx_dataset" in item:
                    del item["otx_dataset"]

        if target not in not_target:
            report_str += target + ": "
            model_str = pformat(value)
            report_str += model_str + "\n"
    return report_str


def get_otx_cli_ascii_banner():
    """Get OTX ASCII banner."""
    return """

 ██████╗     ████████╗    ██╗  ██╗
██╔═══██╗    ╚══██╔══╝    ╚██╗██╔╝
██║   ██║       ██║        ╚███╔╝
██║   ██║       ██║        ██╔██╗
╚██████╔╝       ██║       ██╔╝ ██╗
 ╚═════╝        ╚═╝       ╚═╝  ╚═╝

"""
