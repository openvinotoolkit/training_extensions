"""OMZ wrapper-related code for otx.core.ov."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shutil
import string
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests
from openvino.model_zoo import _common, _reporting
from openvino.model_zoo._configuration import load_models
from openvino.model_zoo.download_engine.downloader import Downloader
from openvino.model_zoo.download_engine.postprocessing import PostprocUnpackArchive
from openvino.model_zoo.omz_converter import ModelOptimizerProperties, convert_to_onnx
from requests.exceptions import HTTPError

from otx.core.file import OTX_CACHE

# pylint: disable=too-many-locals, too-many-branches
OMZ_CACHE = os.path.join(OTX_CACHE, "omz")
os.makedirs(OMZ_CACHE, exist_ok=True)


OMZ_PUBLIC_MODELS: Dict[str, List[str]] = dict(
    cls=[
        "alexnet",
        "caffenet",
        #  "convnext-tiny",                # omz_downloader does not support
        "densenet-121",
        "densenet-121-tf",
        "dla-34",
        "efficientnet-b0",
        "efficientnet-b0-pytorch",
        "efficientnet-v2-b0",
        "efficientnet-v2-s",
        "hbonet-1.0",
        "hbonet-0.25",
        "googlenet-v1",
        "googlenet-v1-tf",
        "googlenet-v2",
        "googlenet-v2-tf",
        "googlenet-v3",
        "googlenet-v3-pytorch",
        "googlenet-v4-tf",
        "inception-resnet-v2-tf",
        #  "levit-128s",                   # IR has hard-codeded batch size of 1
        "mixnet-l",
        "mobilenet-v1-0.25-128",
        "mobilenet-v1-1.0-224",
        "mobilenet-v1-1.0-224-tf",
        "mobilenet-v2",
        "mobilenet-v2-1.0-224",
        "mobilenet-v2-pytorch",
        "mobilenet-v2-1.4-224",
        "mobilenet-v3-small-1.0-224-tf",
        "mobilenet-v3-large-1.0-224-tf",
        #  "nfnet-f0",                     # mo 2022.2 bug
        #  "regnetx-3.2gf",                # omz_converter does not support
        "octave-resnet-26-0.25",
        #  "repvgg-a0",                    # trainig and inference architecture are difference
        #  "repvgg-b1",                    # trainig and inference architecture are difference
        #  "repvgg-b3",                    # trainig and inference architecture are difference
        #  "resnest-50-pytorch",           # IR has hard-coded batch size of 1
        "resnet-18-pytorch",
        "resnet-34-pytorch",
        "resnet-50-pytorch",
        "resnet-50-tf",
        #  "rexnet-v1-x1.0",               # IR has hard-coded batch size of 1
        "se-inception",
        "se-resnet-50",
        "se-resnext-50",
        "shufflenet-v2-x0.5",
        #  "shufflenet-v2-x1.0",           # IR has hard-coded batch size of 1
        "squeezenet1.0",
        "squeezenet1.1",
        #  "swin-tiny-patch4-window7-224", # IR has hard-coded batch size of 1
        #  "t2t-vit-14",                   # IR has hard-coded batch size of 1
        "vgg16",
        "vgg19",
    ],
    det=[],
    seg=[],
)


AVAILABLE_OMZ_MODELS: List[str] = []
for models_ in OMZ_PUBLIC_MODELS.values():
    for model_ in models_:
        AVAILABLE_OMZ_MODELS.append(model_)


class NameSpace:
    """NameSpace class for otx.core.ov.omz_wrapper."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _get_etag(url):
    """Getter etag function from url."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=100)
        if response.status_code != 200:
            return None
        return response.headers.get("ETag", None)
    except HTTPError:
        return None


def _get_ir_path(directory):
    """Getter IR path function from directory path."""
    directory = Path(directory)
    model_path = list(directory.glob("**/*.xml"))
    weight_path = list(directory.glob("**/*.bin"))
    if model_path and weight_path:
        assert len(model_path) == 1 and len(weight_path) == 1
        return dict(model_path=model_path[0], weight_path=weight_path[0])
    return None


def _run_pre_convert(reporter, model, output_dir, args):
    """Run pre-converting function."""
    script = _common.MODEL_ROOT / model.subdirectory_ori / "pre-convert.py"
    if not script.exists():
        return True

    reporter.print_section_heading(
        "{}Running pre-convert script for {}",
        "(DRY RUN) " if args.dry_run else "",
        model.name,
    )

    cmd = [
        str(args.python),
        "--",
        str(script),
        "--",
        str(args.download_dir / model.subdirectory),
        str(output_dir / model.subdirectory),
    ]

    reporter.print("Pre-convert command: {}", _common.command_string(cmd))
    reporter.print(flush=True)

    success = True if args.dry_run else reporter.job_context.subprocess(cmd)
    reporter.print()

    return success


def _update_model(model):
    """Update model configs for omz_wrapper."""
    m_hash = hashlib.sha256()
    for file in model.files:
        url = file.source.url
        etag = _get_etag(url)
        if etag is not None:
            m_hash.update(bytes(etag, "utf-8"))
    model.subdirectory_ori = model.subdirectory
    model.subdirectory = Path(m_hash.hexdigest())

    # FIXME: a bug from openvino-dev==2022.3.0
    # It has been fixed on master branch.
    # After upgrading openvino-dev, we can remove this temporary patch
    if getattr(model, "conversion_to_onnx_args") and not [
        arg for arg in model.conversion_to_onnx_args if arg.startswith("--model-path")
    ]:
        model.conversion_to_onnx_args.append("--model-path=")


def get_model_configuration(model_name):
    """Getter function of model configuration from name."""
    model_configurations = load_models(_common.MODEL_ROOT, {})
    for model in model_configurations:
        if model.name == model_name:
            _update_model(model)
            return model
    return None


def download_model(model, download_dir=OMZ_CACHE, precisions=None, force=False):
    """Function for downloading model from directory."""
    download_dir = Path("") if download_dir is None else Path(download_dir)
    precisions = precisions if precisions else {"FP32"}

    # TODO: need delicate cache management
    if not force and (download_dir / model.subdirectory).exists():
        target_file_names = []
        for postprocessing in model.postprocessing:
            if isinstance(postprocessing, PostprocUnpackArchive):
                target_file_names.append(postprocessing.file)

        done = [False for _ in model.files]
        for i, file in enumerate(model.files):
            filename = file.name
            if filename in target_file_names:
                # TODO
                # here, we assume unarchive is done
                done[i] = True
                continue
            if os.path.exists(download_dir / model.subdirectory / filename):
                done[i] = True

        if all(done):
            return

    reporter = Downloader.make_reporter("text")
    downloader = Downloader(precisions, download_dir)
    failed_models = downloader.bulk_download_model([model], reporter, 1, "text")
    if failed_models:
        reporter.print("FAILED:")
        for failed_model_name in failed_models:
            reporter.print(failed_model_name)
        sys.exit(1)


def _convert(reporter, model, output_dir, namespace, mo_props, requested_precisions):
    """Convert function for OMZ wrapper."""
    if model.mo_args is None:
        reporter.print_section_heading("Skipping {} (no conversions defined)", model.name)
        reporter.print()
        return True

    model_precisions = requested_precisions & model.precisions
    if not model_precisions:
        reporter.print_section_heading("Skipping {} (all conversions skipped)", model.name)
        reporter.print()
        return True

    (output_dir / model.subdirectory).mkdir(parents=True, exist_ok=True)

    if not _run_pre_convert(reporter, model, output_dir, namespace):
        return False

    model_format = model.framework
    mo_extension_dir = mo_props.base_dir / "extensions"
    if not mo_extension_dir.exists():
        mo_extension_dir = mo_props.base_dir

    template_variables = {
        "config_dir": _common.MODEL_ROOT / model.subdirectory_ori,
        "conv_dir": output_dir / model.subdirectory,
        "dl_dir": namespace.download_dir / model.subdirectory,
        "mo_dir": mo_props.base_dir,
        "mo_ext_dir": mo_extension_dir,
    }

    if model.conversion_to_onnx_args:
        if not convert_to_onnx(reporter, model, output_dir, namespace, template_variables):
            return False
        model_format = "onnx"

    expanded_mo_args = [string.Template(arg).substitute(template_variables) for arg in model.mo_args]

    for model_precision in sorted(model_precisions):
        data_type = model_precision.split("-")[0]
        layout_string = ",".join(f"{input.name}({input.layout})" for input in model.input_info if input.layout)
        shape_string = ",".join(str(input.shape) for input in model.input_info if input.shape)

        if layout_string:
            expanded_mo_args.append(f"--layout={layout_string}")
        if shape_string:
            expanded_mo_args.append(f"--input_shape={shape_string}")

        mo_cmd = [
            *mo_props.cmd_prefix,
            f"--framework={model_format}",
            f"--output_dir={output_dir / model.subdirectory / model_precision}",
            f"--model_name={model.name}",
            f"--input={','.join(input.name for input in model.input_info)}".format(),
            *expanded_mo_args,
            *mo_props.extra_args,
        ]
        if "FP16" in data_type:
            mo_cmd.append("--compress_to_fp16")

        reporter.print_section_heading(
            "{}Converting {} to IR ({})",
            "(DRY RUN) " if namespace.dry_run else "",
            model.name,
            model_precision,
        )

        reporter.print("Conversion command: {}", _common.command_string(mo_cmd))

        if not namespace.dry_run:
            reporter.print(flush=True)

            if not reporter.job_context.subprocess(mo_cmd):
                # NOTE: mo returns non zero return code (245) even though it successfully generate IR
                cur_time = time.time()
                time_threshold = 5
                xml_path = output_dir / model.subdirectory / model_precision / f"{model.name}.xml"
                bin_path = output_dir / model.subdirectory / model_precision / f"{model.name}.bin"
                if not (
                    os.path.exists(xml_path)
                    and os.path.exists(bin_path)
                    and os.path.getmtime(xml_path) - cur_time < time_threshold
                    and os.path.getmtime(bin_path) - cur_time < time_threshold
                ):
                    return False

        reporter.print()

    return True


def convert_model(
    model,
    download_dir=OMZ_CACHE,
    output_dir=OMZ_CACHE,
    precisions=None,
    force=False,
    *args,
):  # pylint: disable=keyword-arg-before-vararg
    """Converting model for OMZ wrapping."""
    download_dir = Path("") if download_dir is None else Path(download_dir)
    output_dir = Path("") if output_dir is None else Path(output_dir)
    precisions = precisions if precisions else {"FP32"}

    out = _get_ir_path(output_dir / model.subdirectory)
    if out and not force:
        return out

    namespace = NameSpace(
        python=shutil.which("python"),
        dry_run=False,
        download_dir=download_dir,
    )

    mo_executable = shutil.which("mo")

    if mo_executable:
        mo_path = Path(mo_executable)
    else:
        try:
            mo_path = Path(os.environ["INTEL_OPENVINO_DIR"]) / "tools/mo/openvino/tools/mo/mo.py"
            if not mo_path.exists():
                mo_path = Path(os.environ["INTEL_OPENVINO_DIR"]) / "tools/model_optimizer/mo.py"
        except KeyError:
            sys.exit(
                "Unable to locate Model Optimizer. "
                + "Use --mo or run setupvars.sh/setupvars.bat from the OpenVINO toolkit."
            )

    mo_path = mo_path.resolve()
    mo_cmd_prefix = [namespace.python, "--", str(mo_path)]

    if str(mo_path).lower().endswith(".py"):
        mo_dir = mo_path.parent
    else:
        mo_package_path, stderr = _common.get_package_path(namespace.python, "openvino.tools.mo")
        mo_dir = mo_package_path

        if mo_package_path is None:
            mo_package_path, stderr = _common.get_package_path(args.python, "mo")
            if mo_package_path is None:
                sys.exit(f"Unable to load Model Optimizer. Errors occurred: {stderr}")
            mo_dir = mo_package_path.parent

    reporter = _reporting.Reporter(_reporting.DirectOutputContext())
    mo_props = ModelOptimizerProperties(
        cmd_prefix=mo_cmd_prefix,
        extra_args=[],
        base_dir=mo_dir,
    )
    shared_convert_args = (output_dir, namespace, mo_props, precisions)

    results = []
    models = []
    if model.model_stages:
        for model_stage in model.model_stages:
            results.append(_convert(reporter, model_stage, *shared_convert_args))
            models.append(model_stage)
    else:
        results.append(_convert(reporter, model, *shared_convert_args))
        models.append(model)

    failed_models = [model.name for model, successful in zip(models, results) if not successful]

    if failed_models:
        reporter.print("FAILED:")
        for failed_model_name in failed_models:
            reporter.print(failed_model_name)
        sys.exit(1)

    return _get_ir_path(output_dir / model.subdirectory)


def get_omz_model(model_name, download_dir=OMZ_CACHE, output_dir=OMZ_CACHE, force=False):
    """Get OMZ model from name and download_dir."""
    model = get_model_configuration(model_name)
    download_model(model, download_dir=download_dir, force=force)
    return convert_model(model, download_dir=download_dir, output_dir=output_dir, force=force)
