"""Common test case and helpers for OTX"""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Union

import onnx
import onnxruntime
import pytest
import torch
import yaml

from otx.api.entities.model_template import ModelCategory, ModelStatus
from otx.cli.tools.find import SUPPORTED_BACKBONE_BACKENDS as find_supported_backends
from otx.cli.tools.find import SUPPORTED_TASKS as find_supported_tasks
from otx.cli.utils.nncf import get_number_of_fakequantizers_in_xml
from otx.algorithms.common.utils.utils import is_xpu_available
from tests.test_suite.e2e_test_system import e2e_pytest_component

try:
    import intel_extension_for_pytorch
except ImportError:
    pass


def get_template_rel_dir(template):
    return os.path.dirname(os.path.relpath(template.model_template_path))


def get_template_dir(template, root) -> str:

    # Get the template directory of the algorithm.
    # The location of the template files are as follows:
    # ~/training_extensions/src/otx/algorithms/<algorithm>/**/template.yaml
    # To get the ``algorithm``, index of the "algorithms" can be
    # searched, where ``algorithm`` comes next.
    template_path_parts = template.model_template_path.split(os.sep)
    idx = template_path_parts.index("algorithms")
    algorithm = template_path_parts[idx + 1]

    algo_backend_dir = f"src/otx/algorithms/{algorithm}"
    work_dir = os.path.join(root, f"src/otx/algorithms/{algorithm}")
    template_dir = os.path.dirname(os.path.relpath(template.model_template_path, start=algo_backend_dir))
    template_work_dir = os.path.join(work_dir, template_dir)

    os.makedirs(template_work_dir, exist_ok=True)
    return template_work_dir


def runner(
    cmd,
    stdout_stream=sys.stdout.buffer,
    stderr_stream=sys.stderr.buffer,
    **kwargs,
):
    async def stream_handler(in_stream, out_stream):
        output = bytearray()
        # buffer line
        line = bytearray()
        while True:
            c = await in_stream.read(1)
            if not c:
                break
            line.extend(c)
            if c == b"\n":
                out_stream.write(line)
                output.extend(line)
                line = bytearray()
        return output

    async def run_and_capture(cmd):
        environ = os.environ.copy()
        environ["PYTHONUNBUFFERED"] = "1"
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environ,
            **kwargs,
        )

        try:
            stdout, stderr = await asyncio.gather(
                stream_handler(process.stdout, stdout_stream),
                stream_handler(process.stderr, stderr_stream),
            )
        except Exception:
            process.kill()
            raise
        finally:
            rc = await process.wait()
        return rc, stdout, stderr

    rc, stdout, stderr = asyncio.run(run_and_capture(cmd))

    return rc, stdout, stderr


def check_run(cmd, **kwargs):
    rc, _, stderr = runner(cmd, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

    if rc != 0:
        stderr = stderr.decode("utf-8").splitlines()
        i = 0
        for i, line in enumerate(stderr):
            if line.startswith("Traceback"):
                break
        stderr = "\n".join(stderr[i:])
    assert rc == 0, stderr


def otx_train_testing(template, root, otx_dir, args, deterministic=True):
    template_work_dir = get_template_dir(template, root)
    command_line = ["otx", "train", template.model_template_path]
    for arg in [
        "--train-ann_file",
        "--train-data-roots",
        "--val-ann-file",
        "--val-data-roots",
        "--unlabeled-data-roots",
        "--unlabeled-file-list",
    ]:
        arg_value = args.get(arg, None)
        if arg_value:
            command_line.extend([arg, os.path.join(otx_dir, arg_value)])
    command_line.extend(["--output", f"{template_work_dir}/trained_{template.model_template_id}"])
    command_line.extend(["--workspace", f"{template_work_dir}"])
    if "--load-weights" in args:
        if not os.path.exists(args["--load-weights"]):
            pytest.skip(reason=f"required file is not exist - {args['--load-weights']}")
        command_line.extend(["--load-weights", args["--load-weights"]])
    if "--gpus" in args:
        command_line.extend(["--gpus", args["--gpus"]])
        if "--multi-gpu-port" in args:
            command_line.extend(["--multi-gpu-port", args["--multi-gpu-port"]])
    if "--train-type" in args:
        command_line.extend(["--train-type", args["--train-type"]])
    if deterministic:
        command_line.extend(["--deterministic"])
    if "train_params" in args:
        command_line.extend(args["train_params"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth")
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/models/label_schema.json")


def otx_resume_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "train",
        template.model_template_path,
    ]
    for option in [
        "--train-ann-file",
        "--train-data-roots",
        "--val-ann-file",
        "--val-data-roots",
        "--unlabeled-data-roots",
        "--unlabeled-file-list",
        "--resume-from",
    ]:
        if option in args:
            command_line.extend([option, f"{os.path.join(otx_dir, args[option])}"])

    if "--resume-from" in args:
        if not os.path.exists(args["--resume-from"]):
            pytest.skip(reason=f"required file is not exist - {args['--resume-from']}")

    command_line.extend(["--output", f"{template_work_dir}/trained_for_resume_{template.model_template_id}"])
    command_line.extend(["--workspace", f"{template_work_dir}"])
    command_line.extend(args["train_params"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth")
    assert os.path.exists(
        f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/label_schema.json"
    )


def otx_hpo_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    if os.path.exists(f"{template_work_dir}/hpo"):
        shutil.rmtree(f"{template_work_dir}/hpo")

    command_line = ["otx", "train", template.model_template_path]

    for arg in ["--train-data-roots", "--val-data-roots"]:
        arg_value = args.get(arg, None)
        if arg_value:
            command_line.extend([arg, os.path.join(otx_dir, arg_value)])
    command_line.extend(["--output", f"{template_work_dir}/hpo_trained_{template.model_template_id}"])
    command_line.extend(["--workspace", f"{template_work_dir}"])
    command_line.extend(["--enable-hpo", "--hpo-time-ratio", "1"])

    command_line.extend(args["train_params"])
    check_run(command_line)
    trials_json = list(
        filter(
            lambda x: x.name.split(".")[0].isnumeric(),
            Path(f"{template_work_dir}/hpo_trained_{template.model_template_id}/hpo/").rglob("*.json"),
        )
    )
    assert trials_json
    for trial_json in trials_json:
        with trial_json.open("r") as f:
            trial_result = json.load(f)
        assert trial_result.get("score")

    assert os.path.exists(f"{template_work_dir}/hpo_trained_{template.model_template_id}/models/weights.pth")
    assert os.path.exists(f"{template_work_dir}/hpo_trained_{template.model_template_id}/models/label_schema.json")


def otx_export_testing(template, root, dump_features=False, half_precision=False, check_ir_meta=False, is_onnx=False):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    save_path = f"{template_work_dir}/exported_{template.model_template_id}"
    command_line = [
        "otx",
        "export",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--output",
        save_path,
    ]

    if dump_features:
        command_line[-1] += "_w_features"
        save_path = command_line[-1]
        command_line.append("--dump-features")
    if half_precision:
        command_line[-1] += "_fp16"
        save_path = command_line[-1]
        command_line.append("--half-precision")
    if is_onnx:
        command_line.extend(["--export-type", "onnx"])

    check_run(command_line)

    path_to_xml = os.path.join(save_path, "openvino.xml")
    assert os.path.exists(os.path.join(save_path, "label_schema.json"))
    if not is_onnx:
        if any(map(lambda x: x in template.model_template_id, ("Visual_Prompting", "Zero_Shot"))):
            path_to_xml = os.path.join(save_path, "visual_prompting_decoder.xml")
            assert os.path.exists(os.path.join(save_path, "visual_prompting_image_encoder.xml"))
            assert os.path.exists(os.path.join(save_path, "visual_prompting_image_encoder.bin"))
            assert os.path.exists(os.path.join(save_path, "visual_prompting_decoder.xml"))
            assert os.path.exists(os.path.join(save_path, "visual_prompting_decoder.bin"))
        else:
            assert os.path.exists(path_to_xml)
            assert os.path.exists(os.path.join(save_path, "openvino.bin"))
            ckpt = torch.load(f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth")
            input_size = ckpt.get("input_size", None)
            if input_size:
                with open(path_to_xml, encoding="utf-8") as xml_stream:
                    xml_model = xml_stream.read()
                    assert f"{input_size[1]},{input_size[0]}" in xml_model
    else:
        if any(map(lambda x: x in template.model_template_id, ("Visual_Prompting", "Zero_Shot"))):
            assert os.path.exists(os.path.join(save_path, "visual_prompting_image_encoder.onnx"))
            assert os.path.exists(os.path.join(save_path, "visual_prompting_decoder.onnx"))
        else:
            path_to_onnx = os.path.join(save_path, "model.onnx")
            assert os.path.exists(path_to_onnx)

            if check_ir_meta:
                onnx_model = onnx.load(path_to_onnx)
                is_model_type_presented = False
                for prop in onnx_model.metadata_props:
                    assert "model_info" in prop.key
                    if "model_type" in prop.key:
                        is_model_type_presented = True
                assert is_model_type_presented

            # In case of tile classifier mmdeploy inserts mark nodes in onnx, making it non-standard
            if not os.path.exists(os.path.join(save_path, "tile_classifier.onnx")):
                onnx.checker.check_model(path_to_onnx)
                onnxruntime.InferenceSession(path_to_onnx)
            return

    if dump_features:
        with open(path_to_xml, encoding="utf-8") as stream:
            xml_model = stream.read()
            assert "feature_vector" in xml_model

    if half_precision:
        with open(path_to_xml, encoding="utf-8") as stream:
            xml_model = stream.read()
            assert "FP16" in xml_model

    if check_ir_meta:
        with open(path_to_xml, encoding="utf-8") as stream:
            xml_model = stream.read()
            assert "model_info" in xml_model
            assert "model_type" in xml_model
            assert "labels" in xml_model


def otx_eval_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        weights_path,
        "--output",
        f"{template_work_dir}/trained_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    command_line.extend(args.get("eval_params", []))
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/performance.json")


def otx_eval_openvino_testing(
    template,
    root,
    otx_dir,
    args,
    threshold=0.0,
    half_precision=False,
    is_visual_prompting=False,
):
    template_work_dir = get_template_dir(template, root)
    weights_file = "visual_prompting_decoder" if is_visual_prompting else "openvino"
    weights_path = f"{template_work_dir}/exported_{template.model_template_id}/{weights_file}.xml"
    output_path = f"{template_work_dir}/exported_{template.model_template_id}"
    perf_path = f"{template_work_dir}/exported_{template.model_template_id}/performance.json"

    if half_precision:
        weights_path = f"{template_work_dir}/exported_{template.model_template_id}_fp16/{weights_file}.xml"
        output_path = f"{template_work_dir}/exported_{template.model_template_id}_fp16"
        perf_path = f"{template_work_dir}/exported_{template.model_template_id}_fp16/performance.json"

    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        weights_path,
        "--output",
        output_path,
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(perf_path)
    with open(f"{template_work_dir}/trained_{template.model_template_id}/performance.json") as read_file:
        trained_performance = json.load(read_file)
    with open(perf_path) as read_file:
        exported_performance = json.load(read_file)

    compare_model_accuracy(exported_performance, trained_performance, threshold)


def otx_demo_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "demo",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        os.path.join(otx_dir, args["--input"]),
        "--delay",
        "-1",
        "--output",
        os.path.join(template_work_dir, "output"),
    ]
    check_run(command_line)
    assert os.path.exists(os.path.join(template_work_dir, "output"))


def otx_demo_openvino_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "demo",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        os.path.join(otx_dir, args["--input"]),
        "--delay",
        "-1",
        "--output",
        os.path.join(template_work_dir, "output"),
    ]
    check_run(command_line)
    assert os.path.exists(os.path.join(template_work_dir, "output"))


def otx_deploy_openvino_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    deployment_dir = f"{template_work_dir}/deployed_{template.model_template_id}"
    command_line = [
        "otx",
        "deploy",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--output",
        deployment_dir,
    ]
    check_run(command_line)
    check_run(["unzip", "-o", "openvino.zip"], cwd=deployment_dir)
    # TODO: Need to check Requirements.txt & new environment is working
    # check_run(
    #     ["python3", "-m", "venv", "venv"],
    #     cwd=os.path.join(deployment_dir, "python"),
    # )
    # check_run(
    #     ["python3", "-m", "pip", "install", "wheel"],
    #     cwd=os.path.join(deployment_dir, "python"),
    # )
    # check_run(
    #     ["python3", "-m", "pip", "install", "pip", "--upgrade"],
    #     cwd=os.path.join(deployment_dir, "python"),
    # )
    # check_run(
    #     ["python3", "-m", "pip", "install", "torch>=1.8.1, <=1.9.1"],
    #     cwd=os.path.join(deployment_dir, "python"),
    # )
    # check_run(
    #     [
    #         "python3",
    #         "-m",
    #         "pip",
    #         "install",
    #         "-r",
    #         os.path.join(deployment_dir, "python", "requirements.txt"),
    #     ],
    #     cwd=os.path.join(deployment_dir, "python"),
    # )
    check_run(
        [
            "python3",
            "demo.py",
            "-m",
            "../model",
            "-i",
            os.path.join(otx_dir, args["--input"]),
            "--inference_type",
            "sync",
            "--no_show",
            "--output",
            os.path.join(deployment_dir, "output"),
        ],
        cwd=os.path.join(deployment_dir, "python"),
    )
    assert os.path.exists(os.path.join(deployment_dir, "output"))

    check_run(
        [
            "python3",
            "demo.py",
            "-m",
            "../model",
            "-i",
            os.path.join(otx_dir, args["--input"]),
            "--inference_type",
            "async",
            "--no_show",
            "--output",
            os.path.join(deployment_dir, "output"),
        ],
        cwd=os.path.join(deployment_dir, "python"),
    )
    assert os.path.exists(os.path.join(deployment_dir, "output"))


def otx_eval_deployment_testing(template, root, otx_dir, args, threshold=0.0):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/deployed_{template.model_template_id}/openvino.zip"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        weights_path,
        "--output",
        f"{template_work_dir}/deployed_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/deployed_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/exported_{template.model_template_id}/performance.json") as read_file:
        exported_performance = json.load(read_file)
    with open(f"{template_work_dir}/deployed_{template.model_template_id}/performance.json") as read_file:
        deployed_performance = json.load(read_file)

    compare_model_accuracy(deployed_performance, deployed_performance, threshold)


def otx_demo_deployment_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    deployment_dir = f"{template_work_dir}/deployed_{template.model_template_id}"

    weights_path = f"{deployment_dir}/openvino.zip"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "demo",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        os.path.join(otx_dir, args["--input"]),
        "--delay",
        "-1",
        "--output",
        os.path.join(deployment_dir, "output"),
    ]
    check_run(command_line)
    assert os.path.exists(os.path.join(deployment_dir, "output"))


def ptq_optimize_testing(template, root, otx_dir, args, is_visual_prompting=False):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml"
    if is_visual_prompting:
        weights_path = f"{template_work_dir}/exported_{template.model_template_id}/visual_prompting_decoder.xml"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "optimize",
        template.model_template_path,
        "--train-data-roots",
        f'{os.path.join(otx_dir, args["--train-data-roots"])}',
        "--val-data-roots",
        f'{os.path.join(otx_dir, args["--val-data-roots"])}',
        "--output",
        f"{template_work_dir}/ptq_{template.model_template_id}",
        "--load-weights",
        weights_path,
    ]

    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    if is_visual_prompting:
        assert os.path.exists(
            f"{template_work_dir}/ptq_{template.model_template_id}/visual_prompting_image_encoder.xml"
        )
        assert os.path.exists(
            f"{template_work_dir}/ptq_{template.model_template_id}/visual_prompting_image_encoder.bin"
        )
        assert os.path.exists(f"{template_work_dir}/ptq_{template.model_template_id}/visual_prompting_decoder.xml")
        assert os.path.exists(f"{template_work_dir}/ptq_{template.model_template_id}/visual_prompting_decoder.bin")
    else:
        assert os.path.exists(f"{template_work_dir}/ptq_{template.model_template_id}/openvino.xml")
        assert os.path.exists(f"{template_work_dir}/ptq_{template.model_template_id}/openvino.bin")
    assert os.path.exists(f"{template_work_dir}/ptq_{template.model_template_id}/label_schema.json")


def _validate_fq_in_xml(xml_path, path_to_ref_data, compression_type, test_name, update=False):
    num_fq = get_number_of_fakequantizers_in_xml(xml_path)
    assert os.path.exists(path_to_ref_data), f"Reference file does not exist: {path_to_ref_data} [num_fq = {num_fq}]"

    with open(path_to_ref_data, encoding="utf-8") as stream:
        ref_data = yaml.safe_load(stream)
    ref_num_fq = ref_data.get(test_name, {}).get(compression_type, {}).get("number_of_fakequantizers", -1)
    if update:
        print(f"Updating FQ refs: {ref_num_fq}->{num_fq} for {compression_type}")
        ref_data[test_name][compression_type]["number_of_fakequantizers"] = num_fq
        with open(path_to_ref_data, encoding="utf-8", mode="w") as stream:
            stream.write(yaml.safe_dump(ref_data))
    assert num_fq == ref_num_fq, f"Incorrect number of FQs in optimized model: {num_fq} != {ref_num_fq}"


def ptq_validate_fq_testing(template, root, otx_dir, task_type, test_name):
    template_work_dir = get_template_dir(template, root)
    if "visual_prompting" == task_type:
        xml_paths = [
            f"{template_work_dir}/ptq_{template.model_template_id}/visual_prompting_image_encoder.xml",
            f"{template_work_dir}/ptq_{template.model_template_id}/visual_prompting_decoder.xml",
        ]
    else:
        xml_paths = [f"{template_work_dir}/ptq_{template.model_template_id}/openvino.xml"]

    for xml_path in xml_paths:
        if not os.path.exists(xml_path):
            pytest.skip(reason=f"required file is not exist - {xml_path}")

    if "visual_prompting" == task_type:
        paths_to_ref_data = [
            os.path.join(
                otx_dir,
                "tests",
                "e2e/cli",
                task_type,
                "reference",
                template.model_template_id,
                "compressed_image_encoder.yml",
            ),
            os.path.join(
                otx_dir,
                "tests",
                "e2e/cli",
                task_type,
                "reference",
                template.model_template_id,
                "compressed_decoder.yml",
            ),
        ]
    else:
        paths_to_ref_data = [
            os.path.join(
                otx_dir, "tests", "e2e/cli", task_type, "reference", template.model_template_id, "compressed_model.yml"
            )
        ]

    compression_type = "ptq_xpu" if is_xpu_available() else "ptq"
    for xml_path, path_to_ref_data in zip(xml_paths, paths_to_ref_data):
        _validate_fq_in_xml(xml_path, path_to_ref_data, compression_type, test_name)


def ptq_eval_testing(template, root, otx_dir, args, is_visual_prompting=False):
    template_work_dir = get_template_dir(template, root)
    if is_visual_prompting:
        weights_path = f"{template_work_dir}/ptq_{template.model_template_id}/visual_prompting_decoder.xml"
    else:
        weights_path = f"{template_work_dir}/ptq_{template.model_template_id}/openvino.xml"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--output",
        f"{template_work_dir}/ptq_{template.model_template_id}",
        "--load-weights",
        weights_path,
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/ptq_{template.model_template_id}/performance.json")


def nncf_optimize_testing(template, root, otx_dir, args):
    if template.entrypoints.nncf is None:
        pytest.skip("NNCF QAT is disabled: entrypoints.nncf in template is not specified")

    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "optimize",
        template.model_template_path,
        "--train-data-roots",
        f'{os.path.join(otx_dir, args["--train-data-roots"])}',
        "--val-data-roots",
        f'{os.path.join(otx_dir, args["--val-data-roots"])}',
        "--load-weights",
        weights_path,
        "--output",
        f"{template_work_dir}/nncf_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    command_line.extend(args["train_params"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth")
    assert os.path.exists(f"{template_work_dir}/nncf_{template.model_template_id}/label_schema.json")


def nncf_export_testing(template, root):
    if template.entrypoints.nncf is None:
        pytest.skip("NNCF QAT is disabled: entrypoints.nncf in template is not specified")
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "export",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--output",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}",
    ]
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml")
    assert os.path.exists(f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.bin")
    assert os.path.exists(f"{template_work_dir}/exported_nncf_{template.model_template_id}/label_schema.json")
    original_bin_size = os.path.getsize(f"{template_work_dir}/exported_{template.model_template_id}/openvino.bin")
    compressed_bin_size = os.path.getsize(
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.bin"
    )
    assert compressed_bin_size < original_bin_size, f"{compressed_bin_size=}, {original_bin_size=}"
    ckpt = torch.load(f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth")
    input_size = ckpt.get("input_size", None)
    if input_size:
        with open(
            f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml", encoding="utf-8"
        ) as xml_stream:
            xml_model = xml_stream.read()
            assert f"{input_size[1]},{input_size[0]}" in xml_model


def nncf_validate_fq_testing(template, root, otx_dir, task_type, test_name):
    if template.entrypoints.nncf is None:
        pytest.skip("NNCF QAT is disabled: entrypoints.nncf in template is not specified")
    template_work_dir = get_template_dir(template, root)

    xml_path = f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml"
    if not os.path.exists(xml_path):
        pytest.skip(reason=f"required file is not exist - {xml_path}")

    path_to_ref_data = os.path.join(
        otx_dir, "tests", "e2e/cli", task_type, "reference", template.model_template_id, "compressed_model.yml"
    )

    _validate_fq_in_xml(xml_path, path_to_ref_data, "nncf", test_name)


def nncf_eval_testing(template, root, otx_dir, args, threshold=0.01):
    if template.entrypoints.nncf is None:
        pytest.skip("NNCF QAT is disabled: entrypoints.nncf in template is not specified")
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        weights_path,
        "--output",
        f"{template_work_dir}/nncf_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/nncf_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/nncf_{template.model_template_id}/nncf_performance.json") as read_file:
        trained_performance = json.load(read_file)
    with open(f"{template_work_dir}/nncf_{template.model_template_id}/performance.json") as read_file:
        evaluated_performance = json.load(read_file)

    compare_model_accuracy(evaluated_performance, trained_performance, threshold)


def nncf_eval_openvino_testing(template, root, otx_dir, args):
    if template.entrypoints.nncf is None:
        pytest.skip("NNCF QAT is disabled: entrypoints.nncf in template is not specified")
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        weights_path,
        "--output",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/exported_nncf_{template.model_template_id}/performance.json")


def xfail_templates(templates, xfail_template_ids_reasons):
    xfailed_templates = []
    for template in templates:
        reasons = [
            reason for template_id, reason in xfail_template_ids_reasons if template_id == template.model_template_id
        ]
        if len(reasons) == 0:
            xfailed_templates.append(template)
        elif len(reasons) == 1:
            xfailed_templates.append(pytest.param(template, marks=pytest.mark.xfail(reason=reasons[0])))
        else:
            raise RuntimeError(
                "More than one reason for template. If you have more than one Jira tickets, list them in one reason."
            )
    return xfailed_templates


def otx_explain_testing(template, root, otx_dir, args, trained=False):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    test_algorithm = "ClassWiseSaliencyMap"

    train_ann_file = args.get("--train-ann-file", "")
    if "hierarchical" in train_ann_file:
        train_type = "hierarchical"
    elif "multilabel" in train_ann_file:
        train_type = "multilabel"
    else:
        train_type = "default"

    save_dir = f"explain_{template.model_template_id}/{test_algorithm}/{train_type}/"
    output_dir = os.path.join(template_work_dir, save_dir)
    data_input = os.path.join(otx_dir, args["--input"])
    command_line = [
        "otx",
        "explain",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        data_input,
        "--output",
        output_dir,
        "--explain-algorithm",
        test_algorithm,
    ]
    check_run(command_line)
    assert os.path.exists(output_dir)
    if trained:
        assert len(os.listdir(output_dir)) > 0
        assert all([os.path.splitext(fname)[1] in [".tiff", ".log"] for fname in os.listdir(output_dir)])


def otx_explain_testing_all_classes(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    test_algorithm = "ClassWiseSaliencyMap"

    train_ann_file = args.get("--train-ann-file", "")
    if "hierarchical" in train_ann_file:
        train_type = "hierarchical"
    elif "multilabel" in train_ann_file:
        train_type = "multilabel"
    else:
        train_type = "default"

    save_dir = f"explain_all_classes_{template.model_template_id}/{test_algorithm}/{train_type}/"
    output_dir = os.path.join(template_work_dir, save_dir)
    data_input = os.path.join(otx_dir, args["--input"])
    command_line = [
        "otx",
        "explain",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        data_input,
        "--output",
        output_dir,
        "--explain-algorithm",
        test_algorithm,
        "--explain-all-classes",
    ]
    check_run(command_line)
    assert os.path.exists(output_dir)

    save_dir_explain_only_predicted_classes = f"explain_{template.model_template_id}/{test_algorithm}/{train_type}/"
    output_dir_explain_only_predicted_classes = os.path.join(template_work_dir, save_dir_explain_only_predicted_classes)
    if test_algorithm == "ActivationMap":
        assert len(os.listdir(output_dir)) == len(os.listdir(output_dir_explain_only_predicted_classes))
    else:
        assert len(os.listdir(output_dir)) >= len(os.listdir(output_dir_explain_only_predicted_classes))
    assert all([os.path.splitext(fname)[1] in [".tiff", ".log"] for fname in os.listdir(output_dir)])


def otx_explain_testing_process_saliency_maps(template, root, otx_dir, args, trained=False):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    test_algorithm = "ClassWiseSaliencyMap"

    train_ann_file = args.get("--train-ann-file", "")
    if "hierarchical" in train_ann_file:
        train_type = "hierarchical"
    elif "multilabel" in train_ann_file:
        train_type = "multilabel"
    else:
        train_type = "default"

    save_dir = f"explain_process_saliency_maps_{template.model_template_id}/{test_algorithm}/{train_type}/"
    output_dir = os.path.join(template_work_dir, save_dir)
    data_input = os.path.join(otx_dir, args["--input"])
    command_line = [
        "otx",
        "explain",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        data_input,
        "--output",
        output_dir,
        "--explain-algorithm",
        test_algorithm,
        "--process-saliency-maps",
    ]
    check_run(command_line)
    assert os.path.exists(output_dir)
    if trained:
        assert len(os.listdir(output_dir)) > 0
        assert all([os.path.splitext(fname)[1] in [".png", ".log"] for fname in os.listdir(output_dir)])


def otx_explain_openvino_testing(template, root, otx_dir, args, trained=False):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    test_algorithm = "ClassWiseSaliencyMap"

    train_ann_file = args.get("--train-ann-file", "")
    if "hierarchical" in train_ann_file:
        train_type = "hierarchical"
    elif "multilabel" in train_ann_file:
        train_type = "multilabel"
    else:
        train_type = "default"

    save_dir = f"explain_ov_{template.model_template_id}/{test_algorithm}/{train_type}/"
    output_dir = os.path.join(template_work_dir, save_dir)
    data_input = os.path.join(otx_dir, args["--input"])
    command_line = [
        "otx",
        "explain",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        data_input,
        "--output",
        output_dir,
        "--explain-algorithm",
        test_algorithm,
    ]
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml")
    check_run(command_line)
    assert os.path.exists(output_dir)
    if trained:
        assert len(os.listdir(output_dir)) > 0
        assert all([os.path.splitext(fname)[1] in [".tiff", ".log"] for fname in os.listdir(output_dir)])


def otx_explain_all_classes_openvino_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    test_algorithm = "ClassWiseSaliencyMap"

    train_ann_file = args.get("--train-ann-file", "")
    if "hierarchical" in train_ann_file:
        train_type = "hierarchical"
    elif "multilabel" in train_ann_file:
        train_type = "multilabel"
    else:
        train_type = "default"

    save_dir = f"explain_ov_all_classes_{template.model_template_id}/{test_algorithm}/{train_type}/"
    output_dir = os.path.join(template_work_dir, save_dir)
    data_input = os.path.join(otx_dir, args["--input"])
    command_line = [
        "otx",
        "explain",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        data_input,
        "--output",
        output_dir,
        "--explain-algorithm",
        test_algorithm,
        "--explain-all-classes",
    ]
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml")
    check_run(command_line)
    assert os.path.exists(output_dir)

    save_dir_explain_only_predicted_classes = f"explain_ov_{template.model_template_id}/{test_algorithm}/{train_type}/"
    output_dir_explain_only_predicted_classes = os.path.join(template_work_dir, save_dir_explain_only_predicted_classes)
    if test_algorithm == "ActivationMap":
        assert len(os.listdir(output_dir)) == len(os.listdir(output_dir_explain_only_predicted_classes))
    else:
        assert len(os.listdir(output_dir)) >= len(os.listdir(output_dir_explain_only_predicted_classes))
    assert all([os.path.splitext(fname)[1] in [".tiff", ".log"] for fname in os.listdir(output_dir)])


def otx_explain_process_saliency_maps_openvino_testing(template, root, otx_dir, args, trained=False):
    template_work_dir = get_template_dir(template, root)

    weights_path = f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml"
    if not os.path.exists(weights_path):
        pytest.skip(reason=f"required file is not exist - {weights_path}")

    test_algorithm = "ClassWiseSaliencyMap"

    train_ann_file = args.get("--train-ann-file", "")
    if "hierarchical" in train_ann_file:
        train_type = "hierarchical"
    elif "multilabel" in train_ann_file:
        train_type = "multilabel"
    else:
        train_type = "default"

    save_dir = f"explain_ov_process_saliency_maps_{template.model_template_id}/{test_algorithm}/{train_type}/"
    output_dir = os.path.join(template_work_dir, save_dir)
    data_input = os.path.join(otx_dir, args["--input"])
    command_line = [
        "otx",
        "explain",
        template.model_template_path,
        "--load-weights",
        weights_path,
        "--input",
        data_input,
        "--output",
        output_dir,
        "--explain-algorithm",
        test_algorithm,
        "--process-saliency-maps",
    ]
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml")
    check_run(command_line)
    assert os.path.exists(output_dir)
    if trained:
        assert len(os.listdir(output_dir)) > 0
        assert all([os.path.splitext(fname)[1] in [".png", ".log"] for fname in os.listdir(output_dir)])


def otx_find_testing():
    """Performs several options of available otx find."""
    # Find all model template
    command_line = ["otx", "find", "--template"]
    check_run(command_line)

    # Find command per tasks
    for task in find_supported_tasks:
        command_line = ["otx", "find", "--template", "--task", task]
        check_run(command_line)

    # Find Backbones per backends
    for backbone_backends in find_supported_backends:
        command_line = [
            "otx",
            "find",
            "--backbone",
            backbone_backends,
        ]
        check_run(command_line)


def otx_build_task_testing(root, task):
    """Build OTX-workspace per tasks.

    Build and verify the otx-workspace corresponding to each task.
    """
    # Build otx-workspace per tasks check - Default Model Template only
    command_line = [
        "otx",
        "build",
        "--task",
        task,
        "--workspace",
        os.path.join(root, f"otx-workspace-{task}"),
    ]
    check_run(command_line)


def otx_build_backbone_testing(root, backbone_args):
    """Build backbone & Update model testing.

    Build each backbone to create backbone.yaml into workspace,
    build for the default model for the task,
    and even test updating the model config.
    This is done on the premise that the otx_workspace
    has been created well through otx_build_task_testing.
    """
    task, backbone = backbone_args
    task_workspace = os.path.join(root, f"otx-workspace-{task}")
    command_line = [
        "otx",
        "build",
        "--task",
        f"{task}",
        "--workspace",
        task_workspace,
    ]
    check_run(command_line)
    assert os.path.exists(task_workspace)

    # Build model.py from backbone type
    command_line = [
        "otx",
        "build",
        "--backbone",
        backbone,
        "--workspace",
        task_workspace,
    ]
    check_run(command_line)
    from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig

    model_config = OTXConfig.fromfile(os.path.join(task_workspace, "model.py"))
    assert os.path.exists(os.path.join(task_workspace, "model.py"))
    assert "backbone" in model_config["model"], "'backbone' is not in model configs"
    assert (
        model_config["model"]["backbone"]["type"] == backbone
    ), f"{model_config['model']['backbone']['type']} != {backbone}"


def otx_build_testing(root, args: Dict[str, str], expected: Dict[str, str]):
    workspace_root = os.path.join(root, "otx-workspace")
    command_line = ["otx", "build", "--workspace", workspace_root]
    for option, value in args.items():
        command_line.extend([option, value])
    check_run(command_line)
    from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig

    template_config = OTXConfig.fromfile(os.path.join(workspace_root, "template.yaml"))
    assert template_config.name == expected["model"]
    assert (
        template_config.hyper_parameters.parameter_overrides.algo_backend.train_type.default_value
        == expected["train_type"]
    )


def otx_build_auto_config(root, otx_dir: str, args: Dict[str, str]):
    workspace_root = os.path.join(root, "otx-workspace")
    command_line = ["otx", "build", "--workspace", workspace_root]

    for option, value in args.items():
        if option in ["--train-data-roots", "--val-data-roots"]:
            command_line.extend([option, f"{os.path.join(otx_dir, value)}"])
        elif option in ["--task"]:
            command_line.extend([option, args[option]])
    check_run(command_line)


def otx_train_auto_config(root, otx_dir: str, args: Dict[str, str], use_output: bool = True):
    work_dir = os.path.join(root, "otx-workspace")
    command_line = ["otx", "train"]

    for option, value in args.items():
        if option == "template":
            command_line.extend([args[option]])
        elif option in ["--train-data-roots", "--val-data-roots"]:
            command_line.extend([option, f"{os.path.join(otx_dir, value)}"])
    if use_output:
        command_line.extend(["--output", f"{work_dir}"])
    command_line.extend(["--workspace", f"{work_dir}"])
    command_line.extend(args["train_params"])
    check_run(command_line)


def generate_model_template_testing(templates):
    class _TestModelTemplates:
        @e2e_pytest_component
        def test_model_category(self):
            stat = {
                ModelCategory.SPEED: 0,
                ModelCategory.BALANCE: 0,
                ModelCategory.ACCURACY: 0,
                ModelCategory.OTHER: 0,
            }
            for template in templates:
                stat[template.model_category] += 1
            assert stat[ModelCategory.SPEED] == 1
            assert stat[ModelCategory.BALANCE] <= 1
            assert stat[ModelCategory.ACCURACY] == 1

        @e2e_pytest_component
        def test_model_status(self):
            for template in templates:
                if template.model_status == ModelStatus.DEPRECATED:
                    assert template.model_category == ModelCategory.OTHER

        @e2e_pytest_component
        def test_default_for_task(self):
            num_default_model = 0
            for template in templates:
                if template.is_default_for_task:
                    num_default_model += 1
                    assert template.model_category != ModelCategory.OTHER
                    assert template.model_status == ModelStatus.ACTIVE
            assert num_default_model == 1

    return _TestModelTemplates


def compare_model_accuracy(performance_to_test: Dict, target_performance: Dict, threshold: Union[float, int]):
    for k in target_performance.keys():
        if k == "avg_time_per_image":
            continue
        assert (
            performance_to_test[k] >= target_performance[k]
            or abs(target_performance[k] - performance_to_test[k]) / (target_performance[k] + 1e-10) <= threshold
        ), f"{target_performance[k]=}, {performance_to_test[k]=}"
