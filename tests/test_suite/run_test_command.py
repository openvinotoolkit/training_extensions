# Copyright (C) 2021 Intel Corporation
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

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict

import pytest
import yaml

from otx.cli.tools.find import SUPPORTED_BACKBONE_BACKENDS as find_supported_backends
from otx.cli.tools.find import SUPPORTED_TASKS as find_supported_tasks
from otx.cli.utils.nncf import get_number_of_fakequantizers_in_xml


def get_template_rel_dir(template):
    return os.path.dirname(os.path.relpath(template.model_template_path))


def get_template_dir(template, root) -> str:

    # Get the template directory of the algorithm.
    # The location of the template files are as follows:
    # ~/training_extensions/otx/algorithms/<algorithm>/**/template.yaml
    # To get the ``algorithm``, index of the "algorithms" can be
    # searched, where ``algorithm`` comes next.
    template_path_parts = template.model_template_path.split(os.sep)
    idx = template_path_parts.index("algorithms")
    algorithm = template_path_parts[idx + 1]

    algo_backend_dir = f"otx/algorithms/{algorithm}"
    work_dir = os.path.join(root, f"otx/algorithms/{algorithm}")
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


def otx_train_testing(template, root, otx_dir, args):
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
    command_line.extend(["--save-model-to", f"{template_work_dir}/trained_{template.model_template_id}"])
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    if "--load-weights" in args:
        command_line.extend(["--load-weights", args["--load-weights"]])
    if "--gpus" in args:
        command_line.extend(["--gpus", args["--gpus"]])
        if "--multi-gpu-port" in args:
            command_line.extend(["--multi-gpu-port", args["--multi-gpu-port"]])
    if "train_params" in args:
        command_line.extend(args["train_params"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/weights.pth")
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/label_schema.json")


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

    command_line.extend(["--save-model-to", f"{template_work_dir}/trained_for_resume_{template.model_template_id}"])
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    command_line.extend(args["train_params"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/trained_for_resume_{template.model_template_id}/weights.pth")
    assert os.path.exists(f"{template_work_dir}/trained_for_resume_{template.model_template_id}/label_schema.json")


def otx_hpo_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    if os.path.exists(f"{template_work_dir}/hpo"):
        shutil.rmtree(f"{template_work_dir}/hpo")

    command_line = ["otx", "train", template.model_template_path]

    for arg in ["--train-data-roots", "--val-data-roots"]:
        arg_value = args.get(arg, None)
        if arg_value:
            command_line.extend([arg, os.path.join(otx_dir, arg_value)])
    command_line.extend(["--save-model-to", f"{template_work_dir}/hpo_trained_{template.model_template_id}"])
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    command_line.extend(["--enable-hpo", "--hpo-time-ratio", "1"])

    command_line.extend(args["train_params"])
    check_run(command_line)
    trials_json = list(
        filter(lambda x: x.name.split(".")[0].isnumeric(), Path(f"{template_work_dir}/hpo/").rglob("*.json"))
    )
    assert trials_json
    for trial_json in trials_json:
        with trial_json.open("r") as f:
            trial_result = json.load(f)
        assert trial_result.get("score")

    assert os.path.exists(f"{template_work_dir}/hpo_trained_{template.model_template_id}/weights.pth")
    assert os.path.exists(f"{template_work_dir}/hpo_trained_{template.model_template_id}/label_schema.json")


def otx_export_testing(template, root):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "export",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-model-to",
        f"{template_work_dir}/exported_{template.model_template_id}",
    ]
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml")
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}/openvino.bin")
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}/label_schema.json")


def otx_export_testing_w_features(template, root):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "export",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-model-to",
        f"{template_work_dir}/exported_{template.model_template_id}_w_features",
        "--dump-features",
    ]
    check_run(command_line)

    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.bin")
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}_w_features/label_schema.json")

    path_to_xml = f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml"
    assert os.path.exists(path_to_xml)
    with open(path_to_xml, encoding="utf-8") as stream:
        xml_model = stream.read()
    assert "feature_vector" in xml_model


def otx_eval_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-performance",
        f"{template_work_dir}/trained_{template.model_template_id}/performance.json",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    command_line.extend(args.get("eval_params", []))
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/performance.json")


def otx_eval_openvino_testing(
    template, root, otx_dir, args, threshold=0.0, criteria=None, reg_threshold=0.10, result_dict=None
):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/exported_{template.model_template_id}/performance.json",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/trained_{template.model_template_id}/performance.json") as read_file:
        trained_performance = json.load(read_file)
    with open(f"{template_work_dir}/exported_{template.model_template_id}/performance.json") as read_file:
        exported_performance = json.load(read_file)

    if isinstance(criteria, dict) and template.name in criteria.keys():
        model_criteria = criteria[template.name]
        modified_criteria = model_criteria - (model_criteria * reg_threshold)

    for k in trained_performance.keys():
        if isinstance(criteria, dict) and template.name in criteria.keys():
            result_dict[k] = round(exported_performance[k], 3)
            assert (
                exported_performance[k] >= modified_criteria
            ), f"Current exported model performance: ({exported_performance[k]}) < criteria: ({modified_criteria})."

        assert (
            exported_performance[k] >= trained_performance[k]
            or abs(trained_performance[k] - exported_performance[k]) / (trained_performance[k] + 1e-10) <= threshold
        ), f"{trained_performance[k]=}, {exported_performance[k]=}"


def otx_demo_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "demo",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--input",
        os.path.join(otx_dir, args["--input"]),
        "--delay",
        "-1",
    ]
    check_run(command_line)


def otx_demo_openvino_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "demo",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--input",
        os.path.join(otx_dir, args["--input"]),
        "--delay",
        "-1",
    ]
    check_run(command_line)


def otx_deploy_openvino_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    deployment_dir = f"{template_work_dir}/deployed_{template.model_template_id}"
    command_line = [
        "otx",
        "deploy",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--save-model-to",
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
            "--no_show",
        ],
        cwd=os.path.join(deployment_dir, "python"),
    )


def otx_eval_deployment_testing(
    template, root, otx_dir, args, threshold=0.0, criteria=None, reg_threshold=0.10, result_dict=None
):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/deployed_{template.model_template_id}/openvino.zip",
        "--save-performance",
        f"{template_work_dir}/deployed_{template.model_template_id}/performance.json",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/deployed_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/exported_{template.model_template_id}/performance.json") as read_file:
        exported_performance = json.load(read_file)
    with open(f"{template_work_dir}/deployed_{template.model_template_id}/performance.json") as read_file:
        deployed_performance = json.load(read_file)

    if isinstance(criteria, dict) and template.name in criteria.keys():
        model_criteria = criteria[template.name]
        modified_criteria = model_criteria - (model_criteria * reg_threshold)

    for k in exported_performance.keys():
        if isinstance(criteria, dict) and template.name in criteria.keys():
            result_dict[k] = round(deployed_performance[k], 3)
            assert (
                exported_performance[k] >= modified_criteria
            ), f"Current deployed model performance: ({deployed_performance[k]}) < criteria: ({modified_criteria})."
        assert (
            deployed_performance[k] >= exported_performance[k]
            or abs(exported_performance[k] - deployed_performance[k]) / (exported_performance[k] + 1e-10) <= threshold
        ), f"{exported_performance[k]=}, {deployed_performance[k]=}"


def otx_demo_deployment_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "demo",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/deployed_{template.model_template_id}/openvino.zip",
        "--input",
        os.path.join(otx_dir, args["--input"]),
        "--delay",
        "-1",
    ]
    check_run(command_line)


def pot_optimize_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "optimize",
        template.model_template_path,
        "--train-data-roots",
        f'{os.path.join(otx_dir, args["--train-data-roots"])}',
        "--val-data-roots",
        f'{os.path.join(otx_dir, args["--val-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--save-model-to",
        f"{template_work_dir}/pot_{template.model_template_id}",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/pot_{template.model_template_id}/openvino.xml")
    assert os.path.exists(f"{template_work_dir}/pot_{template.model_template_id}/openvino.bin")
    assert os.path.exists(f"{template_work_dir}/pot_{template.model_template_id}/label_schema.json")


def _validate_fq_in_xml(xml_path, path_to_ref_data, compression_type, test_name):
    num_fq = get_number_of_fakequantizers_in_xml(xml_path)
    assert os.path.exists(path_to_ref_data), f"Reference file does not exist: {path_to_ref_data} [num_fq = {num_fq}]"

    with open(path_to_ref_data, encoding="utf-8") as stream:
        ref_data = yaml.safe_load(stream)
    ref_num_fq = ref_data.get(test_name, {}).get(compression_type, {}).get("number_of_fakequantizers", -1)
    assert num_fq == ref_num_fq, f"Incorrect number of FQs in optimized model: {num_fq} != {ref_num_fq}"


def pot_validate_fq_testing(template, root, otx_dir, task_type, test_name):
    template_work_dir = get_template_dir(template, root)
    xml_path = f"{template_work_dir}/pot_{template.model_template_id}/openvino.xml"
    path_to_ref_data = os.path.join(
        otx_dir, "tests", "e2e/cli", task_type, "reference", template.model_template_id, "compressed_model.yml"
    )
    _validate_fq_in_xml(xml_path, path_to_ref_data, "pot", test_name)


def pot_eval_testing(template, root, otx_dir, args, criteria=None, reg_threshold=0.10, result_dict=None):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/pot_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/pot_{template.model_template_id}/performance.json",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/pot_{template.model_template_id}/performance.json")

    with open(f"{template_work_dir}/pot_{template.model_template_id}/performance.json") as read_file:
        pot_performance = json.load(read_file)

    if isinstance(criteria, dict) and template.name in criteria.keys():
        model_criteria = criteria[template.name]
        modified_criteria = model_criteria - (model_criteria * reg_threshold)

    for k in pot_performance.keys():
        if isinstance(criteria, dict) and template.name in criteria.keys():
            result_dict[k] = round(pot_performance[k], 3)
            assert (
                pot_performance[k] >= modified_criteria
            ), f"Current POT model performance: ({pot_performance[k]}) < criteria: ({modified_criteria})."


def nncf_optimize_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "optimize",
        template.model_template_path,
        "--train-data-roots",
        f'{os.path.join(otx_dir, args["--train-data-roots"])}',
        "--val-data-roots",
        f'{os.path.join(otx_dir, args["--val-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-model-to",
        f"{template_work_dir}/nncf_{template.model_template_id}",
        "--save-performance",
        f"{template_work_dir}/nncf_{template.model_template_id}/train_performance.json",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    command_line.extend(args["train_params"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth")
    assert os.path.exists(f"{template_work_dir}/nncf_{template.model_template_id}/label_schema.json")


def nncf_export_testing(template, root):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "export",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth",
        "--save-model-to",
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


def nncf_validate_fq_testing(template, root, otx_dir, task_type, test_name):
    template_work_dir = get_template_dir(template, root)
    xml_path = f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml"
    path_to_ref_data = os.path.join(
        otx_dir, "tests", "e2e/cli", task_type, "reference", template.model_template_id, "compressed_model.yml"
    )

    _validate_fq_in_xml(xml_path, path_to_ref_data, "nncf", test_name)


def nncf_eval_testing(
    template, root, otx_dir, args, threshold=0.001, criteria=None, reg_threshold=0.10, result_dict=None
):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth",
        "--save-performance",
        f"{template_work_dir}/nncf_{template.model_template_id}/performance.json",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/nncf_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/nncf_{template.model_template_id}/train_performance.json") as read_file:
        trained_performance = json.load(read_file)
    with open(f"{template_work_dir}/nncf_{template.model_template_id}/performance.json") as read_file:
        evaluated_performance = json.load(read_file)

    if isinstance(criteria, dict) and template.name in criteria.keys():
        model_criteria = criteria[template.name]
        modified_criteria = model_criteria - (model_criteria * reg_threshold)

    for k in trained_performance.keys():
        if isinstance(criteria, dict) and template.name in criteria.keys():
            result_dict[k] = round(evaluated_performance[k], 3)
            assert (
                evaluated_performance[k] >= modified_criteria
            ), f"Current nncf model performance: ({evaluated_performance[k]}) < criteria: ({modified_criteria})."
        assert (
            evaluated_performance[k] >= trained_performance[k]
            or abs(trained_performance[k] - evaluated_performance[k]) / (trained_performance[k] + 1e-10) <= threshold
        ), f"{trained_performance[k]=}, {evaluated_performance[k]=}"


def nncf_eval_openvino_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/performance.json",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
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


def otx_explain_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    if "RCNN" in template.model_template_id:
        test_algorithm = "ActivationMap"
    else:
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
    command_line = [
        "otx",
        "explain",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--explain-data-root",
        os.path.join(otx_dir, args["--input"]),
        "--save-explanation-to",
        output_dir,
        "--explain-algorithm",
        test_algorithm,
    ]
    check_run(command_line)
    assert os.path.exists(output_dir)
    assert len(os.listdir(output_dir)) > 0


def otx_explain_openvino_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    if "RCNN" in template.model_template_id:
        test_algorithm = "ActivationMap"
    else:
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
    command_line = [
        "otx",
        "explain",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml",
        "--explain-data-root",
        os.path.join(otx_dir, args["--input"]),
        "--save-explanation-to",
        output_dir,
        "--explain-algorithm",
        test_algorithm,
    ]
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}_w_features/openvino.xml")
    check_run(command_line)
    assert os.path.exists(output_dir)
    assert len(os.listdir(output_dir)) > 0


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
        "--work-dir",
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
        "--work-dir",
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
        "--work-dir",
        task_workspace,
    ]
    check_run(command_line)
    from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig

    model_config = MPAConfig.fromfile(os.path.join(task_workspace, "model.py"))
    assert os.path.exists(os.path.join(task_workspace, "model.py"))
    assert "backbone" in model_config["model"], "'backbone' is not in model configs"
    assert (
        model_config["model"]["backbone"]["type"] == backbone
    ), f"{model_config['model']['backbone']['type']} != {backbone}"


def otx_build_testing(root, args: Dict[str, str], expected: Dict[str, str]):
    workspace_root = os.path.join(root, "otx-workspace")
    command_line = ["otx", "build", "--work-dir", workspace_root]
    for option, value in args.items():
        command_line.extend([option, value])
    check_run(command_line)
    from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig

    template_config = MPAConfig.fromfile(os.path.join(workspace_root, "template.yaml"))
    assert template_config.name == expected["model"]
    assert (
        template_config.hyper_parameters.parameter_overrides.algo_backend.train_type.default_value
        == expected["train_type"]
    )


def otx_build_auto_config(root, otx_dir: str, args: Dict[str, str]):
    workspace_root = os.path.join(root, "otx-workspace")
    command_line = ["otx", "build", "--work-dir", workspace_root]

    for option, value in args.items():
        if option in ["--train-data-roots", "--val-data-roots"]:
            command_line.extend([option, f"{os.path.join(otx_dir, value)}"])
        elif option in ["--task"]:
            command_line.extend([option, args[option]])
    check_run(command_line)


def otx_train_auto_config(root, otx_dir: str, args: Dict[str, str]):
    work_dir = os.path.join(root, "otx-workspace")
    command_line = ["otx", "train"]

    for option, value in args.items():
        if option == "template":
            command_line.extend([args[option]])
        elif option in ["--train-data-roots", "--val-data-roots"]:
            command_line.extend([option, f"{os.path.join(otx_dir, value)}"])
    command_line.extend(["--save-model-to", f"{work_dir}"])
    command_line.extend(["--work-dir", f"{work_dir}"])
    command_line.extend(args["train_params"])
    check_run(command_line)


def otx_eval_compare(
    template,
    root,
    otx_dir,
    args,
    criteria,
    result_dict,
    threshold=0.10,
):
    template_work_dir = get_template_dir(template, root)

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-performance",
        f"{template_work_dir}/trained_{template.model_template_id}/performance.json",
    ]
    command_line.extend(["--work-dir", f"{template_work_dir}"])
    command_line.extend(args.get("eval_params", []))
    check_run(command_line)

    performance_json_path = f"{template_work_dir}/trained_{template.model_template_id}/performance.json"
    assert os.path.exists(performance_json_path)

    with open(performance_json_path) as read_file:
        trained_performance = json.load(read_file)

    model_criteria = criteria[template.name]
    modified_criteria = model_criteria - (model_criteria * threshold)
    for k in trained_performance.keys():
        result_dict[k] = round(trained_performance[k], 3)
        assert (
            trained_performance[k] >= modified_criteria
        ), f"Current model performance: ({trained_performance[k]}) < criteria: ({modified_criteria})."

    result_dict["Model size (MB)"] = round(
        os.path.getsize(f"{template_work_dir}/trained_{template.model_template_id}/weights.pth") / 1e6, 2
    )


def otx_eval_e2e_train_time(train_time_criteria, e2e_train_time, template, threshold=0.10):
    """Measure train+val time and comapre with test criteria.

    Test criteria was set by previous measurement.
    """
    e2e_train_time_criteria = train_time_criteria[template.name]
    modified_train_criteria = e2e_train_time_criteria - (e2e_train_time_criteria * threshold)

    assert (
        e2e_train_time >= modified_train_criteria
    ), f"Current model e2e time: ({e2e_train_time}) < criteria: ({modified_train_criteria})."


def otx_eval_e2e_eval_time(eval_time_criteria, e2e_eval_time, template, threshold=0.10):
    """Measure evaluation time and comapre with test criteria.

    Test criteria was set by previous measurement.
    """
    e2e_eval_time_criteria = eval_time_criteria[template.name]
    modified_eval_criteria = e2e_eval_time_criteria - (e2e_eval_time_criteria * threshold)

    assert (
        e2e_eval_time >= modified_eval_criteria
    ), f"Current model e2e time: ({e2e_eval_time}) < criteria: ({modified_eval_criteria})."
