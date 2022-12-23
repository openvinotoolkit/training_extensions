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

import json
import os
import shutil
import subprocess  # nosec

import pytest

from otx.cli.tools.build import SUPPORTED_TASKS as build_supported_tasks
from otx.cli.tools.find import SUPPORTED_BACKBONE_BACKENDS as find_supported_backends
from otx.cli.tools.find import SUPPORTED_TASKS as find_supported_tasks


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


def check_run(cmd, **kwargs):
    result = subprocess.run(cmd, stderr=subprocess.PIPE, **kwargs)
    assert result.returncode == 0, result.stderr.decode("utf=8")


def otx_train_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "train",
        template.model_template_path,
        "--train-ann-file",
        f'{os.path.join(otx_dir, args["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(otx_dir, args["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(otx_dir, args["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(otx_dir, args["--val-data-roots"])}',
        "--save-model-to",
        f"{template_work_dir}/trained_{template.model_template_id}",
    ]
    if "--unlabeled-data-roots" in args:
        command_line.extend(["--unlabeled-data-roots", f'{os.path.join(otx_dir, args["--unlabeled-data-roots"])}'])
    if "--unlabeled-file-list" in args:
        command_line.extend(["--unlabeled-file-list"], f'{os.path.join(otx_dir, args["--unlabeled-file-list"])}')
    if "--load-weights" in args:
        command_line.extend(["--load-weights", f'{os.path.join(otx_dir, args["--load-weights"])}'])
    if "--unlabeled-data-roots" in args:
        command_line.extend(["--unlabeled-data-roots", f'{os.path.join(otx_dir, args["--unlabeled-data-roots"])}'])
    if "--unlabeled-file-list" in args:
        command_line.extend(["--unlabeled-file-list"], f'{os.path.join(otx_dir, args["--unlabeled-file-list"])}')
    command_line.extend(args["train_params"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/weights.pth")
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/label_schema.json")


def otx_hpo_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    if os.path.exists(f"{template_work_dir}/hpo"):
        shutil.rmtree(f"{template_work_dir}/hpo")
    command_line = [
        "otx",
        "train",
        template.model_template_path,
        "--train-ann-file",
        f'{os.path.join(otx_dir, args["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(otx_dir, args["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(otx_dir, args["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(otx_dir, args["--val-data-roots"])}',
        "--save-model-to",
        f"{template_work_dir}/hpo_trained_{template.model_template_id}",
        "--enable-hpo",
        "--hpo-time-ratio",
        "1",
    ]
    command_line.extend(args["train_params"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/hpo/hpopt_status.json")
    with open(f"{template_work_dir}/hpo/hpopt_status.json", "r") as f:
        assert json.load(f).get("best_config_id", None) is not None
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


def otx_eval_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(otx_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-performance",
        f"{template_work_dir}/trained_{template.model_template_id}/performance.json",
    ]
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/trained_{template.model_template_id}/performance.json")


def otx_eval_openvino_testing(template, root, otx_dir, args, threshold):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(otx_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/exported_{template.model_template_id}/performance.json",
    ]
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/exported_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/trained_{template.model_template_id}/performance.json") as read_file:
        trained_performance = json.load(read_file)
    with open(f"{template_work_dir}/exported_{template.model_template_id}/performance.json") as read_file:
        exported_performance = json.load(read_file)

    for k in trained_performance.keys():
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


def otx_eval_deployment_testing(template, root, otx_dir, args, threshold):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(otx_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/deployed_{template.model_template_id}/openvino.zip",
        "--save-performance",
        f"{template_work_dir}/deployed_{template.model_template_id}/performance.json",
    ]
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/deployed_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/exported_{template.model_template_id}/performance.json") as read_file:
        exported_performance = json.load(read_file)
    with open(f"{template_work_dir}/deployed_{template.model_template_id}/performance.json") as read_file:
        deployed_performance = json.load(read_file)

    for k in exported_performance.keys():
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
        "--train-ann-file",
        f'{os.path.join(otx_dir, args["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(otx_dir, args["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(otx_dir, args["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(otx_dir, args["--val-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--save-model-to",
        f"{template_work_dir}/pot_{template.model_template_id}",
    ]
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/pot_{template.model_template_id}/openvino.xml")
    assert os.path.exists(f"{template_work_dir}/pot_{template.model_template_id}/openvino.bin")
    assert os.path.exists(f"{template_work_dir}/pot_{template.model_template_id}/label_schema.json")


def pot_eval_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(otx_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/pot_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/pot_{template.model_template_id}/performance.json",
    ]
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/pot_{template.model_template_id}/performance.json")


def nncf_optimize_testing(template, root, otx_dir, args):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "optimize",
        template.model_template_path,
        "--train-ann-file",
        f'{os.path.join(otx_dir, args["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(otx_dir, args["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(otx_dir, args["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(otx_dir, args["--val-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-model-to",
        f"{template_work_dir}/nncf_{template.model_template_id}",
        "--save-performance",
        f"{template_work_dir}/nncf_{template.model_template_id}/train_performance.json",
    ]
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


def nncf_eval_testing(template, root, otx_dir, args, threshold):
    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(otx_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth",
        "--save-performance",
        f"{template_work_dir}/nncf_{template.model_template_id}/performance.json",
    ]
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/nncf_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/nncf_{template.model_template_id}/train_performance.json") as read_file:
        trained_performance = json.load(read_file)
    with open(f"{template_work_dir}/nncf_{template.model_template_id}/performance.json") as read_file:
        evaluated_performance = json.load(read_file)

    for k in trained_performance.keys():
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
        "--test-ann-file",
        f'{os.path.join(otx_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/performance.json",
    ]
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
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
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


def otx_find_testing():
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


def otx_build_testing(root, args):
    # Build otx-workspace per tasks check - Default Model Template only
    for task in build_supported_tasks:
        command_line = [
            "otx",
            "build",
            "--task",
            task,
            "--workspace-root",
            os.path.join(root, f"otx-workspace-{task}"),
        ]
        check_run(command_line)

    for task, backbone in args.items():
        task_workspace = os.path.join(root, f"otx-workspace-{task}")
        # Build Backbone.yaml from backbone type
        command_line = [
            "otx",
            "build",
            "--backbone",
            backbone,
            "--workspace-root",
            task_workspace,
            "--save-backbone-to",
            os.path.join(task_workspace, "backbone.yaml"),
        ]
        check_run(command_line)
        assert os.path.exists(os.path.join(task_workspace, "backbone.yaml"))

        # Build model.py from backbone.yaml
        command_line = [
            "otx",
            "build",
            "--model",
            os.path.join(task_workspace, "model.py"),
            "--backbone",
            os.path.join(task_workspace, "backbone.yaml"),
            "--workspace-root",
            task_workspace,
        ]
        check_run(command_line)

        # Build model.py from backbone type
        command_line = [
            "otx",
            "build",
            "--model",
            os.path.join(task_workspace, "model.py"),
            "--backbone",
            backbone,
            "--workspace-root",
            task_workspace,
        ]
        check_run(command_line)
