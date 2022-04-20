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
from subprocess import run  # nosec

import pytest


def get_template_rel_dir(template):
    return os.path.dirname(os.path.relpath(template.model_template_path))


def get_some_vars(template, root):
    template_dir = get_template_rel_dir(template)
    algo_backend_dir = "/".join(template_dir.split("/")[:2])
    work_dir = os.path.join(root, os.path.basename(algo_backend_dir))
    template_work_dir = os.path.join(work_dir, template_dir)
    os.makedirs(template_work_dir, exist_ok=True)
    return work_dir, template_work_dir, algo_backend_dir


def create_venv(algo_backend_dir, work_dir):
    venv_dir = f"{work_dir}/venv"
    if not os.path.exists(venv_dir):
        assert run([f"./{algo_backend_dir}/init_venv.sh", venv_dir]).returncode == 0
        assert (
            run(
                [f"{work_dir}/venv/bin/python", "-m", "pip", "install", "-e", "ote_cli"]
            ).returncode
            == 0
        )


def extract_export_vars(path):
    vars = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                line = line.replace("export ", "").split("=")
                assert len(line) == 2
                vars[line[0].strip()] = line[1].strip()
    return vars


def collect_env_vars(work_dir):
    vars = extract_export_vars(f"{work_dir}/venv/bin/activate")
    vars.update({"PATH": f"{work_dir}/venv/bin/:" + os.environ["PATH"]})
    vars_map = {
        "HTTP_PROXY": ["http_proxy", "HTTP_PROXY"],
        "HTTPS_PROXY": ["https_proxy", "HTTPS_PROXY"],
        "NO_PROXY": ["no_proxy", "NO_PROXY"],
    }
    for var, aliases in vars_map.items():
        for alias in aliases:
            if alias in os.environ:
                vars.update({var: os.environ[alias]})
                break
    return vars


def patch_demo_py(src_path, dst_path):
    with open(src_path) as read_file:
        content = [line for line in read_file]
        replaced = False
        for i, line in enumerate(content):
            if "visualizer = create_visualizer(models[-1].task_type)" in line:
                content[i] = "    visualizer = Visualizer(); visualizer.show = show\n"
                replaced = True
        assert replaced
        content = [
            "from ote_sdk.usecases.exportable_code.visualizers import Visualizer\n",
            "def show(self):\n",
            "    pass\n\n",
        ] + content
        with open(dst_path, "w") as write_file:
            write_file.write("".join(content))


def ote_train_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "train",
        template.model_template_path,
        "--train-ann-file",
        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
        "--save-model-to",
        f"{template_work_dir}/trained_{template.model_template_id}",
    ]
    if "--load-weights" in args:
        command_line.extend(
            ["--load-weights", f'{os.path.join(ote_dir, args["--load-weights"])}']
        )
    command_line.extend(args["train_params"])
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
    )
    assert os.path.exists(
        f"{template_work_dir}/trained_{template.model_template_id}/label_schema.json"
    )


def ote_hpo_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    if os.path.exists(f"{template_work_dir}/hpo"):
        shutil.rmtree(f"{template_work_dir}/hpo")
    command_line = [
        "ote",
        "train",
        template.model_template_path,
        "--train-ann-file",
        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
        "--save-model-to",
        f"{template_work_dir}/hpo_trained_{template.model_template_id}",
        "--enable-hpo",
        "--hpo-time-ratio",
        "1",
    ]
    command_line.extend(args["train_params"])
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f"{template_work_dir}/hpo/hpopt_status.json")
    with open(f"{template_work_dir}/hpo/hpopt_status.json", "r") as f:
        assert json.load(f).get("best_config_id", None) is not None
    assert os.path.exists(
        f"{template_work_dir}/hpo_trained_{template.model_template_id}/weights.pth"
    )
    assert os.path.exists(
        f"{template_work_dir}/hpo_trained_{template.model_template_id}/label_schema.json"
    )


def ote_export_testing(template, root):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "export",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-model-to",
        f"{template_work_dir}/exported_{template.model_template_id}",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml"
    )
    assert os.path.exists(
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.bin"
    )
    assert os.path.exists(
        f"{template_work_dir}/exported_{template.model_template_id}/label_schema.json"
    )


def ote_eval_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(ote_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(ote_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-performance",
        f"{template_work_dir}/trained_{template.model_template_id}/performance.json",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/trained_{template.model_template_id}/performance.json"
    )


def ote_eval_openvino_testing(template, root, ote_dir, args, threshold):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(ote_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(ote_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/exported_{template.model_template_id}/performance.json",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/exported_{template.model_template_id}/performance.json"
    )
    with open(
        f"{template_work_dir}/trained_{template.model_template_id}/performance.json"
    ) as read_file:
        trained_performance = json.load(read_file)
    with open(
        f"{template_work_dir}/exported_{template.model_template_id}/performance.json"
    ) as read_file:
        exported_performance = json.load(read_file)

    for k in trained_performance.keys():
        assert (
            abs(trained_performance[k] - exported_performance[k])
            / (trained_performance[k] + 1e-10)
            <= threshold
        ), f"{trained_performance[k]=}, {exported_performance[k]=}"


def ote_demo_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "demo",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--input",
        os.path.join(ote_dir, args["--input"]),
        "--delay",
        "-1",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0


def ote_demo_openvino_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "demo",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--input",
        os.path.join(ote_dir, args["--input"]),
        "--delay",
        "-1",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0


def ote_deploy_openvino_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    deployment_dir = f"{template_work_dir}/deployed_{template.model_template_id}"
    command_line = [
        "ote",
        "deploy",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--save-model-to",
        deployment_dir,
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert run(["unzip", "-o", "openvino.zip"], cwd=deployment_dir).returncode == 0
    assert (
        run(
            ["python3", "-m", "venv", "venv"],
            cwd=os.path.join(deployment_dir, "python"),
        ).returncode
        == 0
    )
    assert (
        run(
            ["python3", "-m", "pip", "install", "wheel"],
            cwd=os.path.join(deployment_dir, "python"),
            env=collect_env_vars(os.path.join(deployment_dir, "python")),
        ).returncode
        == 0
    )

    assert (
        run(
            ["python3", "-m", "pip", "install", "pip", "--upgrade"],
            cwd=os.path.join(deployment_dir, "python"),
            env=collect_env_vars(os.path.join(deployment_dir, "python")),
        ).returncode
        == 0
    )
    assert (
        run(
            [
                "python3",
                "-m",
                "pip",
                "install",
                "-r",
                os.path.join(deployment_dir, "python", "requirements.txt"),
            ],
            cwd=os.path.join(deployment_dir, "python"),
            env=collect_env_vars(os.path.join(deployment_dir, "python")),
        ).returncode
        == 0
    )

    # Patch demo since we are not able to run cv2.imshow on CI.
    patch_demo_py(
        os.path.join(deployment_dir, "python", "demo.py"),
        os.path.join(deployment_dir, "python", "demo_patched.py"),
    )

    assert (
        run(
            [
                "python3",
                "demo_patched.py",
                "-m",
                "../model",
                "-i",
                os.path.join(ote_dir, args["--input"]),
            ],
            cwd=os.path.join(deployment_dir, "python"),
            env=collect_env_vars(os.path.join(deployment_dir, "python")),
        ).returncode
        == 0
    )


def ote_eval_deployment_testing(template, root, ote_dir, args, threshold):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(ote_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(ote_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/deployed_{template.model_template_id}/openvino.zip",
        "--save-performance",
        f"{template_work_dir}/deployed_{template.model_template_id}/performance.json",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/deployed_{template.model_template_id}/performance.json"
    )
    with open(
        f"{template_work_dir}/exported_{template.model_template_id}/performance.json"
    ) as read_file:
        exported_performance = json.load(read_file)
    with open(
        f"{template_work_dir}/deployed_{template.model_template_id}/performance.json"
    ) as read_file:
        deployed_performance = json.load(read_file)

    for k in exported_performance.keys():
        assert (
            abs(exported_performance[k] - deployed_performance[k])
            / (exported_performance[k] + 1e-10)
            <= threshold
        ), f"{exported_performance[k]=}, {deployed_performance[k]=}"


def ote_demo_deployment_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "demo",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/deployed_{template.model_template_id}/openvino.zip",
        "--input",
        os.path.join(ote_dir, args["--input"]),
        "--delay",
        "-1",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0


def pot_optimize_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "optimize",
        template.model_template_path,
        "--train-ann-file",
        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
        "--save-model-to",
        f"{template_work_dir}/pot_{template.model_template_id}",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/pot_{template.model_template_id}/openvino.xml"
    )
    assert os.path.exists(
        f"{template_work_dir}/pot_{template.model_template_id}/openvino.bin"
    )
    assert os.path.exists(
        f"{template_work_dir}/pot_{template.model_template_id}/label_schema.json"
    )


def pot_eval_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(ote_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(ote_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/pot_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/pot_{template.model_template_id}/performance.json",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/pot_{template.model_template_id}/performance.json"
    )


def nncf_optimize_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "optimize",
        template.model_template_path,
        "--train-ann-file",
        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
        "--save-model-to",
        f"{template_work_dir}/nncf_{template.model_template_id}",
        "--save-performance",
        f"{template_work_dir}/nncf_{template.model_template_id}/train_performance.json",
    ]
    command_line.extend(args["train_params"])
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth"
    )
    assert os.path.exists(
        f"{template_work_dir}/nncf_{template.model_template_id}/label_schema.json"
    )


def nncf_export_testing(template, root):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "export",
        template.model_template_path,
        "--load-weights",
        f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth",
        "--save-model-to",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml"
    )
    assert os.path.exists(
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.bin"
    )
    assert os.path.exists(
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/label_schema.json"
    )
    original_bin_size = os.path.getsize(
        f"{template_work_dir}/exported_{template.model_template_id}/openvino.bin"
    )
    compressed_bin_size = os.path.getsize(
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.bin"
    )
    assert (
        compressed_bin_size < original_bin_size
    ), f"{compressed_bin_size=}, {original_bin_size=}"


def nncf_eval_testing(template, root, ote_dir, args, threshold):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(ote_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(ote_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth",
        "--save-performance",
        f"{template_work_dir}/nncf_{template.model_template_id}/performance.json",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/nncf_{template.model_template_id}/performance.json"
    )
    with open(
        f"{template_work_dir}/nncf_{template.model_template_id}/train_performance.json"
    ) as read_file:
        trained_performance = json.load(read_file)
    with open(
        f"{template_work_dir}/nncf_{template.model_template_id}/performance.json"
    ) as read_file:
        evaluated_performance = json.load(read_file)

    for k in trained_performance.keys():
        assert (
            abs(trained_performance[k] - evaluated_performance[k])
            / (trained_performance[k] + 1e-10)
            <= threshold
        ), f"{trained_performance[k]=}, {evaluated_performance[k]=}"


def nncf_eval_openvino_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [
        "ote",
        "eval",
        template.model_template_path,
        "--test-ann-file",
        f'{os.path.join(ote_dir, args["--test-ann-files"])}',
        "--test-data-roots",
        f'{os.path.join(ote_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml",
        "--save-performance",
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/performance.json",
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(
        f"{template_work_dir}/exported_nncf_{template.model_template_id}/performance.json"
    )


def xfail_templates(templates, xfail_template_ids_reasons):
    xfailed_templates = []
    for template in templates:
        reasons = [
            reason
            for template_id, reason in xfail_template_ids_reasons
            if template_id == template.model_template_id
        ]
        if len(reasons) == 0:
            xfailed_templates.append(template)
        elif len(reasons) == 1:
            xfailed_templates.append(
                pytest.param(template, marks=pytest.mark.xfail(reason=reasons[0]))
            )
        else:
            raise RuntimeError(
                "More than one reason for template. If you have more than one Jira tickets, list them in one reason."
            )
    return xfailed_templates
