"""Tests for anomaly classification with OTE CLI"""

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
import os
from subprocess import run
from pathlib import Path
import pytest
from ote_cli.registry import Registry
from tests.ote_cli.common import collect_env_vars, get_some_vars, create_venv

args = {
        '--train-ann-file': 'data/anomaly/train.json',
        '--train-data-roots': 'data/anomaly/shapes',
        '--val-ann-file': 'data/anomaly/val.json',
        '--val-data-roots': 'data/anomaly/shapes',
        '--test-ann-files': 'data/anomaly/test.json',
        '--test-data-roots': 'data/anomaly/shapes',
}

root: Path = Path("/tmp/ote_cli/")
ote_dir = os.getcwd()

templates = Registry('external').filter(task_type='ANOMALY_CLASSIFICATION').templates
templates_ids = [template.model_template_id for template in templates]


@pytest.mark.parametrize("template", templates, ids=templates_ids)
def test_ote_train(template):
    work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
    os.makedirs(template_work_dir, exist_ok=True)
    create_venv(algo_backend_dir, work_dir, template_work_dir)
    command_line = [
        "ote",
        "train",
        template.model_template_id,
        f"--train-ann-files={os.path.join(ote_dir, args['--train-ann-file'])}",
        f"--train-data-roots={os.path.join(ote_dir, args['--train-data-roots'])}",
        f"--val-ann-files={os.path.join(ote_dir, args['--val-ann-file'])}",
        f"--val-data-roots={os.path.join(ote_dir, args['--val-data-roots'])}",
        f"--save-model-to={template_work_dir}/trained.pth"
    ]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
