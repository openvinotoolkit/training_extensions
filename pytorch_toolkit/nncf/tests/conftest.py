"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from pathlib import Path

import pytest
import torch

TEST_ROOT = Path(__file__).parent.absolute()
PROJECT_ROOT = TEST_ROOT.parent
EXAMPLES_DIR = PROJECT_ROOT / 'examples'


@pytest.fixture(scope="function", autouse=True)
def empty_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_addoption(parser):
    parser.addoption(
        "--data", type=str, default=None,
        help="Path to test datasets, e.g. CIFAR10 - for sanity tests or CIFAR100 - for weekly ones"
    )
    parser.addoption(
        "--torch-home", type=str, default=None, help="Path to cached test models, downloaded by torchvision"
    )
    parser.addoption(
        "--weekly-models", type=str, default=None, help="Path to models' weights for weekly tests"
    )


@pytest.fixture(scope="module")
def dataset_dir(request):
    return request.config.getoption("--data")


@pytest.fixture(scope="module")
def weekly_models_path(request):
    return request.config.getoption("--weekly-models")


@pytest.fixture(autouse=True)
def torch_home_dir(request, monkeypatch):
    torch_home = request.config.getoption("--torch-home")
    if torch_home:
        monkeypatch.setenv('TORCH_HOME', torch_home)
