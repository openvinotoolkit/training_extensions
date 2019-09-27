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

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time

import pytest
import torch

# pylint: disable=redefined-outer-name
from examples.common.optimizer import get_default_weight_decay
from examples.common.utils import get_name, is_binarization
from nncf.config import Config
from nncf.dynamic_graph import reset_context
from tests.conftest import EXAMPLES_DIR, PROJECT_ROOT, TEST_ROOT


class Command:
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.exec_time = -1
        self.output = []  # store output here
        self.kwargs = {}
        self.timeout = False

        # set system/version dependent "start_new_session" analogs
        if sys.platform == "win32":
            self.kwargs.update(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        elif sys.version_info < (3, 2):  # assume posix
            self.kwargs.update(preexec_fn=os.setsid)
        else:  # Python 3.2+ and Unix
            self.kwargs.update(start_new_session=True)

    def kill_process_tree(self, pid):
        try:
            if sys.platform != "win32":
                os.killpg(pid, signal.SIGKILL)
            else:
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)])
        except OSError as err:
            print(err)

    def run(self, timeout=3600):

        def target():
            start_time = time.time()
            self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
                                            bufsize=1, **self.kwargs)
            self.timeout = False

            self.output = []
            for line in self.process.stdout:
                line = line.decode('utf-8')
                self.output.append(line)
                sys.stdout.write(line)

            sys.stdout.flush()
            self.process.stdout.close()

            self.process.wait()
            self.exec_time = time.time() - start_time

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            try:
                print("Error: process taking too long to complete--terminating" + ", [ " + self.cmd + " ]")
                self.kill_process_tree(self.process.pid)
                self.exec_time = timeout
                self.timeout = True
                thread.join()
            except OSError as e:
                print(self.process.pid, "Exception when try to kill task by PID, " + e.strerror)
                raise
        returncode = self.process.wait()
        print("Process returncode = " + str(returncode))
        return returncode

    def get_execution_time(self):
        return self.exec_time


class ConfigFactory:
    """Allows to modify config file before test run"""

    def __init__(self, base_config, config_path):
        self.config = base_config
        self.config_path = str(config_path)

    def serialize(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        return self.config_path

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value


def create_command_line(args, sample_type):
    python_path = PROJECT_ROOT.as_posix()
    executable = EXAMPLES_DIR.joinpath(sample_type, 'main.py').as_posix()
    cli_args = " ".join(key if val is None else "{} {}".format(key, val) for key, val in args.items())
    return "PYTHONPATH={path} {python_exe} {main_py} {args}".format(
        path=python_path, main_py=executable, args=cli_args, python_exe=sys.executable
    )


SAMPLE_TYPES = ["classification", "segmentation"]

DATASETS = {
    "classification": ["cifar10", "cifar100"],
    "segmentation": ["camvid"],
}

CONFIGS = {
    "classification": [TEST_ROOT.joinpath("data", "configs", "squeezenet1_1_cifar10_sparsity_int8.json"),
                       TEST_ROOT.joinpath("data", "configs", "resnet18_cifar100_bin_xnor.json")],
    "segmentation": [TEST_ROOT.joinpath("data", "configs", "unet_camvid_int8.json")]
}

BATCHSIZE_PER_GPU = {
    "classification": [256, 256],
    "segmentation": [2],
}

DATASET_PATHS = {
    "classification": {
        x: lambda dataset_root: dataset_root if dataset_root else os.path.join(
            tempfile.gettempdir(), x) for x in DATASETS["classification"]
    },
    "segmentation":
        {
            DATASETS["segmentation"][0]: lambda dataset_root: TEST_ROOT.joinpath("data", "mock_datasets", "camvid")
        }
}

CONFIG_PARAMS = list()
for sample_type in SAMPLE_TYPES:
    for tpl in list(zip(CONFIGS[sample_type], DATASETS[sample_type], BATCHSIZE_PER_GPU[sample_type])):
        CONFIG_PARAMS.append((sample_type,) + tpl)


@pytest.fixture(scope="module", params=CONFIG_PARAMS,
                ids=["-".join([p[0], p[1].name, p[2], str(p[3])]) for p in CONFIG_PARAMS])
def config(request, tmp_path_factory, dataset_dir):
    sample_type, config_path, dataset_name, batch_size = request.param
    dataset_path = DATASET_PATHS[sample_type][dataset_name](dataset_dir)

    with config_path.open() as f:
        jconfig = json.load(f)

    if "checkpoint_save_dir" in jconfig.keys():
        del jconfig["checkpoint_save_dir"]

    jconfig["dataset"] = dataset_name

    checkpoint_save_dir = str(tmp_path_factory.mktemp("models"))

    return {
        "sample_type": sample_type,
        'config': jconfig,
        "model_name": jconfig["model"],
        "dataset_path": dataset_path,
        "checkpoint_save_dir": checkpoint_save_dir,
        "batch_size": batch_size,
        "results_dir": str(tmp_path_factory.mktemp("results_dir"))
    }


def test_pretrained_model_export(config, tmp_path):
    c = config
    config_factory = ConfigFactory(c['config'], tmp_path / 'config.json')

    if isinstance(config_factory['compression'], list):
        config_factory['compression'][0]['initializer'] = {'num_init_steps': 0}
    else:
        config_factory['compression']['initializer'] = {'num_init_steps': 0}

    onnx_path = os.path.join(str(tmp_path), "model.onnx")
    args = {
        "--mode": "test",
        "--config": config_factory.serialize(),
        "--to-onnx": onnx_path
    }

    runner = Command(create_command_line(args, c["sample_type"]))
    res = runner.run()
    assert res == 0
    assert os.path.exists(onnx_path)


@pytest.mark.parametrize(" multiprocessing_distributed",
                         (True, False),
                         ids=['distributed', 'dataparallel'])
def test_pretrained_model_eval(config, tmp_path, multiprocessing_distributed):
    c = config

    config_factory = ConfigFactory(c['config'], tmp_path / 'config.json')
    args = {
        "--mode": "test",
        "--data": c["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": c["batch_size"] * torch.cuda.device_count(),
        "--workers": 1,
    }

    if multiprocessing_distributed:
        args["--multiprocessing-distributed"] = None

    runner = Command(create_command_line(args, c["sample_type"]))
    res = runner.run()
    assert res == 0


@pytest.mark.parametrize(
    "multiprocessing_distributed", [
        pytest.param(True, marks=pytest.mark.dependency(name=["train_distributed"])),
        pytest.param(False, marks=pytest.mark.dependency(name=["train_dataparallel"]))],
    ids=['distributed', 'dataparallel'])
def test_pretrained_model_train(config, tmp_path, multiprocessing_distributed):
    c = config

    checkpoint_save_dir = os.path.join(c["checkpoint_save_dir"],
                                       "distributed" if multiprocessing_distributed else "data_parallel")
    config_factory = ConfigFactory(config['config'], tmp_path / 'config.json')
    args = {
        "--mode": "train",
        "--data": c["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": c["batch_size"] * torch.cuda.device_count(),
        "--workers": 1,
        "--epochs": 1,
        "--checkpoint-save-dir": checkpoint_save_dir,
        "--dist-url": "tcp://127.0.0.1:8989"
    }

    if multiprocessing_distributed:
        args["--multiprocessing-distributed"] = None

    runner = Command(create_command_line(args, c["sample_type"]))
    res = runner.run()
    assert res == 0
    assert os.path.exists(os.path.join(checkpoint_save_dir, get_name(config_factory.config) + "_last.pth"))


@pytest.mark.parametrize(
    "multiprocessing_distributed", [
        pytest.param(True, marks=pytest.mark.dependency(depends=["train_distributed"])),
        pytest.param(False, marks=pytest.mark.dependency(depends=["train_dataparallel"]))],
    ids=['distributed', 'dataparallel'])
def test_trained_model_export(config, tmp_path, multiprocessing_distributed):
    c = config

    config_factory = ConfigFactory(config['config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(c["checkpoint_save_dir"],
                             "distributed" if multiprocessing_distributed else "data_parallel",
                             get_name(config_factory.config) + "_last.pth")
    onnx_path = os.path.join(str(tmp_path), "model.onnx")
    args = {
        "--mode": "test",
        "--config": config_factory.serialize(),
        "--to-onnx": onnx_path,
        "--weights": ckpt_path
    }

    runner = Command(create_command_line(args, c["sample_type"]))
    res = runner.run()
    assert res == 0
    assert os.path.exists(onnx_path)


@pytest.mark.parametrize(
    "multiprocessing_distributed", [
        pytest.param(True, marks=pytest.mark.dependency(depends=["train_distributed"])),
        pytest.param(False, marks=pytest.mark.dependency(depends=["train_dataparallel"]))],
    ids=['distributed', 'dataparallel'])
def test_trained_model_eval(config, tmp_path, multiprocessing_distributed):
    c = config

    config_factory = ConfigFactory(config['config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(c["checkpoint_save_dir"],
                             "distributed" if multiprocessing_distributed else "data_parallel",
                             get_name(config_factory.config) + "_last.pth")
    args = {
        "--mode": "test",
        "--data": c["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": c["batch_size"] * torch.cuda.device_count(),
        "--workers": 1,
        "--weights": ckpt_path,
    }

    if multiprocessing_distributed:
        args["--multiprocessing-distributed"] = None

    runner = Command(create_command_line(args, c["sample_type"]))
    res = runner.run()
    assert res == 0


@pytest.mark.parametrize(
    "multiprocessing_distributed", [
        pytest.param(True, marks=pytest.mark.dependency(depends=["train_distributed"])),
        pytest.param(False, marks=pytest.mark.dependency(depends=["train_dataparallel"]))],
    ids=['distributed', 'dataparallel'])
def test_resume(config, tmp_path, multiprocessing_distributed):
    c = config

    checkpoint_save_dir = os.path.join(str(tmp_path), "models")
    config_factory = ConfigFactory(config['config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(c["checkpoint_save_dir"],
                             "distributed" if multiprocessing_distributed else "data_parallel",
                             get_name(config_factory.config) + "_last.pth")
    args = {
        "--mode": "train",
        "--data": c["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": c["batch_size"] * torch.cuda.device_count(),
        "--workers": 1,
        "--epochs": 2,
        "--checkpoint-save-dir": checkpoint_save_dir,
        "--resume": ckpt_path,
    }

    if multiprocessing_distributed:
        args["--multiprocessing-distributed"] = None

    runner = Command(create_command_line(args, c["sample_type"]))
    res = runner.run()
    assert res == 0
    assert os.path.exists(os.path.join(checkpoint_save_dir, get_name(config_factory.config) + "_last.pth"))


@pytest.mark.parametrize(('algo', 'ref_weight_decay'),
                         (('rb_sparsity', 0),
                          ('const_sparsity', 1e-4),
                          ('magnitude_sparsity', 1e-4),
                          ('quantization', 1e-4)))
def test_get_default_weight_decay(algo, ref_weight_decay):
    config = Config()
    config.update({"compression": {"algorithm": algo}})
    assert ref_weight_decay == get_default_weight_decay(config)


def test_cpu_only_mode_produces_cpu_only_model(config, tmp_path, mocker):
    reset_context('orig')
    reset_context('quantized_graphs')
    c = config

    config_factory = ConfigFactory(config['config'], tmp_path / 'config.json')
    args = {
        "--data": c["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": c["batch_size"] * torch.cuda.device_count(),
        "--workers": 1,
        "--epochs": 1,
        "--cpu-only": None
    }

    command_line = " ".join(key if val is None else "{} {}".format(key, val) for key, val in args.items())

    if config["sample_type"] == "classification":
        import examples.classification.main as sample
        if is_binarization(config['config']):
            mocker.patch("examples.classification.binarization_worker.train_epoch_bin")
            mocker.patch("examples.classification.binarization_worker.validate")
            import examples.classification.binarization_worker as bin_worker
            bin_worker.validate.return_value = (0, 0)
        else:
            mocker.patch("examples.classification.main.train_epoch")
            mocker.patch("examples.classification.main.validate")
            sample.validate.return_value = (0, 0)
    elif config["sample_type"] == "segmentation":
        import examples.segmentation.main as sample
        import examples.segmentation.train
        mocker.spy(examples.segmentation.train.Train, "__init__")

    sample.main(shlex.split(command_line))

    # pylint: disable=no-member
    if config["sample_type"] == "classification":
        if is_binarization(config['config']):
            import examples.classification.binarization_worker as bin_worker
            model_to_be_trained = bin_worker.train_epoch_bin.call_args[0][2]  # model
        else:
            model_to_be_trained = sample.train_epoch.call_args[0][1]  # model
    elif config["sample_type"] == "segmentation":
        model_to_be_trained = examples.segmentation.train.Train.__init__.call_args[0][1]  # model
    for p in model_to_be_trained.parameters():
        assert not p.is_cuda
