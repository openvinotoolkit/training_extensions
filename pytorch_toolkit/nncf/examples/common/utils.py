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

import datetime
import os
import pdb
import sys
import tarfile
import json
from os import path as osp
from pathlib import Path
from shutil import copyfile

from tensorboardX import SummaryWriter


def get_name(config):
    dataset = config.get('dataset', 'imagenet')
    if dataset is None:
        dataset = 'imagenet'
    retval = config["model"] + "_" + dataset
    compression_config = config.get('compression', [])
    if not isinstance(compression_config, list):
        compression_config = [compression_config, ]
    for algo_dict in compression_config:
        algo_name = algo_dict["algorithm"]
        if algo_name == "quantization":
            params_dict = algo_dict.get("params", {})
            activations = params_dict.get('activations', {})
            bits = activations.get('bits', 8)
            retval += "_int{}".format(bits)
        else:
            retval += "_{}".format(algo_name)
    return retval


def write_metrics(config, metrics):
    if os.path.isfile(config.metrics_dump):
        print("File ", config.metrics_dump, " is found")
        path = Path(config.metrics_dump)
        data = json.loads(path.read_text(encoding='utf-8'))
        data.update(metrics)
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    else:
        print("Creating ", config.metrics_dump, " file")
        with open(config.metrics_dump, 'w') as outfile:
            json.dump(metrics, outfile)


def configure_paths(config):
    d = datetime.datetime.now()
    run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
    config.name = get_name(config)
    config.log_dir = osp.join(config.log_dir, "{}/{}".format(config.name, run_id))
    os.makedirs(config.log_dir)

    if config.checkpoint_save_dir is None:
        config.checkpoint_save_dir = config.log_dir

    # create aux dirs
    config.intermediate_checkpoints_path = config.log_dir + '/intermediate_checkpoints'
    os.makedirs(config.intermediate_checkpoints_path)
    os.makedirs(config.checkpoint_save_dir, exist_ok=True)


def configure_logging(config):
    config.tee = TeedStream(osp.join(config.log_dir, 'output.log'))
    config.tb = SummaryWriter(config.log_dir)


def is_on_first_rank(config):
    return not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                      and config.rank % config.ngpus_per_node == 0)


def create_code_snapshot(root, dst_path, extensions=(".py", ".json", ".cpp", ".cu")):
    """Creates tarball with the source code"""
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


def print_args(config):
    for arg in sorted(config):
        print("{: <27s}: {}".format(arg, config.get(arg)))


def make_link(src, dst, exists_ok=True):
    if osp.exists(dst) and exists_ok:
        os.remove(dst)
    dev1 = os.stat(osp.dirname(dst)).st_dev
    dev2 = os.stat(src).st_dev
    if dev1 != dev2:
        copyfile(src, dst)
    else:
        os.link(src, dst)


def make_additional_checkpoints(checkpoint_path, is_best, epoch, config):
    if is_best:
        best_path = osp.join(config.checkpoint_save_dir, '{}_best.pth'.format(config.name))
        copyfile(checkpoint_path, best_path)
    if epoch % config.save_freq == 0:
        intermediate_checkpoint = osp.join(config.intermediate_checkpoints_path,
                                           'epoch_{}.pth'.format(epoch))
        copyfile(checkpoint_path, intermediate_checkpoint)


class TeedStream:
    """Copy stdout to the file"""

    def __init__(self, fname, mode='w'):
        self.file = open(str(fname), mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


# pylint:disable=no-member
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def is_binarization(config):
    compression_config = config.get('compression', {})
    if isinstance(compression_config, list):
        compression_config = compression_config[0]
    algo_type = compression_config.get("algorithm")
    if algo_type is not None and algo_type == "binarization":
        return True
    return False
