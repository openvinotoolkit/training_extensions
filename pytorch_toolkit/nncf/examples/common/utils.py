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
import json
import logging
import pdb
import sys
from os import path as osp
from pathlib import Path

import os
import tarfile
from shutil import copyfile
from tensorboardX import SummaryWriter
from texttable import Texttable

from examples.common.example_logger import logger as default_logger

GENERAL_LOG_FILE_NAME = "output.log"
NNCF_LOG_FILE_NAME = "nncf_output.log"


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
            initializer = algo_dict.get("initializer", {})
            precision = initializer.get("precision", {})
            if precision:
                retval += "_mixed_int"
            else:
                activations = algo_dict.get('activations', {})
                a_bits = activations.get('bits', 8)
                weights = algo_dict.get('weights', {})
                w_bits = weights.get('bits', 8)
                if a_bits == w_bits:
                    retval += "_int{}".format(a_bits)
                else:
                    retval += "_a_int{}_w_int{}".format(a_bits, w_bits)
        else:
            retval += "_{}".format(algo_name)
    return retval


def write_metrics(acc, filename):
    avg = round(acc * 100, 2)
    metrics = {"Accuracy": avg}
    if os.path.isfile(filename):
        path = Path(filename)
        data = json.loads(path.read_text(encoding='utf-8'))
        data.update(metrics)
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    else:
        with open(filename, 'w') as outfile:
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


def configure_logging(sample_logger, config):
    config.tb = SummaryWriter(config.log_dir)

    training_pipeline_log_file_handler = logging.FileHandler(osp.join(config.log_dir, GENERAL_LOG_FILE_NAME))
    training_pipeline_log_file_handler.setFormatter(logging.Formatter("%(message)s"))
    sample_logger.addHandler(training_pipeline_log_file_handler)

    nncf_log_file_handler = logging.FileHandler(osp.join(config.log_dir, NNCF_LOG_FILE_NAME))
    nncf_log_file_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    from nncf.nncf_logger import logger as nncf_logger
    nncf_logger.addHandler(nncf_log_file_handler)


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


def print_args(config, logger=default_logger):
    for arg in sorted(config):
        logger.info("{: <27s}: {}".format(arg, config.get(arg)))


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


def print_statistics(stats, logger=default_logger):
    for key, val in stats.items():
        if isinstance(val, Texttable):
            logger.info(key)
            logger.info(val.draw())
        else:
            logger.info("{}: {}".format(key, val))
