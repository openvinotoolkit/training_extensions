# Copyright (C) 2020 Intel Corporation
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

# pylint: disable=W1203,C0411,C0413,no-value-for-parameter

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile

from mmcv.utils import Config
import yaml

from eval import main as evaluate

sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}/../../')
from tools.misc import train, get_work_dir


def parse_args():
    """ Parses input args. """

    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='A path to model training configuration file (.py).')
    parser.add_argument('gpu_num', type=int,
                        help='A number of GPUs to use in training.')
    parser.add_argument('out',
                        help='A path to output file where models metrics will be saved (.yml).')
    parser.add_argument('--update_config',
                        help='Update configuration file by parameters specified here.'
                             'Use quotes if you are going to change several params.',
                        default='')
    parser.add_argument('--show-dir', '--show_dir', dest='show_dir',
                        help='A directory where images with drawn detected objects will be saved.')

    return parser.parse_args()


def is_clustering_needed(cfg):
    if cfg.total_epochs > 0:
        return False
    if not hasattr(cfg.model, 'bbox_head') or not cfg.model.bbox_head.type == 'SSDHead':
        return False
    if not cfg.model.bbox_head.anchor_generator.type == 'SSDAnchorGeneratorClustered':
        return False
    return True


def cluster(cfg, config_path, update_config):
    mmdetection_tools = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'
    logging.info('Clustering started...')
    widths = cfg.model.bbox_head.anchor_generator.widths
    n_clust = 0
    for w in widths:
        n_clust += len(w) if isinstance(w, (list, tuple)) else 1
    n_clust = ' --n_clust ' + str(n_clust)

    group_as = ''
    if isinstance(widths[0], (list, tuple)):
        group_as = ' --group_as ' + ' '.join([str(len(w)) for w in widths])

    config = ' --config ' + config_path

    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    out = f' --out {tmp_file.name}'

    if 'pipeline' in cfg.data.train:
        img_shape = [t for t in cfg.data.train.pipeline if t['type'] == 'Resize'][0][
            'img_scale']
    else:
        img_shape = [t for t in cfg.data.train.dataset.pipeline if t['type'] == 'Resize'][0][
            'img_scale']

    img_shape = f' --image_size_wh {img_shape[0]} {img_shape[1]}'

    subprocess.run(f'python {mmdetection_tools}/cluster_boxes.py'
                   f'{config}'
                   f'{n_clust}'
                   f'{group_as}'
                   f'{update_config}'
                   f'{img_shape}'
                   f'{out}'.split(' '), check=True)

    with open(tmp_file.name) as src_file:
        content = json.load(src_file)
        widths, heights = content['widths'], content['heights']

    if not update_config:
        update_config = ' --update_config'
    update_config += f' model.bbox_head.anchor_generator.widths={str(widths).replace(" ", "")}'
    update_config += f' model.bbox_head.anchor_generator.heights={str(heights).replace(" ", "")}'
    logging.info('... clustering completed.')

    return update_config


def main():
    """ Main function. """
    logging.basicConfig(level=logging.DEBUG)

    args = parse_args()
    logging.info(f'Commandline:\n{" ".join(sys.argv)}')

    cfg = Config.fromfile(args.config)

    update_config = f' --update_config {args.update_config}' if args.update_config else ''

    if is_clustering_needed(cfg):
        update_config = cluster(cfg, args.config, update_config)

    logging.info('Training started ...')
    training_info = train(args.config, args.gpu_num, update_config)
    logging.info('... training completed.')
    work_dir = get_work_dir(cfg, args.update_config)

    logging.info('Evaluation started ...')
    evaluate(os.path.join(work_dir, "config.py"), os.path.join(work_dir, "latest.pth"), args.out, '', args.show_dir)
    logging.info('... evaluation completed.')

    with open(args.out, 'a+') as dst_file:
        yaml.dump(training_info, dst_file)


if __name__ == '__main__':
    main()
