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

# pylint: disable=C0301,W0622,R0914

import argparse
import json
import subprocess
import os
import tempfile
import yaml

from mmcv.utils import Config

from eval import eval


def parse_args():
    """ Parses input args. """

    args = argparse.ArgumentParser()
    args.add_argument('config',
                      help='A path to model training configuration file (.py).')
    args.add_argument('gpu_num',
                      help='A number of GPU to use in training.')
    args.add_argument('out',
                      help='A path to output file where models metrics will be saved (.yml).')
    args.add_argument(
        '--update_config',
        help='Update configuration file by parameters specified here.'
             'Use quotes if you are going to change several params.',
        default='')

    return args.parse_args()


def is_clustering_needed(cfg, update_config):
    # resume_from = [p.split('=') for p in update_config.strip().split(' ') if p.startswith('resume_from=')]
    # resume_from = resume_from[0][1] if resume_from else str(cfg.resume_from)
    # if resume_from.lower() not in ['', 'none']:
    #    return False
    if not hasattr(cfg.model, 'bbox_head') or not cfg.model.bbox_head.type == 'SSDHead':
        return False
    if not cfg.model.bbox_head.anchor_generator.type == 'SSDAnchorGeneratorClustered':
        return False
    return True


def main():
    """ Main function. """

    args = parse_args()

    mmdetection_tools = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'

    cfg = Config.fromfile(args.config)

    update_config = f' --update_config {args.update_config}' if args.update_config else ''

    if is_clustering_needed(cfg, update_config):
        widths = cfg.model.bbox_head.anchor_generator.widths
        n_clust = 0
        for w in widths:
            n_clust += len(w) if isinstance(w, (list, tuple)) else 1
        n_clust = ' --n_clust ' + str(n_clust)

        group_as = ''
        if isinstance(widths[0], (list, tuple)):
            group_as = ' --group_as ' + ' '.join([str(len(w)) for w in widths])

        config = ' --config ' + args.config

        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        out = f' --out {tmp_file.name}'

        if 'pipeline' in cfg.data.train:
            img_shape = [t for t in cfg.data.train.pipeline if t['type'] == 'Resize'][0]['img_scale']
        else:
            img_shape = [t for t in cfg.data.train.dataset.pipeline if t['type'] == 'Resize'][0]['img_scale']

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

    subprocess.run(f'{mmdetection_tools}/dist_train.sh'
                   f' {args.config}'
                   f' {args.gpu_num}'
                   f'{update_config}'.split(' '), check=True)

    overrided_work_dir = [p.split('=') for p in args.update_config.strip().split(' ') if p.startswith('work_dir=')]
    if overrided_work_dir:
        cfg.work_dir = overrided_work_dir[0][1]

    eval(args.config, os.path.join(cfg.work_dir, "latest.pth"), args.out, args.update_config)

    with open(args.out, 'r+') as dst_file:
        content = yaml.load(dst_file, Loader=yaml.FullLoader)
        content['training_gpu_num'] = args.gpu_num
        yaml.dump(content, dst_file)


if __name__ == '__main__':
    main()
