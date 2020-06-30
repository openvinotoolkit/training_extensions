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

# pylint: disable=C0301,W0622,R0914,R0913

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import sys
import yaml

from mmcv.utils import Config

MMDETECTION_TOOLS = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'


def parse_args():
    """ Parses input args. """

    args = argparse.ArgumentParser()
    args.add_argument('config',
                      help='A path to model training configuration file (.py).')
    args.add_argument('snapshot',
                      help='A path to pre-trained snapshot (.pth).')
    args.add_argument('out',
                      help='A path to output file where models metrics will be saved (.yml).')
    args.add_argument('--update_config',
                      help='Update configuration file by parameters specified here.'
                           'Use quotes if you are going to change several params.',
                      default='')

    return args.parse_args()


def replace_text_in_file(path, replace_what, replace_by):
    """ Replaces text in file. """

    with open(path) as read_file:
        content = '\n'.join([line.rstrip() for line in read_file.readlines()])
        if content.find(replace_what) == -1:
            return False
        content = content.replace(replace_what, replace_by)
    with open(path, 'w') as write_file:
        write_file.write(content)
    return True


def collect_f1(path):
    metric = 'hmean'
    content = []
    hmean = []
    with open(path) as read_file:
        content += [line.split() for line in read_file.readlines()]
        for line in content:
            if (len(line) > 0) and (line[0] == 'Text'):
                for word in line[2:]:
                    if word.startswith(metric):
                        hmean.append(float(word.replace(metric+'=', '')))
    return hmean


def sha256sum(filename):
    """ Computes sha256sum. """

    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def coco_f1_eval(config_path, work_dir, snapshot, res_pkl, outputs, update_config):
    """ Computes COCO F1. """

    with open(os.path.join(work_dir, 'test_py_stdout'), 'w') as test_py_stdout:
        update_config = f' --update_config {update_config}' if update_config else ''
        subprocess.run(
            f'python {MMDETECTION_TOOLS}/test.py'
            f' {config_path} {snapshot}'
            f' --out {res_pkl} --eval f1{update_config}'.split(' '), stdout=test_py_stdout, check=True)
    hmean = collect_f1(os.path.join(work_dir, 'test_py_stdout'))[0]
    outputs.append({'key': 'f1', 'value': hmean * 100, 'unit': '%', 'display_name': 'Harmonic mean'})
    return outputs


# def custom_ap_eval(config_path, work_dir, res_pkl, outputs, update_config):
#     """ Computes AP on faces that are greater than 64x64. """
#
#     res_custom_metrics = os.path.join(work_dir, "custom_metrics.json")
#     update_config = f'--update_config {update_config}' if update_config else ''
#     subprocess.run(
#         f'python {FACE_DETECTION_TOOLS}/wider_custom_eval.py'
#         f' {config_path} {res_pkl} --out {res_custom_metrics} {update_config}'.split(' '), check=True)
#     with open(res_custom_metrics) as read_file:
#         ap_64x64 = [x['average_precision'] for x in json.load(read_file) if x['object_size'][0] == 64][0]
#         outputs.append({'key': 'ap_64x64', 'value': ap_64x64, 'display_name': 'AP for faces > 64x64', 'unit': '%'})
#     return outputs


def get_complexity_and_size(cfg, config_path, work_dir, outputs):
    """ Gets complexity and size of a model. """

    image_shape = [x['img_scale'] for x in cfg.test_pipeline if 'img_scale' in x][0][::-1]
    image_shape = " ".join([str(x) for x in image_shape])

    res_complexity = os.path.join(work_dir, "complexity.json")

    subprocess.run(
        f'python {MMDETECTION_TOOLS}/get_flops.py'
        f' {config_path}'
        f' --shape {image_shape}'
        f' --out {res_complexity}'.split(' '), check=True)
    with open(res_complexity) as read_file:
        content = json.load(read_file)
        outputs.extend(content)
    return outputs


def get_file_size_and_sha256(snapshot):
    """ Gets size and sha256 of a file. """

    return {
        'sha256': sha256sum(snapshot),
        'size': os.path.getsize(snapshot),
        'name': os.path.basename(snapshot),
        'source': snapshot
    }


def eval(config_path, snapshot, out, update_config):
    """ Main evaluation procedure. """

    cfg = Config.fromfile(config_path)

    work_dir = tempfile.mkdtemp()
    print('results are stored in:', work_dir)

    if os.path.islink(snapshot):
        snapshot = os.path.join(os.path.dirname(snapshot), os.readlink(snapshot))

    files = get_file_size_and_sha256(snapshot)

    metrics = []

    metrics = get_complexity_and_size(cfg, config_path, work_dir, metrics)
    res_pkl = os.path.join(work_dir, "res.pkl")
    metrics = coco_f1_eval(config_path, work_dir, snapshot, res_pkl, metrics, update_config)

    for metric in metrics:
        metric['value'] = round(metric['value'], 3)

    outputs = {
        'files': [files],
        'metrics': metrics
    }

    if os.path.exists(out):
        with open(out) as read_file:
            content = yaml.load(read_file, Loader=yaml.FullLoader)
        content.update(outputs)
        outputs = content

    with open(out, 'w') as write_file:
        yaml.dump(outputs, write_file)


def main():
    """ Main function. """

    args = parse_args()
    eval(args.config, args.snapshot, args.out, args.update_config)


if __name__ == '__main__':
    main()
