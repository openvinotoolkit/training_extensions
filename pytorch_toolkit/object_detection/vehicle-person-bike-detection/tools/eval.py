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
import hashlib
import json
import os
import subprocess
import tempfile
import yaml

from mmcv.utils import Config

MMDETECTION_TOOLS = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'
FACE_DETECTION_TOOLS = os.path.dirname(__file__)


def parse_args():
    """ Parses input args. """

    args = argparse.ArgumentParser()
    args.add_argument('config',
                      help='A path to model training configuration file (.py).')
    args.add_argument('snapshot',
                      help='A path to pre-trained snapshot (.pth).')
    args.add_argument('out',
                      help='A path to output file where models metrics will be saved (.yml).')

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


def collect_ap(path):
    """ Collects average precision values in log file. """

    average_precisions = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        for line in content:
            if line.startswith(beginning):
                average_precisions.append(float(line.replace(beginning, '')))
    return average_precisions


def sha256sum(filename):
    """ Computes sha256sum. """

    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def coco_ap_eval(config_path, work_dir, snapshot, res_pkl, outputs):
    """ Computes COCO AP. """

    with open(os.path.join(work_dir, 'test_py_stdout'), 'w') as test_py_stdout:
        subprocess.run(
            f'python {MMDETECTION_TOOLS}/test.py'
            f' {config_path} {snapshot}'
            f' --out {res_pkl} --eval bbox'.split(' '), stdout=test_py_stdout, check=True)
    average_precision = collect_ap(os.path.join(work_dir, 'test_py_stdout'))[0]
    outputs.append({'key': 'ap', 'value': average_precision * 100, 'unit': '%', 'display_name': 'AP @ [IoU=0.50:0.95]'})
    return outputs


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


def eval(config_path, snapshot, out):
    """ Main evaluation procedure. """

    cfg = Config.fromfile(config_path)

    work_dir = tempfile.mkdtemp()
    print('results are stored in:', work_dir)

    if os.path.islink(snapshot):
        snapshot = os.path.join(os.path.dirname(snapshot), os.readlink(snapshot))

    files = get_file_size_and_sha256(snapshot)

    metrics = []
    res_pkl = os.path.join(work_dir, "res.pkl")
    metrics = coco_ap_eval(config_path, work_dir, snapshot, res_pkl, metrics)
    metrics = get_complexity_and_size(cfg, config_path, work_dir, metrics)

    for metric in metrics:
        metric['value'] = round(metric['value'], 3)

    outputs = {
        'files': [files],
        'metrics': metrics
    }

    if os.path.exists(out):
        with open(out) as read_file:
            content = yaml.load(read_file)
        content.update(outputs)
        outputs = content

    with open(out, 'w') as write_file:
        yaml.dump(outputs, write_file)


def main():
    """ Main function. """

    args = parse_args()
    eval(args.config, args.snapshot, args.out)


if __name__ == '__main__':
    main()
