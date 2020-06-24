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
import sys

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
    args.add_argument('--wider_dir',
                      help='Specify this  path if you would like to test your model on WiderFace dataset.',
                      default='data/wider_face')

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


def compute_wider_metrics(config_path, snapshot, work_dir, wider_dir, outputs):
    """ Computes WiderFace metrics on easy, medium, hard subsets. """

    wider_data_folder = wider_dir
    os.makedirs(wider_data_folder, exist_ok=True)

    wider_data_zip = os.path.join(wider_data_folder, 'WIDER_val.zip')
    if not os.path.exists(wider_data_zip):
        print('', file=sys.stderr)
        print('#########################################################################', file=sys.stderr)
        print('Cannot compute WiderFace metrics, failed to find WIDER_val.zip here:', file=sys.stderr)
        print(f'    {os.path.abspath(wider_data_zip)}', file=sys.stderr)
        print('Please download the data from', file=sys.stderr)
        print('    https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view', file=sys.stderr)
        print('Save downloaded data as:', file=sys.stderr)
        print(f'    {os.path.abspath(wider_data_zip)}', file=sys.stderr)
        print(f'#########################################################################', file=sys.stderr)

        return outputs

    subprocess.run(f'unzip -q -o {wider_data_zip} -d {wider_data_folder}'.split(' '), check=True)

    eval_tools_zip = os.path.join(wider_data_folder, 'eval_tools.zip')
    if not os.path.exists(eval_tools_zip):
        subprocess.run(
            f'wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip'
            f' -O {eval_tools_zip}'.split(' '), check=True)
    subprocess.run(f'unzip -q -o {eval_tools_zip} -d {wider_data_folder}'.split(' '), check=True)

    wider_annotation_zip = os.path.join(wider_data_folder, 'ider_face_split.zip')
    if not os.path.exists(wider_annotation_zip):
        subprocess.run(
            f'wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
            f' -O {wider_annotation_zip}'.split(' '), check=True)
    subprocess.run(f'unzip -q -o {wider_annotation_zip} -d {wider_data_folder}'.split(' '), check=True)

    wider_annotation = os.path.join(wider_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    wider_images = os.path.join(wider_dir, 'WIDER_val', 'images')
    wider_coco_annotation = os.path.join(wider_dir, 'instances_val.json')
    subprocess.run(
        f'python {FACE_DETECTION_TOOLS}/wider_to_coco.py'
        f' {wider_annotation} {wider_images} {wider_coco_annotation}'.split(' '), check=True)

    res_pkl = os.path.join(work_dir, 'wider_face_res.pkl')

    with open(os.path.join(work_dir, 'test_py_on_wider_stdout_'), 'w') as test_py_stdout:
        subprocess.run(
            f'python {MMDETECTION_TOOLS}/test.py'
            f' {config_path} {snapshot}'
            f' --out {res_pkl}'
            f' --update_config data.test.ann_file={wider_coco_annotation} data.test.img_prefix={wider_dir}'.split(' '),
            stdout=test_py_stdout, check=True)

    wider_face_predictions = tempfile.mkdtemp()
    subprocess.run(
        f'python {FACE_DETECTION_TOOLS}/test_out_to_wider_predictions.py'
        f' {config_path} {res_pkl} {wider_face_predictions}'
        f' --update_config data.test.ann_file={wider_coco_annotation} data.test.img_prefix={wider_dir}'.split(' '),
        check=True)

    res_wider_metrics = os.path.join(work_dir, "wider_metrics.json")
    subprocess.run(
        f'python {FACE_DETECTION_TOOLS}/wider_face_eval.py'
        f' -g {wider_data_folder}/eval_tools/ground_truth/'
        f' -p {wider_face_predictions}'
        f' --out {res_wider_metrics}'.split(' '), check=True)
    with open(res_wider_metrics) as read_file:
        content = json.load(read_file)
        outputs.extend(content)
    return outputs


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


def custom_ap_eval(config_path, work_dir, res_pkl, outputs):
    """ Computes AP on faces that are greater than 64x64. """

    res_custom_metrics = os.path.join(work_dir, "custom_metrics.json")
    subprocess.run(
        f'python {FACE_DETECTION_TOOLS}/wider_custom_eval.py'
        f' {config_path} {res_pkl} --out {res_custom_metrics}'.split(' '), check=True)
    with open(res_custom_metrics) as read_file:
        ap_64x64 = [x['average_precision'] for x in json.load(read_file) if x['object_size'][0] == 64][0]
        outputs.append({'key': 'ap_64x64', 'value': ap_64x64, 'display_name': 'AP for faces > 64x64', 'unit': '%'})
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


def eval(config_path, snapshot, wider_dir, out):
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
    metrics = coco_ap_eval(config_path, work_dir, snapshot, res_pkl, metrics)
    metrics = custom_ap_eval(config_path, work_dir, res_pkl, metrics)
    metrics = compute_wider_metrics(config_path, snapshot, work_dir, wider_dir, metrics)

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
    eval(args.config, args.snapshot, args.wider_dir, args.out)


if __name__ == '__main__':
    main()
