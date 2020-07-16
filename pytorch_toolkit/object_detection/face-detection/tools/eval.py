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
import json
import os
import subprocess
import sys
import tempfile

from common.misc import evaluate, coco_ap_eval

MMDETECTION_TOOLS = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'
FACE_DETECTION_TOOLS = os.path.dirname(__file__)


def parse_args():
    """ Parses input args. """

    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='A path to model training configuration file (.py).')
    parser.add_argument('snapshot',
                        help='A path to pre-trained snapshot (.pth).')
    parser.add_argument('out',
                        help='A path to output file where models metrics will be saved (.yml).')
    parser.add_argument('--wider_dir',
                        help='Specify this  path if you would like to test your model on WiderFace dataset.',
                        default='data/wider_face')
    parser.add_argument('--update_config',
                        help='Update configuration file by parameters specified here.'
                             'Use quotes if you are going to change several params.',
                        default='')

    return parser.parse_args()


def get_compute_wider_metrics(wider_dir):
    def compute_wider_metrics(config_path, work_dir, snapshot, outputs, update_args):
        """ Computes WiderFace metrics on easy, medium, hard subsets. """

        wider_data_folder = wider_dir
        os.makedirs(wider_data_folder, exist_ok=True)

        wider_data_zip = os.path.join(wider_data_folder, 'WIDER_val.zip')
        if not os.path.exists(wider_data_zip):
            print('', file=sys.stderr)
            print('#########################################################################',
                  file=sys.stderr)
            print('Cannot compute WiderFace metrics, failed to find WIDER_val.zip here:',
                  file=sys.stderr)
            print(f'    {os.path.abspath(wider_data_zip)}', file=sys.stderr)
            print('Please download the data from', file=sys.stderr)
            print('    https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view',
                  file=sys.stderr)
            print('Save downloaded data as:', file=sys.stderr)
            print(f'    {os.path.abspath(wider_data_zip)}', file=sys.stderr)
            print(f'#########################################################################',
                  file=sys.stderr)

            return outputs

        subprocess.run(f'unzip -q -o {wider_data_zip} -d {wider_data_folder}'.split(' '),
                       check=True)

        eval_tools_zip = os.path.join(wider_data_folder, 'eval_tools.zip')
        if not os.path.exists(eval_tools_zip):
            subprocess.run(
                f'wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip'
                f' -O {eval_tools_zip}'.split(' '), check=True)
        subprocess.run(f'unzip -q -o {eval_tools_zip} -d {wider_data_folder}'.split(' '),
                       check=True)

        wider_annotation_zip = os.path.join(wider_data_folder, 'ider_face_split.zip')
        if not os.path.exists(wider_annotation_zip):
            subprocess.run(
                f'wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
                f' -O {wider_annotation_zip}'.split(' '), check=True)
        subprocess.run(f'unzip -q -o {wider_annotation_zip} -d {wider_data_folder}'.split(' '),
                       check=True)

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
                f' --update_config data.test.ann_file={wider_coco_annotation} data.test.img_prefix={wider_dir}'.split(
                    ' '),
                stdout=test_py_stdout, check=True)

        wider_face_predictions = tempfile.mkdtemp()
        subprocess.run(
            f'python {FACE_DETECTION_TOOLS}/test_out_to_wider_predictions.py'
            f' {config_path} {res_pkl} {wider_face_predictions}'
            f' --update_config data.test.ann_file={wider_coco_annotation} data.test.img_prefix={wider_dir}'.split(
                ' '),
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

    return compute_wider_metrics


def custom_ap_eval(config_path, work_dir, snapshot, outputs, update_config):
    """ Computes AP on faces that are greater than 64x64. """

    res_pkl = os.path.join(work_dir, 'res.pkl')
    if not os.path.exists(res_pkl):
        # produces res.pkl
        coco_ap_eval(config_path, work_dir, snapshot, outputs, update_config)
    res_custom_metrics = os.path.join(work_dir, "custom_metrics.json")
    update_config = f' --update_config {update_config}' if update_config else ''
    subprocess.run(
        f'python {FACE_DETECTION_TOOLS}/wider_custom_eval.py'
        f' {config_path} {res_pkl} --out {res_custom_metrics}{update_config}'.split(' '),
        check=True)
    with open(res_custom_metrics) as read_file:
        ap_64x64 = \
        [x['average_precision'] for x in json.load(read_file) if x['object_size'][0] == 64][0]
        outputs.append(
            {'key': 'ap_64x64', 'value': ap_64x64, 'display_name': 'AP for faces > 64x64',
             'unit': '%'})
    return outputs


def main(config, snapshot, out, update_config, wider_dir):
    """ Main function. """

    metrics_functions = [coco_ap_eval, custom_ap_eval, get_compute_wider_metrics(wider_dir)]
    evaluate(config, snapshot, out, update_config, metrics_functions)


if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.snapshot, args.out, args.update_config, args.wider_dir)
