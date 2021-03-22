"""
 Copyright (c) 2020 Intel Corporation

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
import subprocess
import sys
import tempfile

from ote import MMDETECTION_TOOLS
from ote.datasets.face_detection.wider_face.convert_annotation import convert_to_coco
from ote.datasets.face_detection.wider_face.convert_predictions import convert_to_wider
from ote.metrics.detection.common import coco_ap_eval_det
from ote.metrics.face_detection.custom_voc_ap_eval import custom_voc_ap_evaluation
from ote.metrics.face_detection.wider_face.wider_face_eval import wider_face_evaluation


def compute_wider_metrics(config_path, work_dir, snapshot, wider_dir, **kwargs):
    """ Computes WiderFace metrics on easy, medium, hard subsets. """

    os.makedirs(wider_dir, exist_ok=True)

    outputs = []

    wider_data_zip = os.path.join(wider_dir, 'WIDER_val.zip')
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
        print('#########################################################################',
              file=sys.stderr)

        return outputs

    subprocess.run(f'unzip -q -o {wider_data_zip} -d {wider_dir}'.split(' '),
                   check=True)

    eval_tools_zip = os.path.join(wider_dir, 'eval_tools.zip')
    if not os.path.exists(eval_tools_zip):
        subprocess.run(
            f'wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip'
            f' -O {eval_tools_zip}'.split(' '), check=True)
    subprocess.run(f'unzip -q -o {eval_tools_zip} -d {wider_dir}'.split(' '),
                   check=True)

    wider_annotation_zip = os.path.join(wider_dir, 'wider_face_split.zip')
    if not os.path.exists(wider_annotation_zip):
        subprocess.run(
            f'wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
            f' -O {wider_annotation_zip}'.split(' '), check=True)
    subprocess.run(f'unzip -q -o {wider_annotation_zip} -d {wider_dir}'.split(' '),
                   check=True)

    wider_annotation = os.path.join(wider_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    wider_images = os.path.join(wider_dir, 'WIDER_val', 'images')
    wider_coco_annotation = os.path.join(wider_dir, 'instances_val.json')
    convert_to_coco(wider_annotation, wider_images, wider_coco_annotation, with_landmarks=False)

    res_pkl = os.path.join(work_dir, 'wider_face_res.pkl')

    test_py_stdout = os.path.join(work_dir, 'test_py_on_wider_stdout_')
    if snapshot.split('.')[-1] in {'xml', 'bin', 'onnx'}:
        if snapshot.split('.')[-1] == 'bin':
            snapshot = '.'.join(snapshot.split('.')[:-1]) + '.xml'
        tool = 'test_exported.py'
    else:
        tool = 'test.py'
    subprocess.run(
        f'python3 {MMDETECTION_TOOLS}/{tool}'
        f' {config_path} {snapshot}'
        f' --out {res_pkl}'
        f' --update_config data.test.ann_file={wider_coco_annotation} data.test.img_prefix={wider_dir}'
        f' | tee {test_py_stdout}',
        check=True, shell=True)

    wider_face_predictions = tempfile.mkdtemp()
    update_config = {
        'data.test.ann_file': wider_coco_annotation,
        'data.test.img_prefix': wider_dir
    }
    convert_to_wider(config_path, res_pkl, wider_face_predictions, update_config)

    res_wider_metrics = os.path.join(work_dir, "wider_metrics.json")
    wider_face_evaluation(wider_face_predictions,
                          os.path.join(wider_dir, 'eval_tools/ground_truth'),
                          iou_thresh=0.5,
                          out=res_wider_metrics)
    with open(res_wider_metrics) as read_file:
        content = json.load(read_file)
        outputs.extend(content)
    return outputs


def custom_ap_eval(config_path, work_dir, snapshot, update_config, **kwargs):
    """ Computes AP on faces that are greater than 64x64. """

    assert isinstance(update_config, dict)

    outputs = []

    res_pkl = os.path.join(work_dir, 'res.pkl')
    if not os.path.exists(res_pkl):
        # produces res.pkl
        coco_ap_eval_det(config_path, work_dir, snapshot, update_config)
    res_custom_metrics = os.path.join(work_dir, "custom_metrics.json")
    custom_voc_ap_evaluation(config_path, res_pkl, 0.5, (1024, 1024), res_custom_metrics, update_config)
    with open(res_custom_metrics) as read_file:
        ap_64x64 = [x['average_precision'] for x in json.load(read_file) if x['object_size'][0] == 64][0]
        outputs.append({
            'key': 'ap_64x64', 'value': ap_64x64, 'display_name': 'AP for faces > 64x64', 'unit': '%'
        })

    return outputs
