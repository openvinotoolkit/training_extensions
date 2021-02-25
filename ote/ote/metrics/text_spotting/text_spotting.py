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
import logging

from ote.metrics.detection.common import run_test_script, update_outputs


def collect_hmeans(path):
    """ Collects hmean values in log file. """

    hmeans = []
    keyword = 'hmean='
    with open(path) as read_file:
        content = [line.strip() for line in read_file]
        for line in content:
            if keyword in line:
                hmeans.append(float(line.split(keyword)[-1]))
    return hmeans


def coco_ap_eval_f1_wordspotting(config_path, work_dir, snapshot, update_config, show_dir='', **kwargs):
    """ Computes text spotting metrics """

    metric_keys = ['f1', 'word_spotting']
    metric_names = ['F1-score', 'Word Spotting']
    outputs = []
    if not(update_config['data.test.ann_file'] and update_config['data.test.img_prefix']):
        logging.warning('Passed empty path to annotation file or data root folder. '
                        'Skipping text spotting metrics calculation.')
        update_outputs(outputs, metric_keys, metric_names, [None for _ in metric_keys])
    else:
        test_py_stdout = run_test_script(config_path, work_dir, snapshot,
                                         update_config, show_dir, ' '.join(metric_keys))

        hmeans = collect_hmeans(test_py_stdout)
        update_outputs(outputs, metric_keys, metric_names, hmeans)

    return outputs
