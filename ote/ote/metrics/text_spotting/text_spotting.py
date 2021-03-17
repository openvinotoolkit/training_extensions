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
import os

from mmcv import Config

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
    config = Config.fromfile(config_path)

    metric_keys = ['f1', 'word_spotting', 'e2e_recognition']
    metric_names = ['F1-score', 'Word Spotting (N)', 'End-to-End recognition (N)']

    if config.get('lexicon_mapping') and config.get('lexicon'):
        if os.path.exists(config.get('lexicon_mapping')) and os.path.exists(config.get('lexicon')):
            metric_keys.append('word_spotting@'
                               f'lexicon_mapping={config.get("lexicon_mapping")},'
                               f'lexicon={config.get("lexicon")}')
            metric_keys.append('e2e_recognition@'
                               f'lexicon_mapping={config.get("lexicon_mapping")},'
                               f'lexicon={config.get("lexicon")}')
            metric_names.append('Word Spotting (G)')
            metric_names.append('End-to-End recognition (G)')
        else:
            if not os.path.exists(config.get('lexicon_mapping')):
                logging.warning(f'Failed to find: {config.get("lexicon_mapping")}')
            if not os.path.exists(config.get('lexicon')):
                logging.warning(f'Failed to find: {config.get("lexicon")}')
            logging.warning('Skip computing metrics with lexicon.')

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
