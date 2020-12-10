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

from ote.metrics.detection.common import coco_ap_eval


def coco_ap_eval_bbox_segm_f1_wordspotting(config_path, work_dir, snapshot, update_config, show_dir='', **kwargs):
    return coco_ap_eval(
        config_path, work_dir, snapshot, update_config, show_dir,
        metric_names=['Bbox AP @ [IoU=0.50:0.95]', 'Segm AP @ [IoU=0.50:0.95]', 'F1-score', 'Word Spotting'],
        metrics='bbox segm f1 word_spotting')
