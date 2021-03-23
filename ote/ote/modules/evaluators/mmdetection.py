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

from ote import MMDETECTION_TOOLS

from .base import BaseEvaluator
from ..registry import EVALUATORS


@EVALUATORS.register_module()
class MMDetectionEvaluator(BaseEvaluator):

    def _get_tools_dir(self):
        return MMDETECTION_TOOLS

    def _get_metric_functions(self):
        from ote.metrics.detection.common import coco_ap_eval_det

        return [coco_ap_eval_det]

    def _get_image_shape(self, cfg):
        image_shape = [x['img_scale'] for x in cfg.test_pipeline if 'img_scale' in x][0][::-1]
        image_shape = " ".join([str(x) for x in image_shape])
        return image_shape
