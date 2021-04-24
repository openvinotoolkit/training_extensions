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

from .mmdetection import MMDetectionEvaluator
from ..registry import EVALUATORS


@EVALUATORS.register_module()
class TextSpottingEvaluator(MMDetectionEvaluator):

    def _get_metric_functions(self):
        from ote.metrics.text_spotting.text_spotting import coco_ap_eval_f1_wordspotting
        return [coco_ap_eval_f1_wordspotting]
