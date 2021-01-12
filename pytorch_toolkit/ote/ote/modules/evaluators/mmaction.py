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

from ote import MMACTION_TOOLS

from .base import BaseEvaluator
from ..registry import EVALUATORS


@EVALUATORS.register_module()
class MMActionEvaluator(BaseEvaluator):
    def _get_tools_dir(self):
        return MMACTION_TOOLS

    def _get_metric_functions(self):
        from ote.metrics.classification.common import mean_accuracy_eval

        return [mean_accuracy_eval]

    def _get_image_shape(self, cfg):
        image_size = cfg.input_img_size if isinstance(cfg.input_img_size, (tuple, list)) else [cfg.input_img_size] * 2
        assert len(image_size) == 2

        image_shape = [cfg.input_clip_length, image_size[0], image_size[1]]
        image_shape = " ".join([str(x) for x in image_shape])

        return image_shape
