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

from .mmdetection import MMDetectionArgConverterMap
from ..registry import ARG_CONVERTER_MAPS


@ARG_CONVERTER_MAPS.register_module()
class MMDetectionWiderArgConverterMap(MMDetectionArgConverterMap):
    def test_out_args_map(self):
        out_map = super().test_out_args_map()
        out_map.update({
            'wider_dir': 'wider_dir'
        })
        return out_map
