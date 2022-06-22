"""
 Copyright (c) 2022 Intel Corporation

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

try:
    from openvino.model_zoo.model_api.models.model import WrapperError
    from openvino.model_zoo.model_api.models.instance_segmentation import MaskRCNNModel
    from openvino.model_zoo.model_api.models.ssd import SSD
except ImportError as e:
    import warnings
    warnings.warn("ModelAPI was not found.")


class OTEMaskRCNNModel(MaskRCNNModel):
    __model__ = 'OTE_MaskRCNN'

    def _get_outputs(self):
        output_match_dict = {}
        outputs = ['boxes', 'labels', 'masks', 'feature_vector', 'saliency_map']
        for output in outputs:
            for output_name, output_meta in self.outputs.items():
                if output in output_meta.names:
                    output_match_dict[output] = output_name
                    break
        return output_match_dict


class OTESSDModel(SSD):
    __model__ = 'OTE_SSD'

    def _get_outputs(self):
        output_match_dict = {}
        outputs = ['boxes', 'labels', 'feature_vector', 'saliency_map']
        for output in outputs:
            for output_name, output_meta in self.outputs.items():
                if output in output_meta.names:
                    output_match_dict[output] = output_name
                    break
        return output_match_dict