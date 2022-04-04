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
except ImportError as e:
    import warnings
    warnings.warn("ModelAPI was not found.")


class OTEMaskRCNNModel(MaskRCNNModel):
    __model__ = 'OTE_MaskRCNN'

    def _get_outputs(self):
        outputs = {}
        for layer_name in self.outputs:
            if layer_name.startswith('TopK'):
                continue
            layer_shape = self.outputs[layer_name].shape

            if len(layer_shape) == 1:
                outputs['labels'] = layer_name
            elif len(layer_shape) == 2:
                outputs['boxes'] = layer_name
            elif len(layer_shape) == 3:
                outputs['masks'] = layer_name
            elif len(layer_shape) == 4:
                outputs['feature_vector'] = layer_name
            else:
                raise WrapperError(self.__model__, "Unexpected output layer shape {} with name {}".format(layer_shape, layer_name))

        return outputs
