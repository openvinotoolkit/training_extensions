"""Adapter Wrapper of OTX Visual Prompting."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Union, Dict, Optional, Any
from pathlib import Path
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter
from openvino.runtime import Core


class VisualPromptingOpenvinoAdapter(OpenvinoAdapter):
    """OpenVINO Adapter Wrapper of OTX Visual Prompting.
    
    This wrapper has multiple models, image encoder and decoder for visual prompting.
    """
    def __init__(
        self,
        core: Core,
        model_paths: Dict[str, Union[str, Path]],
        weights_paths: Optional[Dict[str, Union[str, Path, None]]] = None,
        model_parameters: Dict = {},
        device: str = "CPU",
        plugin_config: Optional[dict] = None,
        max_num_requests: int = 0
    ):
        assert all(module in model_paths for module in ["image_encoder", "decoder"])
        self.image_encoder_adapter = OpenvinoAdapter(core, model_paths.get("image_encoder"), weights_paths.get("image_encoder", None), model_parameters, device, plugin_config, max_num_requests)
        self.decoder_adapter = OpenvinoAdapter(core, model_paths.get("decoder"), weights_paths.get("decoder", None), model_parameters, device, plugin_config, max_num_requests)

        self.device = device

    def load_model(self):
        """Load model of OTX Visual Prompting."""
        self.image_encoder_adapter.load_model()
        self.decoder_adapter.load_model()

    def get_input_layers(self) -> Dict[str, Dict[str, Any]]:
        """Get input layers of OTX Visual Prompting."""
        inputs = self.image_encoder_adapter.get_input_layers()
        inputs["decoder"] = self.decoder_adapter.get_input_layers()
        return inputs

    def get_output_layers(self) -> Dict[str, Dict[str, Any]]:
        """Get output layers of OTX Visual Prompting."""
        outputs = {}
        outputs["image_encoder"] = self.image_encoder_adapter.get_output_layers()
        outputs["decoder"] = self.decoder_adapter.get_output_layers()
        return outputs

    def set_callback(self, callback_fn):
        self.image_encoder_adapter.async_queue.set_callback(callback_fn)
        self.decoder_adapter.async_queue.set_callback(callback_fn)

    def is_ready(self) -> bool:
        return self.async_queue.is_ready()
