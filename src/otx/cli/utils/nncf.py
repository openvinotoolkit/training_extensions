"""NNCF-related utils."""

# Copyright (C) 2021-2023 Intel Corporation
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

import torch


def is_checkpoint_nncf(path: str) -> bool:
    """Check if checkpoint is NNCF checkpoint.

    The function uses metadata stored in a checkpoint to check if the
    checkpoint was the result of training of NNCF-compressed model.
    """
    state = torch.load(path, map_location="cpu")
    is_nncf = bool(state.get("meta", {}).get("nncf_enable_compression")) or "nncf_metainfo" in state
    return is_nncf


def get_number_of_fakequantizers_in_xml(path_to_xml: str) -> int:
    """Return number of FakeQuantize layers.

    Return number of FakeQuantize layers in the model by parsing file without loading model.

    Args:
        path_to_xml (str): Path to xml file.

    Returns:
        int: Number of FakeQuantize layers.
    """
    num_fq = 0
    with open(path_to_xml, "r", encoding="UTF-8") as stream:
        for line in stream.readlines():
            if 'type="FakeQuantize"' in line:
                num_fq += 1
    return num_fq
