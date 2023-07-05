"""Collections of IR-related utils for common OTX algorithms."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, Tuple

from openvino.runtime import Core, serialize


def embed_ir_model_data(xml_file: str, data_items: Dict[Tuple[str], Any]) -> None:
    """Embeds serialized data to IR xml file.

    Args:
        xml_file : a path to IR xml file.
        data_items : a dict with tuple-keyworded serialized objects.
    """

    core = Core()
    model = core.read_model(xml_file)
    for k, data in data_items.items():
        model.set_rt_info(data, list(k))
    serialize(model, xml_file)
