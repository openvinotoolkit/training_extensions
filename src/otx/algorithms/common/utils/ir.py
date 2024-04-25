"""Collections of IR-related utils for common OTX algorithms."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import Any, Dict, Tuple

from openvino import Type
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, save_model

from otx.algorithms.common.utils.utils import is_xpu_available


def check_if_quantized(model: Any) -> bool:
    """Checks if OpenVINO model is already quantized."""
    nodes = model.get_ops()
    for op in nodes:
        if "FakeQuantize" == op.get_type_name():
            return True
    return False


def embed_ir_model_data(xml_file: str, data_items: Dict[Tuple[str, str], Any]) -> None:
    """Embeds serialized data to IR xml file.

    Args:
        xml_file : a path to IR xml file.
        data_items : a dict with tuple-keyworded serialized objects.
    """

    core = Core()
    model = core.read_model(xml_file)
    for k, data in data_items.items():
        model.set_rt_info(data, list(k))

    # workaround for CVS-138901
    if is_xpu_available():
        pre_post_processor = PrePostProcessor(model)
        for output in model.outputs:
            if "labels" in output.get_names() and output.get_element_type() == Type.f32:
                pre_post_processor.output("labels").tensor().set_element_type(Type.i64)
        model = pre_post_processor.build()

    # workaround for CVS-110054
    tmp_xml_path = Path(Path(xml_file).parent) / "tmp.xml"
    save_model(model, str(tmp_xml_path), compress_to_fp16=False)
    tmp_xml_path.rename(xml_file)
    Path(str(tmp_xml_path.parent / tmp_xml_path.stem) + ".bin").unlink()
