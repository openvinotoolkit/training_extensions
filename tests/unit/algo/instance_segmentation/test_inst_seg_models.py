# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from otx.algo.instance_segmentation.maskrcnn import MaskRCNNEfficientNet, MaskRCNNResNet50, MaskRCNNSwinT
from otx.algo.instance_segmentation.rtmdet_inst import RTMDetInstTiny
from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchPredEntity,
)
from otx.core.types.export import OTXExportFormatType


class TestInstSegModel:
    @pytest.mark.parametrize(
        "model",
        [MaskRCNNEfficientNet(2), MaskRCNNResNet50(2), MaskRCNNSwinT(2), RTMDetInstTiny(2)],
    )
    def test_model(self, model, tmpdir, fxt_inst_seg_data_entity) -> None:
        _, _, batch_data_entity = fxt_inst_seg_data_entity
        output = model(batch_data_entity)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_mask" in output

        model.eval()
        output = model(batch_data_entity)
        assert isinstance(output, InstanceSegBatchPredEntity)

        exported_model_path = model.export(
            output_dir=Path(tmpdir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
        )
        Path.exists(exported_model_path)
