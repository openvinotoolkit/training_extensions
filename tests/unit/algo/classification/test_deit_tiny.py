# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import pytest
from otx.algo.classification.deit_tiny import (
    DeitTinyForHLabelCls,
    DeitTinyForMulticlassCls,
    DeitTinyForMultilabelCls,
)
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType


class TestDeitTiny:
    @pytest.fixture(
        params=[
            (DeitTinyForMulticlassCls, "fxt_multiclass_cls_batch_data_entity"),
            (DeitTinyForMultilabelCls, "fxt_multilabel_cls_batch_data_entity"),
            (DeitTinyForHLabelCls, "fxt_hlabel_cls_batch_data_entity"),
        ],
        ids=["multiclass", "multilabel", "hlabel"],
    )
    def fxt_model_and_input(self, request, fxt_hlabel_data):
        model_cls, input_fxt_name = request.param
        fxt_input = request.getfixturevalue(input_fxt_name)
        num_classes = fxt_hlabel_data.num_classes

        if model_cls == DeitTinyForHLabelCls:
            model = model_cls(hlabel_info=fxt_hlabel_data)
        else:
            model = model_cls(num_classes=num_classes)

        return model, fxt_input

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_deit_tiny(self, fxt_model_and_input, explain_mode, mocker):
        fxt_model, fxt_input = fxt_model_and_input

        fxt_model.train()
        assert isinstance(fxt_model.forward(fxt_input), OTXBatchLossEntity)

        fxt_model.eval()
        assert not isinstance(fxt_model.forward(fxt_input), OTXBatchLossEntity)

        fxt_model.explain_mode = explain_mode
        preds = fxt_model.predict_step(fxt_input, batch_idx=0)
        assert len(preds.labels) == fxt_input.batch_size
        assert len(preds.scores) == fxt_input.batch_size
        assert preds.has_xai_outputs == explain_mode

        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_b0_ckpt")
        fxt_model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multiclass", "model.model.")

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_export(self, fxt_model_and_input, explain_mode, tmpdir):
        base_name = "exported_model"

        fxt_model, _ = fxt_model_and_input
        fxt_model.eval()
        fxt_model.explain_mode = explain_mode

        fxt_model.export(
            output_dir=Path(tmpdir),
            base_name=base_name,
            export_format=OTXExportFormatType.OPENVINO,
            precision=OTXPrecisionType.FP16,
        )
