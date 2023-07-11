"""MMOVModel for otx.core.ov.models.mmov_model."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch

# TODO: Need to remove line 1 (ignore mypy) and fix mypy issues
from .ov_model import OVModel  # type: ignore[attr-defined]
from .parser_mixin import ParserMixin  # type: ignore[attr-defined]

# TODO: Need to fix pylint issues
# pylint: disable=keyword-arg-before-vararg


class MMOVModel(OVModel, ParserMixin):
    """MMOVModel for OMZ model type."""

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        *args,
        **kwargs
    ):
        parser = kwargs.pop("parser", None)
        parser_kwargs = kwargs.pop("parser_kwargs", {})
        inputs, outputs = super().parse(
            model_path_or_model=model_path_or_model,
            weight_path=weight_path,
            inputs=inputs,
            outputs=outputs,
            parser=parser,
            **parser_kwargs,
        )

        super().__init__(
            model_path_or_model=model_path_or_model,
            weight_path=weight_path,
            inputs=inputs,
            outputs=outputs,
            *args,
            **kwargs,
        )

    def forward(self, inputs, gt_label=None):
        """Function forward."""
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        assert len(inputs) == len(self.inputs)
        feed_dict = dict()
        for key, input_ in zip(self.inputs, inputs):
            feed_dict[key] = input_

        if gt_label is not None:
            assert "gt_label" not in self.features
            self.features["gt_label"] = gt_label

        outputs = super().forward(**feed_dict)
        outputs = tuple(outputs.values())
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs
