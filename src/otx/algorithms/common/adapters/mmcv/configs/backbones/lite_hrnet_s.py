"""Backbone config of OCR-Lite-HRnet-s."""

# Copyright (C) 2022 Intel Corporation
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

model = dict(
    backbone=dict(
        type="LiteHRNet",
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        extra=dict(
            stem=dict(
                stem_channels=32,
                out_channels=32,
                expand_ratio=1,
                strides=(2, 2),
                extra_stride=True,
                input_norm=False,
            ),
            num_stages=2,
            stages_spec=dict(
                neighbour_weighting=False,
                weighting_module_version="v1",
                num_modules=(4, 4),
                num_branches=(2, 3),
                num_blocks=(2, 2),
                module_type=("LITE", "LITE"),
                with_fuse=(True, True),
                reduce_ratios=(8, 8),
                num_channels=(
                    (60, 120),
                    (60, 120, 240),
                ),
            ),
            out_modules=dict(
                conv=dict(enable=False, channels=160),
                position_att=dict(
                    enable=False,
                    key_channels=64,
                    value_channels=240,
                    psp_size=(1, 3, 6, 8),
                ),
                local_att=dict(enable=False),
            ),
            out_aggregator=dict(enable=False),
            add_input=False,
        ),
    ),
)
