"""Data Pipeline for Self-SL model of Segmentation Task."""

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

# pylint: disable=invalid-name

__resize_target_size = 224
__img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

__train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='TwoCropTransform',
         view0=[
            dict(type='RandomResizedCrop', size=__resize_target_size),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='ProbCompose',
                 transforms=[
                    dict(type='ColorJitter',
                         brightness=0.4,
                         contrast=0.4,
                         saturation=0.2,
                         hue=0.1)
                 ],
                 probs=[0.8]),
            dict(type='RandomGrayscale', p=0.2),
            dict(type='GaussianBlur', kernel_size=23),
            dict(type='Normalize', **__img_norm_cfg),
         ],
         view1=[
            dict(type='RandomResizedCrop', size=__resize_target_size),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='ProbCompose',
                 transforms=[
                    dict(type='ColorJitter',
                         brightness=0.4,
                         contrast=0.4,
                         saturation=0.2,
                         hue=0.1)
                 ],
                 probs=[0.8]),
            dict(type='RandomGrayscale', p=0.2),
            dict(type='ProbCompose', transforms=[dict(type='GaussianBlur', kernel_size=23)], probs=[0.1]),
            dict(type='ProbCompose', transforms=[dict(type='Solarization', threshold=128)], probs=[0.2]),
            dict(type='Normalize', **__img_norm_cfg),
         ]
    ),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

data = dict(train=dict(type='MPASegDataset', pipeline=__train_pipeline))
