"""
 Copyright (c) 2019 Intel Corporation

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

import argparse
import torch
import numpy as np
from PIL import Image

from config.default_config import get_default_config

from data.transforms import build_transforms
from torchreid.utils import load_pretrained_weights

from models.builder import build_model


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--output-name', type=str, default='model')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Verbose mode for onnx.export')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_model(
            name=cfg.model.name,
            num_classes=1041,  # Does not matter in conversion
            loss=cfg.loss.name,
            pretrained=False,
            use_gpu=True,
            feature_dim=cfg.model.feature_dim,
            fpn_cfg=cfg.model.fpn,
            pooling_type=cfg.model.pooling_type,
            input_size=(cfg.data.height, cfg.data.width),
            dropout_cfg=cfg.model.dropout,
            IN_first=cfg.model.IN_first,
            extra_blocks=cfg.model.extra_blocks
        )

    load_pretrained_weights(model, cfg.model.load_weights)
    model.eval()

    _, transform = build_transforms(
        cfg.data.height, cfg.data.width,
        transforms=cfg.data.transforms,
        norm_mean=cfg.data.norm_mean,
        norm_std=cfg.data.norm_std,
        apply_masks_to_test=False
    )

    input_size = (cfg.data.height, cfg.data.width, 3)
    img = np.random.rand(*input_size).astype(np.float32)
    img = np.uint8(img * 255)
    im = Image.fromarray(img)
    blob = transform(im).unsqueeze(0)

    torch.onnx.export(model, blob, args.output_name + '.onnx',
                      verbose=False, export_params=True,
                      input_names=['data'], output_names=['reid_embedding'],
                      opset_version=9)  # 9th version resolves nearest upsample issue


if __name__ == '__main__':
    main()
