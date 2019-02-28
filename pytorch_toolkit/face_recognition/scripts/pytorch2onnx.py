"""
 Copyright (c) 2018 Intel Corporation
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

from utils.utils import load_model_state
from model.common import models_backbones, models_landmarks

def main():
    parser = argparse.ArgumentParser(description='Conversion script for FR models from PyTorch to ONNX')
    parser.add_argument('--embed_size', type=int, default=128, help='Size of the face embedding.')
    parser.add_argument('--snap', type=str, required=True, help='Snapshot to convert.')
    parser.add_argument('--device', '-d', default=-1, type=int, help='Device for model placement.')
    parser.add_argument('--output_dir', default='./', type=str, help='Output directory.')
    parser.add_argument('--model', choices=list(models_backbones.keys()) + list(models_landmarks.keys()),
                        type=str, default='rmnet')

    args = parser.parse_args()

    if args.model in models_landmarks.keys():
        model = models_landmarks[args.model]()
    else:
        model = models_backbones[args.model](embedding_size=args.embed_size, feature=True)

    model = load_model_state(model, args.snap, args.device, eval_state=True)
    input_var = torch.rand(1, 3, *model.get_input_res())
    dump_name = args.snap[args.snap.rfind('/') + 1:-3]

    torch.onnx.export(model, input_var, dump_name + '.onnx', verbose=True, export_params=True)

if __name__ == '__main__':
    main()
