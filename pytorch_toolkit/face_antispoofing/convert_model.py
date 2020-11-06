"""
 Copyright (c) 2020 Intel Corporation
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
import os

import torch

from utils import build_model, load_checkpoint, read_py_config

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='converting model to onnx')
    parser.add_argument('--GPU', type=int, default=0, required=False,
                        help='specify which gpu to use')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='path to configuration file')
    parser.add_argument('--model_path', type=str, default='MobileNetv3.onnx', required=False,
                        help='path where to save the model in onnx format')
    parser.add_argument('--img_size', type=tuple, default=(128,128), required=False,
                        help='height and width of the image to resize')
    parser.add_argument('--device', type=str, default='cuda',
                        help='if you want to eval model on cpu, pass "cpu" param')
    args = parser.parse_args()
    # read config
    path_to_config = args.config
    config = read_py_config(path_to_config)
    device = f'cuda:{args.GPU}' if args.device == 'cuda' else 'cpu'
    image_size = args.img_size
    save_path = args.model_path
    num_layers = args.num_layers
    export_onnx(config, device=device, num_layers=num_layers,
                img_size=image_size, save_path=save_path)

def export_onnx(config, device='cuda:0', num_layers=16,
                img_size=(128,128), save_path='model.onnx'):
    # get snapshot
    experiment_snapshot = config.checkpoint.snapshot_name
    experiment_path = config.checkpoint.experiment_path
    path_to_experiment = os.path.join(experiment_path, experiment_snapshot)
    # input to inference model
    dummy_input = torch.rand(size=(1,3,*img_size), device=device)
    # build model
    model = build_model(config, device, strict=True, mode='convert')
    model.to(device)
    # if model trained as data parallel object
    if config.data_parallel.use_parallel:
        model = torch.nn.DataParallel(model, **config.data_parallel.parallel_params)
    # load checkpoint from config
    load_checkpoint(path_to_experiment, model, map_location=torch.device(device),
                    optimizer=None, strict=True)
    # convert model to onnx
    model.eval()

    input_names = ["data"]
    output_names = ["probs"]
    torch.onnx.export(model, dummy_input, save_path, verbose=True,
                      input_names=input_names, output_names=output_names)

if __name__=='__main__':
    main()
