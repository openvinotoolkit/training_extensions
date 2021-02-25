#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
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

import argparse
import os
import torch.onnx
from segthor import train


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch SegTHOR export onnx")
    parser.add_argument("--name", default="test", type=str, help="Experiment name")
    parser.add_argument("--models_path", default="models", type=str, help="Path to models folder")
    parser.add_argument("--input_size", default=(64, 64, 64), help="Input image size", nargs="+", type=int)
    return parser.parse_args()

def main():
    opt = parse_args()
    print(opt)

    trainer = train.Trainer(name=opt.name, models_root=opt.models_path, rewrite=False, connect_tb=False)
    trainer.load_best()
    trainer.model = trainer.model.module.cpu()
    trainer.model = trainer.model.train(False)
    trainer.state.cuda = False

    x = torch.randn(1, 1, opt.input_size[0], opt.input_size[1], opt.input_size[2], requires_grad=True)

    export_dir = os.path.join(opt.models_path, opt.name)
    onnx_path = os.path.join(export_dir, opt.name+".onnx")
    torch.onnx.export(trainer.model,  # model being run
                      [x,],  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      verbose=True)  # store the trained parameter weights inside the model file


if __name__ == "__main__":
    main()
