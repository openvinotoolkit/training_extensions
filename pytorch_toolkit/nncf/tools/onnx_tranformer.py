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

import sys
from argparse import ArgumentParser

import onnx

def rename_quantize(model):
    for node in model.graph.node:
        if node.op_type == 'Quantize':
            node.op_type = 'FakeQuantize'
            node.doc_string = 'Fake quantization operation'

def main(argv):
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-model', help='Path to input model file',
                        required=True)
    parser.add_argument('-o', '--output-model', help='Path to output model file',
                        required=True)
    args = parser.parse_args(args=argv)

    model = onnx.load(args.input_model)

    rename_quantize(model)

    onnx.save(model, args.output_model)


if __name__ == '__main__':
    main(sys.argv[1:])
