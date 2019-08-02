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

from tools.debug.common import compare_activations, print_args

argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--ir-dump-txt", help="IE dump file in text format (ieb from mkldnn_graph.cpp)",
                       required=True)
argparser.add_argument("-p", "--torch-dump-npy", help="PyTorch dump file in npy format", required=True)
args = argparser.parse_args()
print_args(args)

def main():
    compare_activations(args.ir_dump_txt, args.torch_dump_npy)


if __name__ == '__main__':
    main()
