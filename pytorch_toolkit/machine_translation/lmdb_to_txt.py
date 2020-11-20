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
from tqdm import tqdm
from core.dataset import LMDBContainer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to corpus files with comma separator like path1,path2,...,pathN")
    parser.add_argument("--output", type=str, help="path to output lmdb file")
    args = parser.parse_args()

    data = LMDBContainer(args.input)
    with open(args.output, 'w') as f:
        for sample in tqdm(data):
            f.write(sample["text"] + "\n")
