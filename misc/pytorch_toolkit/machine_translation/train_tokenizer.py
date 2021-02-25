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
from core.tokenizer.spbpe_tokenizer import SPBPETokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-list", type=str, help="path to txt file that contains list of corpuses")
    parser.add_argument("--vocab-size", type=int, default=32000, help="vocab size")
    parser.add_argument("--output", type=str, help="path to store trained tokenizer")
    args = parser.parse_args()

    SPBPETokenizer.train(
        args.corpus_list,
        args.vocab_size,
        args.output
    )
