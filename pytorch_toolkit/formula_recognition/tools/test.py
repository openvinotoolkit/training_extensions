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

from im2latex.utils.get_config import get_config
from im2latex.utils.evaluator import Evaluator, RunnerType


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_config = get_config(args.config, section='eval')
    model_folder = test_config.get("model_folder")
    if model_folder is not None:
        validator = Evaluator(test_config)
        best_model, best_result = None, 0
        for model in os.listdir(model_folder):
            validator.runner.reload_model(os.path.join(model_folder, model))
            result = validator.validate()
            if result > best_result:
                best_result = result
                best_model = os.path.join(model_folder, model)
        print("Im2latex metric is: {}".format(best_result))
        print("model = {}".format(best_model))
        exit(0)
    validator = Evaluator(test_config)
    result = validator.validate()
    print("Im2latex metric is: {}".format(result))
