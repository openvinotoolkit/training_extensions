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

from text_recognition.utils.get_config import get_config
from text_recognition.utils.evaluator import Evaluator


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    return args.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    test_config = get_config(arguments.config, section='eval')
    validator = Evaluator(test_config)
    if 'model_folder' in test_config.keys():
        model_folder = test_config.get('model_folder')
        best_model, best_result = None, 0
        for model in os.listdir(model_folder):
            validator.runner.reload_model(os.path.join(model_folder, model))
            result = validator.validate()
            if result > best_result:
                best_result = result
                best_model = os.path.join(model_folder, model)
        print('model = {}'.format(best_model))
        result = best_result
    else:
        result = validator.validate()
    print('Result metric is: {}'.format(result))
