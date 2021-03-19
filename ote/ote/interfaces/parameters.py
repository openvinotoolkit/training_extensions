"""
 Copyright (c) 2021 Intel Corporation

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


class BaseTaskParameters:
    class BaseEnvironmentParameters:
        config_path = ''
        gpus_num = 1
        snapshot_path = ''
        work_dir = ''

    class BaseTrainingParameters:
        batch_size = 32
        num_epochs = 10
        learning_rate = 0.1

    class BaseEvaluationParameters:
        batch_size = 1

    class BaseExportParameters:
        output_folder = ''

    class BaseCompressParameters:
        pass
