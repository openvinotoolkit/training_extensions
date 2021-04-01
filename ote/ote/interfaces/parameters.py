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

from dataclasses import dataclass

class BaseTaskParameters:
    @dataclass
    class BaseEnvironmentParameters:
        config_path: str = ''
        gpu_num: int = 1
        load_weights: str = ''
        work_dir: str = ''

    @dataclass
    class BaseTrainingParameters:
        batch_size: int = 32
        max_num_epochs: int = 10
        base_learning_rate: int = 0.1
        resume_from: str = ''

    @dataclass
    class BaseEvaluationParameters:
        batch_size: int = 32

    @dataclass
    class BaseExportParameters:
        save_model_to: str = ''
        onnx: bool = True
        openvino: bool = True
        openvino_input_format: str = 'BGR'
        openvino_mo_args: str = ''

    @dataclass
    class BaseCompressParameters:
        pass
