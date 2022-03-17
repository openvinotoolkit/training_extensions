import os
import sys
import numpy as np

from ote_sdk.configuration.helper import create as create_parameters_from_parameters_schema
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity, ModelOptimizationType, ModelPrecision, ModelStatus
from ote_sdk.entities.model_template import parse_model_template, TargetDevice
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType

from ote_cli.datasets import get_dataset_class
from ote_cli.registry import Registry
from ote_cli.utils.importing import get_impl_class

from torchtts.task import OTETextToSpeechTask


templates_dir = "../torchtts/integration/"
registry = Registry(templates_dir)
# registry = registry.filter(task_type=sys.executable.split(os.sep)[-4])
print(os.getcwd(), registry)


model_template = registry.get('TTS')
#model_template = parse_model_template(model_template_info['path'])
hyper_parameters = model_template.hyper_parameters.data
print(hyper_parameters)

hyper_parameters = create_parameters_from_parameters_schema(hyper_parameters)

for p in hyper_parameters.learning_parameters.parameters:
    print(f'{p}: {getattr(hyper_parameters.learning_parameters, p)}')


# create task
Task = get_impl_class(model_template.entrypoints.base)

environment = TaskEnvironment(
    model=None,
    hyper_parameters=hyper_parameters,
    label_schema=None,
    model_template=model_template
)

task = Task(task_environment=environment)
print(task)

from addict import Dict

cfg_data = Dict({
    "training_path": "../../datasets/data_ljspeech_melgan",
    "cmudict_path": "../data/cmu_dictionary",
    "text_cleaners": ["english_cleaners"],
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
    "add_noise": True,
    "add_blank": True
})

from torchtts.datasets import get_tts_datasets


trainset, valset = get_tts_datasets(cfg_data)
task.train_like_in_torch(trainset, valset)

# print(task._tokenizer)
