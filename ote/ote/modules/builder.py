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

from mmcv.utils.registry import Registry

from .arg_converters import ArgConverter
from .registry import ARG_PARSERS, ARG_CONVERTER_MAPS, TRAINERS, EVALUATORS, EXPORTERS, COMPRESSION


def build(obj_type, registry, args=None):
    if not isinstance(obj_type, str):
        raise TypeError(f'obj_type must be an str object, but got {type(obj_type)}')
    if not isinstance(registry, Registry):
        raise TypeError(f'registry must be an mmcv.Registry object, but got {type(registry)}')

    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise TypeError(f'args must be an dict object, but got {type(args)}')

    obj_cls = registry.get(obj_type)
    out_obj = obj_cls(**args)

    return out_obj


def build_arg_parser(obj_type):
    return build(obj_type, ARG_PARSERS)


def build_arg_converter(obj_type):
    return ArgConverter(build(obj_type, ARG_CONVERTER_MAPS))


def build_trainer(obj_type):
    return build(obj_type, TRAINERS)


def build_evaluator(obj_type):
    return build(obj_type, EVALUATORS)


def build_exporter(obj_type):
    return build(obj_type, EXPORTERS)

def build_compression_arg_transformer(obj_type):
    return build(obj_type, COMPRESSION)
