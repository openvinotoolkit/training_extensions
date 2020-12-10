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

import os
from abc import ABCMeta, abstractmethod

def map_args(src_args, args_map):
    return {v: src_args[k] for k, v in args_map.items()}

class ArgConverterMaps(metaclass=ABCMeta):
    @abstractmethod
    def train_update_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **train** argparser.parse_args
                  (e.g. `argparser = DefaultArgParser.get_train_parser()`)
            to:   the fields of mmdetection/mmaction/other config for its **train** script in the form
                  compatible with mmcv.Config.merge_from_dict
        """
        pass

    @abstractmethod
    def test_update_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **test** argparser.parse_args
                  (e.g. `argparser = DefaultArgParser.get_test_parser()`)
            to:   the fields of mmdetection/mmaction/other config for its **test** script in the form
                  compatible with mmcv.Config.merge_from_dict
        """
        pass

    @abstractmethod
    def compress_update_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **compress** argparser.parse_args
                  (e.g. `argparser = DefaultArgParser.get_compression_parser()`)
            to:   the fields of mmdetection/mmaction/other config for its **train** script in the form
                  compatible with mmcv.Config.merge_from_dict to makes compression
                  (note that compression makes the original train script with special tuned config;
                   typically the parameters are the same as for training except
                   learning rate and total_epochs parameters)

        """
        pass

    @abstractmethod
    def train_to_compress_update_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **train** argparser.parse_args
            to:   the fields of mmdetection/mmaction/other config for its **train** script in the form
                  compatible with mmcv.Config.merge_from_dict to make **compression**
                  -- it is used if compression should be run inside ote train.py just after
                     finetuning
                     (typically it is the same as `compress_update_args_map` or close to it)
        """
        pass

    @abstractmethod
    def train_out_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **train** argparser.parse_args
            to:   the names of additional parameters of the `__call__` method of the corresponding ote trainer class
                  (e.g. MMDetectionTrainer) in the form suitable for `trainer_class(**kwargs)`
        """
        return {
                'gpu_num': 'gpu_num',
                'tensorboard_dir': 'tensorboard_dir'
               }
    @abstractmethod
    def compress_out_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **compress** argparser.parse_args
            to:   the names of additional parameters of the `__call__` method of the corresponding
                  ote trainer class (e.g. MMDetectionTrainer) in the form suitable
                  for call `trainer_class(**kwargs)`
                  (note that compression make the same trainer class as training, but with tuned
                  config file)
        """
        return {
                'gpu_num': 'gpu_num',
                'tensorboard_dir': 'tensorboard_dir'
               }
    @abstractmethod
    def test_out_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **test** argparser.parse_args
            to:   the names of additional parameters of the `__call__` method of the corresponding ote
                  evaluator class (e.g. MMDetectionEvaluator) in the form suitable for
                  call `evaluator_class(**kwargs)`
        """
        return {
                'load_weights': 'snapshot',
                'save_metrics_to': 'out',
                'save_output_to': 'show_dir'
               }

    @abstractmethod
    def get_extra_train_args(self, args):
        """ Gets from the parsed output of the corresponding ote train argparser.parse_args
            (e.g. `argparser = DefaultArgParser.get_train_parser()`)
            additional changes that should be applied to mmdetection/mmation/other training config file,
            the changes will be in the form compatible with mmcv.Config.merge_from_dict
        """
        return {}

    @abstractmethod
    def get_extra_test_args(self, args):
        """ Gets from the parsed output of the corresponding ote train argparser.parse_args
            (e.g. `argparser = DefaultArgParser.get_train_parser()`)
            additional changes that should be applied to mmdetection/mmation/other testing config file,
            the changes will be in the form compatible with mmcv.Config.merge_from_dict
        """
        return {}

class BaseArgConverter:

    # for update_converted_args_to_load_from_snapshot
    field_load_from = 'load_from'
    field_resume_from = 'resume_from'

    def __init__(self, arg_converter_maps):
        assert isinstance(arg_converter_maps, ArgConverterMaps)
        self.maps = arg_converter_maps

    def convert_train_args(self, model_template_path, args):
        update_args = map_args(args, self.maps.train_update_args_map())

        extra_args = self.maps.get_extra_train_args(args)
        update_args.update(extra_args)

        template_folder = os.path.dirname(model_template_path)
        converted_args = {
            'config': os.path.join(template_folder, args['config']),
            'out': os.path.join(args['save_checkpoints_to'], model_template_path),
            'update_config': update_args,
        }
        converted_args.update(map_args(args, self.maps.train_out_args_map()))

        return converted_args

    def convert_compress_args(self, model_template_path, args):
        update_args = map_args(args, self.maps.compress_update_args_map())

        extra_args = self.maps.get_extra_train_args(args)
        update_args.update(extra_args)

        template_folder = os.path.dirname(model_template_path)
        converted_args = {
            'config': os.path.join(template_folder, args['config']),
            'out': os.path.join(args['save_checkpoints_to'], model_template_path),
            'update_config': update_args,
        }
        converted_args.update(map_args(args, self.maps.compress_out_args_map()))

        return converted_args

    def convert_train_args_to_compress_args(self, model_template_path, args):
        update_args = map_args(args, self.maps.train_to_compress_update_args_map())

        extra_args = self.maps.get_extra_train_args(args)
        update_args.update(extra_args)

        template_folder = os.path.dirname(model_template_path)
        converted_args = {
            'config': os.path.join(template_folder, args['config']),
            'out': os.path.join(args['save_checkpoints_to'], model_template_path),
            'update_config': update_args,
        }
        converted_args.update(map_args(args, self.maps.compress_out_args_map()))

        return converted_args

    def update_converted_args_to_load_from_snapshot(self, converted_args, snapshot_path):
        converted_args['update_config'][self.field_load_from] = snapshot_path
        converted_args['update_config'][self.field_resume_from] = ''

    def convert_test_args(self, model_template_path, args):
        update_args = map_args(args, self.maps.test_update_args_map())

        extra_args = self.maps.get_extra_test_args(args)
        update_args.update(extra_args)

        template_folder = os.path.dirname(model_template_path)
        converted_args = {
            'config': os.path.join(template_folder, args['config']),
            'update_config': update_args,
        }
        converted_args.update(map_args(args, self.maps.test_out_args_map()))

        return converted_args
