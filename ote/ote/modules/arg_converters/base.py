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

from abc import ABCMeta, abstractmethod
from collections import namedtuple


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
                  (i.e. they should have the same form as the keys of a dict passed to mmcv.Config.merge_from_dict)

        The map is used to update the script config file by the data from cmd line arguments of ote/tools/train.py

        Note that the pairs {key:value} (updating the script config) may be converted during
        calling of the script to another appropriate format compatible with the script itself
        (e.g. to the format compatible with yacs.config.CfgNode.merge_from_list, etc)
        """

    @abstractmethod
    def test_update_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **test** argparser.parse_args
                  (e.g. `argparser = DefaultArgParser.get_test_parser()`)
            to:   the fields of mmdetection/mmaction/other config for its **test** script in the form
                  compatible with mmcv.Config.merge_from_dict
                  (i.e. they should have the same form as the keys of a dict passed to mmcv.Config.merge_from_dict)

        The map is used to update the script config file by the data from cmd line arguments of ote/tools/eval.py

        Note that the pairs {key:value} (updating the script config) may be converted just before
        calling of the script to another appropriate format compatible with the script itself
        (e.g. to the format compatible with yacs.config.CfgNode.merge_from_list, etc)
        """

    @abstractmethod
    def compress_update_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **compress** argparser.parse_args
                  (e.g. `argparser = DefaultArgParser.get_compression_parser()`)
            to:   the fields of mmdetection/mmaction/other config for its **train** script in the form
                  compatible with mmcv.Config.merge_from_dict
                  (i.e. they should have the same form as the keys of a dict passed to mmcv.Config.merge_from_dict)

        The map is used to update the script config file by the data from cmd line arguments of ote/tools/compress.py

        Note that the pairs {key:value} (updating the script config) may be converted just before
        calling of the script to another appropriate format compatible with the script itself
        (e.g. to the format compatible with yacs.config.CfgNode.merge_from_list, etc)

        Also note that compression runs the original train script with special tuned config;
        typically the parameters are the same as for training except learning rate and total_epochs parameters.
        """

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

    def compress_out_args_map(self):
        """
        Returns a map:
            from: the name of cmd line arguments of the corresponding **compress** argparser.parse_args
            to:   the names of additional parameters of the `__call__` method of the corresponding
                  ote trainer class (e.g. MMDetectionTrainer) in the form suitable
                  for call `trainer_class(**kwargs)`
                  (note that compression runs the same trainer class as training, but with tuned
                  config file)
        """
        return {
                'gpu_num': 'gpu_num',
                'tensorboard_dir': 'tensorboard_dir',
                'nncf_quantization': 'nncf_quantization',
                'nncf_pruning': 'nncf_pruning',
                'nncf_sparsity': 'nncf_sparsity',
                'nncf_binarization': 'nncf_binarization',
               }

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

    def get_extra_train_args(self, args):
        """ Gets from the parsed output of the corresponding ote train argparser.parse_args
            (e.g. `argparser = DefaultArgParser.get_train_parser()`)
            additional changes that should be applied to mmdetection/mmaction/other config file,
            the changes will be returned in the form compatible with mmcv.Config.merge_from_dict

            Note that the format may be converted just before calling of the train script to another
            appropriate format depending on the script itself, e.g. to the format compatible with
            yacs.config.CfgNode.merge_from_list, etc.
        """
        return {}

    def get_extra_test_args(self, args):
        """ Gets from the parsed output of the corresponding ote test argparser.parse_args
            (e.g. `argparser = DefaultArgParser.get_test_parser()`)
            additional changes that should be applied to mmdetection/mmaction/other config file,
            the changes will be returned in the form compatible with mmcv.Config.merge_from_dict

            Note that the format may be converted just before calling of the test script to another
            appropriate format depending on the script itself, e.g. to the format compatible with
            yacs.config.CfgNode.merge_from_list, etc.
        """
        return {}


    def get_extra_compress_args(self, args):
        """ Gets from the parsed output of the corresponding ote compress argparser.parse_args
            (e.g. `argparser = DefaultArgParser.get_compression_parser()`)
            additional changes that should be applied to mmdetection/mmaction/other config file,
            the changes will be returned in the form compatible with mmcv.Config.merge_from_dict

            Note that the format may be converted just before calling of the train script to another
            appropriate format depending on the script itself, e.g. to the format compatible with
            yacs.config.CfgNode.merge_from_list, etc.
        """
        return {}

HooksForAction = namedtuple('HooksForAction',
                            ['update_args_map_hook', 'get_extra_args_hook', 'out_args_map_hook'])

class GroupHooksForActions:
    @staticmethod
    def get_hooks_for_train(arg_conv_maps):
        return HooksForAction(update_args_map_hook=arg_conv_maps.train_update_args_map,
                              get_extra_args_hook=arg_conv_maps.get_extra_train_args,
                              out_args_map_hook=arg_conv_maps.train_out_args_map)

    @staticmethod
    def get_hooks_for_test(arg_conv_maps):
        return HooksForAction(update_args_map_hook=arg_conv_maps.test_update_args_map,
                              get_extra_args_hook=arg_conv_maps.get_extra_test_args,
                              out_args_map_hook=arg_conv_maps.test_out_args_map)

    @staticmethod
    def get_hooks_for_compress(arg_conv_maps):
        return HooksForAction(update_args_map_hook=arg_conv_maps.compress_update_args_map,
                              get_extra_args_hook=arg_conv_maps.get_extra_compress_args,
                              out_args_map_hook=arg_conv_maps.compress_out_args_map)


class ArgConverter:
    def __init__(self, arg_conv_maps):
        assert isinstance(arg_conv_maps, ArgConverterMaps)
        self.arg_conv_maps = arg_conv_maps

    @staticmethod
    def _convert_args_by_hooks_for_action(hooks_for_action, args):
        assert isinstance(hooks_for_action, HooksForAction)

        update_args_map = hooks_for_action.update_args_map_hook()
        update_args = map_args(args, update_args_map)

        extra_args = hooks_for_action.get_extra_args_hook(args)
        update_args.update(extra_args)

        converted_args = {
            'config': args['config'],
            'update_config': update_args,
        }

        additional_converted_args_map = hooks_for_action.out_args_map_hook()
        additional_converted_args = map_args(args, additional_converted_args_map)
        converted_args.update(additional_converted_args)

        return converted_args

    def convert_train_args(self, args):
        hooks_for_action = GroupHooksForActions.get_hooks_for_train(self.arg_conv_maps)
        return self._convert_args_by_hooks_for_action(hooks_for_action, args)

    def convert_compress_args(self, args):
        hooks_for_action = GroupHooksForActions.get_hooks_for_compress(self.arg_conv_maps)
        return self._convert_args_by_hooks_for_action(hooks_for_action, args)

    def convert_test_args(self, args):
        hooks_for_action = GroupHooksForActions.get_hooks_for_test(self.arg_conv_maps)
        return self._convert_args_by_hooks_for_action(hooks_for_action, args)
