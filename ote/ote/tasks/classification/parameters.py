from ote.interfaces.parameters import BaseTaskParameters
from ote.modules.arg_converters import convert_args_to_parameters


def build_train_parameters(train_kwargs_dict, configs_root):
    return convert_args_to_parameters(train_kwargs_dict, configs_root,
                                      env_type=BaseTaskParameters.BaseEnvironmentParameters,
                                      target_type=BaseTaskParameters.BaseTrainingParameters)


def build_test_parameters(test_kwargs_dict, configs_root):
    return convert_args_to_parameters(test_kwargs_dict, configs_root,
                                      env_type=BaseTaskParameters.BaseEnvironmentParameters,
                                      target_type=BaseTaskParameters.BaseEvaluationParameters)


def build_export_parameters(export_kwargs_dict, configs_root):
    return convert_args_to_parameters(export_kwargs_dict, configs_root,
                                      env_type=BaseTaskParameters.BaseEnvironmentParameters,
                                      target_type=BaseTaskParameters.BaseExportParameters,
                                      use_update_conf=False)
