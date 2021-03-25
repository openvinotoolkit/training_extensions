import os

from ote.interfaces.parameters import BaseTaskParameters


def is_field(name, t):
    if hasattr(t, name):
        return True
    return False


def build_train_parameters(train_kwargs_dict, configs_root):
    config_path = train_kwargs_dict['config']
    env_params_kwargs = {}
    train_params_kwargs = {}
    env_params_kwargs['config_path'] = os.path.join(configs_root, config_path)
    for k, v in train_kwargs_dict['update_config'].items():
        if v:
            if is_field(k, BaseTaskParameters.BaseEnvironmentParameters):
                env_params_kwargs[k] = v
            elif is_field(k, BaseTaskParameters.BaseTrainingParameters):
                train_params_kwargs[k] = v

    env_params = BaseTaskParameters.BaseEnvironmentParameters(**env_params_kwargs)
    train_params = BaseTaskParameters.BaseTrainingParameters(**train_params_kwargs)

    return env_params, train_params


def build_test_parameters(test_kwargs_dict, configs_root):
    config_path = test_kwargs_dict['config']
    env_params_kwargs = {}
    test_params_kwargs = {}
    env_params_kwargs['config_path'] = os.path.join(configs_root, config_path)
    for k, v in test_kwargs_dict['update_config'].items():
        if v:
            if is_field(k, BaseTaskParameters.BaseEnvironmentParameters):
                env_params_kwargs[k] = v
            elif is_field(k, BaseTaskParameters.BaseEvaluationParameters):
                test_params_kwargs[k] = v

    env_params = BaseTaskParameters.BaseEnvironmentParameters(**env_params_kwargs)
    test_params = BaseTaskParameters.BaseEvaluationParameters(**test_params_kwargs)

    return env_params, test_params


def build_export_parameters(export_kwargs_dict, configs_root):
    config_path = export_kwargs_dict['config']
    env_params_kwargs = {}
    export_params_kwargs = {}
    env_params_kwargs['config_path'] = os.path.join(configs_root, config_path)
    env_params_kwargs['snapshot_path'] = export_kwargs_dict['load_weights']

    for k, v in export_kwargs_dict.items():
        if v:
            if is_field(k, BaseTaskParameters.BaseEnvironmentParameters):
                env_params_kwargs[k] = v
            elif is_field(k, BaseTaskParameters.BaseExportParameters):
                export_params_kwargs[k] = v

    env_params = BaseTaskParameters.BaseEnvironmentParameters(**env_params_kwargs)
    export_params = BaseTaskParameters.BaseExportParameters(**export_params_kwargs)

    return env_params, export_params
