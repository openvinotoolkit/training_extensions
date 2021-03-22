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
