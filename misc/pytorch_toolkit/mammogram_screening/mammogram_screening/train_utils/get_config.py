import os
import json


def get_config(action, stage='stage1'):
    """ action: train, test, export or gdrive

    """
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    config_path = os.path.join(root_path, 'configs')

    if action == 'download':
        with open(os.path.join(config_path, 'download_configs.json')) as f1:
            config = json.load(f1)
    else:
        if stage == 'stage1':
            with open(os.path.join(config_path, 'network_stage1_configs.json')) as f1:
                config_file = json.load(f1)
        else:
            with open(os.path.join(config_path, 'network_stage2_configs.json')) as f1:
                config_file = json.load(f1)
        config = config_file[action]

    return config
