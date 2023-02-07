import os
import json

def get_config(action, stage=1, config_path="configs/"):

    root_path = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
    config_path = os.path.join(root_path, 'configs')

    if action == 'download':
        with open(os.path.join(config_path, 'download_config.json')) as f1:
            config = json.load(f1)
    else:
        if stage == 1:
            with open(os.path.join(config_path, 'stage1_config.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]
        else:
            with open(os.path.join(config_path, 'stage2_config.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]

    return config
