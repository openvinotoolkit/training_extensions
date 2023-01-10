import os
import json


def get_config(action, config_path, stage='stage1'):
    """ action: download, stage1

    """

    if action == 'download':
        with open(os.path.join(config_path, 'download_configs.json'),encoding='utf8') as f1:
            config = json.load(f1)
    else:
        if stage == 'stage1':
            with open(os.path.join(config_path, 'network_stage1_configs.json'),encoding='utf8') as f1:
                config_file = json.load(f1)
        else:
            with open(os.path.join(config_path, 'network_stage2_configs.json'),encoding='utf8') as f1:
                config_file = json.load(f1)
        config = config_file[action]

    return config
