import os
import json

def get_config(action):
    """ action: train, val, export or gdrive
    """
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    config_path = os.path.join(root_path, 'configs')
    if action == 'download':
        with open(os.path.join(config_path, 'download_configs.json'), encoding="utf-8") as file:
            config = json.load(file)
    else:
        with open(os.path.join(config_path, 'encoder_config.json'), encoding="utf-8") as file:
            config_file = json.load(file)
        config = config_file[action]
    return config
    