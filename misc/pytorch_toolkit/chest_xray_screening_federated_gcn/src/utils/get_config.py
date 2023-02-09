import os
import json

def get_config(action, gnn=True, config_path=""):

    root_path = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
    config_path = os.path.join(root_path, 'configs')

    if action == 'download':
        with open(os.path.join(config_path, 'download_configs.json')) as f1:
            config = json.load(f1)
    else:
        if gnn is True:
            with open(os.path.join(config_path, 'fl_with_gnn.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]
        else:
            with open(os.path.join(config_path, 'fl_without_gnn.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]

    return config
