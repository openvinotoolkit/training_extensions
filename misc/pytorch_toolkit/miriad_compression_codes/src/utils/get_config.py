import os
import json
from webbrowser import get


def get_config(action, phase, config_path=""):

    root_path = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
    config_path = os.path.join(root_path, 'configs')

    if action == 'download':
        with open(os.path.join(config_path, 'download_configs.json')) as f1:
            config = json.load(f1)
    else:
        if phase == 1:
            with open(os.path.join(config_path, 'phase1_config.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]
        else:
            with open(os.path.join(config_path, 'phase2_config.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]

    return config

conf = get_config('inference', 2, "/home/flamingo2/Downloads/training_extensions/misc/pytorch_toolkit/miriad_compression_codes/configs")
print(conf['path_to_latent'])