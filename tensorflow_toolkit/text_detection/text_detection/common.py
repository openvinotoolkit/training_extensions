import os
import yaml

def load_config(path):
    """ Load saved configuration from yaml file. """

    with open(path, "r") as read_file:
        config = yaml.load(read_file)
    return config


def parse_epoch(path):
    return int(os.path.basename(path).split('.')[0].split('-')[-1])
