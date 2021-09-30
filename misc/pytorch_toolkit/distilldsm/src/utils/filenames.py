import os
import numpy as np
from src.utils import sequences as k_sequences
from src.utils import dataset as pytorch_datasets
from src.utils.utils import load_json

LOAD_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def format_templates(templates, directory="", **kwargs):
    if type(templates) == str:
        return os.path.join(directory, templates).format(**kwargs)
    else:
        return [os.path.join(directory, template).format(**kwargs) for template in templates]


def exists(filenames):
    if type(filenames) == str:
        filenames = [filenames]
    return all([os.path.exists(filename) for filename in filenames])


def generate_filenames_from_templates(subject_ids, feature_templates, target_templates=None, feature_sub_volumes=None,
                                      target_sub_volumes=None, raise_if_not_exists=False, directory="",
                                      skip_targets=False):
    filenames = list()
    # print(subject_ids)
    for subject_id in subject_ids:
        feature_filename = format_templates(feature_templates, directory=directory, subject=subject_id)
        # print(feature_filename)
        # if exists(feature_filename):
        #     print("Yes")
        if skip_targets:
            target_filename = None
        else:
            target_filename = format_templates(target_templates, directory=directory, subject=subject_id)
        # print(target_filename)
        if feature_sub_volumes is not None:
            _feature_sub_volumes = feature_sub_volumes
        else:
            _feature_sub_volumes = None
        if target_sub_volumes is not None:
            _target_sub_volumes = target_sub_volumes
        else:
            _target_sub_volumes = None
        if exists(feature_filename) and (skip_targets or exists(target_filename)):
            filenames.append([feature_filename, _feature_sub_volumes, target_filename, _target_sub_volumes, subject_id])
        elif raise_if_not_exists:
            for filename in (feature_filename, target_filename):
                if not exists(filename):
                    raise FileNotFoundError(filename)
    # print(filenames)
    return filenames

def generate_filenames(config, name, system_config, skip_targets=False):
    if name not in config:
        load_subject_ids(config)

    if config["generate_filenames"] == "templates":
        return generate_filenames_from_templates(config[name], **config["generate_filenames_kwargs"],
                                                 skip_targets=skip_targets)

def load_subject_ids(config):
    if "subjects_filename" in config:
        subjects = load_json(os.path.join(LOAD_PATH, config["subjects_filename"]))
        for key, value in subjects.items():
            config[key] = value

def load_bias(bias_filename):
    return np.fromfile(os.path.join(LOAD_PATH, bias_filename))

def load_sequence(sequence_name):
    try:
        sequence_class = getattr(k_sequences, sequence_name)
    except AttributeError as error:
        sequence_class = getattr(pytorch_datasets, sequence_name)
    return sequence_class
