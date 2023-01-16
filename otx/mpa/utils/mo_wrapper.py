# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import subprocess

# import os
import sys


def __mo_check_requirements(framework="onnx"):
    from mo.utils.versions_checker import check_requirements as mo_check_requirements

    return mo_check_requirements(framework)


def check_requirements_with_version(target, framework=None):
    from mo.utils.version import get_version as mo_get_version

    mo_version = mo_get_version()
    print("mo vesion =", mo_version)
    # # TODO: model optimizer version in OpenVINO 2019.3.3 is 2019.3.0
    # # it will be changed before official release to meet with requirement of VAS
    # required_mo_version = '2019.3.0'
    # if target.lower() == 'fr':
    #     required_mo_version = '2019.1.1'
    # if not mo_version.startswith(required_mo_version):
    #     print('current MO version: {} but required version: {}'.format(mo_version, required_mo_version))
    #     return False
    err_code = __mo_check_requirements(framework)
    if err_code:
        print("mo_check_requriements returns: {}".format(err_code))
        return False
    return True


def check_requirements(framework="onnx"):
    err_code = __mo_check_requirements(framework)
    if err_code:
        print("mo_check_requriements returns: {}".format(err_code))
        return False
    return True


def __mo_main_wrapper(argv, framework=None):
    """MO main wrapper.

    See: openvino_2019.1.094/deployment_tools/model_optimizer/mo/main.py
    """
    # substitute the value of sys.argv with a proper cli arguments to call the
    # OpenVINO model optimizer's main function
    from mo.main import main as mo_main

    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + argv

    # run mo
    from mo.utils.cli_parser import get_all_cli_parser

    ret = mo_main(get_all_cli_parser(), framework)

    # restore sys.argv
    sys.argv = old_argv

    return ret


MO_LOG_LEVELS = ["CRITICAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"]

MO_ARGS = [
    "input_model",
    "input_shape",
    "input",
    "mean_values",
    "scale",
    "model_name",
    "log_level",
    "compress_to_fp16",
    "scale_values",
    "disable_fusing",
    "transformations_config",
    "reverse_input_channels",
    "output_dir",
]


def generate_ir(output_path, model_path, silent, save_xml=True, **mo_kwargs):
    # parse kwargs for the model optimizer
    mo_args = []
    for key, value in mo_kwargs.items():
        if key not in MO_ARGS:
            return -1, "Not supported argument: {}".format(key)
        if value is not None:
            mo_args.append("--{}={}".format(key, value))
        else:
            mo_args.append("--{}".format(key))

    is_output_dir_provided = False
    for mo_arg in mo_args:
        if mo_arg.startswith("--output_dir"):
            is_output_dir_provided = True
            break
    if not is_output_dir_provided:
        mo_args.append("--output_dir={}".format(model_path))
    print("mo-args: {}".format(mo_args))

    if silent:
        # redirect stdout messages from MO to null device
        devnull = open("/dev/null", "w")
        old_stdout = sys.stdout
        sys.stdout = devnull

    # ret = __mo_main_wrapper(mo_args, None)
    # ret = os.system('mo.py ' + ' '.join(mo_args))
    ret = subprocess.run(["mo"] + mo_args, shell=False).returncode
    if ret != 0:
        err_msg = "Failed to run the model optimizer to convert a model"
        return ret, err_msg

    if silent:
        # return back stdout
        sys.stdout = old_stdout

    print("*** Model optimization completed ***")
    # move bin files to workspace
    import os

    bin_filename = mo_kwargs["model_name"]
    output_filename = bin_filename
    # while os.path.exists(os.path.join(output_path, output_filename + '.bin')):
    #    output_filename = output_filename + "_"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.rename(os.path.join(model_path, bin_filename + ".bin"), os.path.join(output_path, output_filename + ".bin"))
    if save_xml:
        os.rename(os.path.join(model_path, bin_filename + ".xml"), os.path.join(output_path, output_filename + ".xml"))

    return ret, "Saved outputs into {}".format(output_path)
