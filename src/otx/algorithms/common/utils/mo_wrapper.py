"""Module for defining Model Optimizer (mo) wrapper."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=import-error

import os
import subprocess
import sys
import time


def __mo_check_requirements(framework="onnx"):
    from mo.utils.versions_checker import check_requirements as mo_check_requirements

    return mo_check_requirements(framework)


def check_requirements_with_version(framework=None):
    """A function that checks the compatibility of Model Optimizer with a specific target and framework version.

    :param framework: str, a string representing the framework name. Default is None.
    :return: bool, True if the compatibility check is successful, False otherwise.
    """
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
        print(f"mo_check_requriements returns: {err_code}")
        return False
    return True


def check_requirements(framework="onnx"):
    """A function that checks the compatibility of Model Optimizer with a specific framework version.

    :param framework: str, a string representing the framework name. Default is "onnx".
    :return: bool, True if the compatibility check is successful, False otherwise.
    """
    err_code = __mo_check_requirements(framework)
    if err_code:
        print("mo_check_requriements returns: {err_code}")
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
    """A function that generates IR from a given model using the Model Optimizer.

    :param output_path: str, a string representing the path to save the IR files.
    :param model_path: str, a string representing the path to the original model file.
    :param silent: bool, a flag indicating whether to suppress the output messages. If True,
                         the function will redirect stdout to null device. If False, print messages to the console.
    :param save_xml: bool, a flag indicating whether to save the XML file. Default is True.
    :param mo_kwargs: keyword arguments for the Model Optimizer.
    :return: tuple of int and str, the return code of the Model Optimizer and the output message.
    """
    # parse kwargs for the model optimizer
    mo_args = []
    for key, value in mo_kwargs.items():
        if key not in MO_ARGS:
            return -1, "Not supported argument: {key}"
        if value is not None:
            mo_args.append(f"--{key}={value}")
        else:
            mo_args.append(f"--{key}")

    is_output_dir_provided = False
    for mo_arg in mo_args:
        if mo_arg.startswith("--output_dir"):
            is_output_dir_provided = True
            break
    if not is_output_dir_provided:
        mo_args.append("--output_dir={model_path}")
    print(f"mo-args: {mo_args}")

    if silent:
        # redirect stdout messages from MO to null device
        devnull = open("/dev/null", "w", encoding="utf-8")  # pylint: disable=consider-using-with
        old_stdout = sys.stdout
        sys.stdout = devnull

    # ret = __mo_main_wrapper(mo_args, None)
    # ret = os.system('mo.py ' + ' '.join(mo_args))
    ret = subprocess.run(["mo"] + mo_args, shell=False, check=True).returncode

    if silent:
        # return back stdout
        sys.stdout = old_stdout

    # NOTE: mo returns non zero return code (245) even though it successfully generate IR
    cur_time = time.time()
    model_name = mo_kwargs.get("model_name", "model")
    if not (
        ret == 245
        and not {f"{model_name}.bin", f"{model_name}.xml"} - set(os.listdir(model_path))
        and (
            os.path.getmtime(os.path.join(model_path, f"{model_name}.bin")) - cur_time < 5
            and os.path.getmtime(os.path.join(model_path, f"{model_name}.xml")) - cur_time < 5
        )
    ):
        return ret, "Failed to run the model optimizer to convert a model"

    print("*** Model optimization completed ***")
    # move bin files to workspace

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.rename(
        os.path.join(model_path, model_name + ".bin"),
        os.path.join(output_path, model_name + ".bin"),
    )
    if save_xml:
        os.rename(
            os.path.join(model_path, model_name + ".xml"),
            os.path.join(output_path, model_name + ".xml"),
        )

    return 0, f"Saved outputs into {output_path}"
