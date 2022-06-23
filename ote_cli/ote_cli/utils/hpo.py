"""
Utils for HPO with hpopt
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-branches, too-many-statements
# disbled some pylint refactor related categories
# TODO: refactor code to resolve these things
import builtins
import collections
import importlib
import multiprocessing
import os
import pickle  # nosec
import shutil
import subprocess  # nosec
import sys
import time
from enum import Enum
from inspect import isclass
from math import ceil
from os import path as osp
from pathlib import Path
from typing import Optional

import torch
import yaml
from ote_sdk.configuration.helper import create
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import TaskType
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters, UpdateProgressCallback

from ote_cli.datasets import get_dataset_class
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import generate_label_schema, read_model, save_model_data

try:
    import hpopt
except ImportError:
    print("cannot import hpopt module")
    hpopt = None


def check_hpopt_available():
    """Check whether hpopt is avaiable"""

    if hpopt is None:
        return False
    return True


def _check_hpo_enabled_task(task_type: TaskType):
    """Check whether HPO available task"""
    return task_type in [
        TaskType.CLASSIFICATION,
        TaskType.DETECTION,
        TaskType.SEGMENTATION,
        TaskType.INSTANCE_SEGMENTATION,
        TaskType.ROTATED_DETECTION,
        TaskType.ANOMALY_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION,
        TaskType.ANOMALY_SEGMENTATION,
    ]


def _is_cls_framework_task(task_type: TaskType):
    """Check whether classification framework task"""
    return task_type == TaskType.CLASSIFICATION


def _is_det_framework_task(task_type: TaskType):
    """Check whether detection framework task"""
    return task_type in [
        TaskType.DETECTION,
        TaskType.INSTANCE_SEGMENTATION,
        TaskType.ROTATED_DETECTION,
    ]


def _is_seg_framework_task(task_type: TaskType):
    """Check whether segmentation framework task"""
    return task_type == TaskType.SEGMENTATION


def _is_anomaly_framework_task(task_type: TaskType):
    """Check whether anomaly framework task"""
    return task_type in [
        TaskType.ANOMALY_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION,
        TaskType.ANOMALY_SEGMENTATION,
    ]


def run_hpo(args, environment, dataset, task_type):
    """Update the environment with better hyper-parameters found by HPO"""
    if not check_hpopt_available():
        return None

    if not _check_hpo_enabled_task(task_type):
        print(
            "Currently supported task types are classification, detection, segmentation and anomaly"
            f"{task_type} is not supported yet."
        )
        return None

    dataset_paths = {
        "train_ann_file": args.train_ann_files,
        "train_data_root": args.train_data_roots,
        "val_ann_file": args.val_ann_files,
        "val_data_root": args.val_data_roots,
    }

    hpo_save_path = os.path.abspath(
        os.path.join(os.path.dirname(args.save_model_to), "hpo")
    )
    hpo = HpoManager(
        environment, dataset, dataset_paths, args.hpo_time_ratio, hpo_save_path
    )
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} [HPO] started hyper-parameter optimization"
    )
    hyper_parameters, hpo_weight_path = hpo.run()
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} [HPO] completed hyper-parameter optimization"
    )

    environment.set_hyper_parameters(hyper_parameters)

    task_class = get_impl_class(environment.model_template.entrypoints.base)
    task_class = get_train_wrapper_task(task_class, task_type)

    task = task_class(task_environment=environment)

    hpopt_cfg = _load_hpopt_config(
        osp.join(
            osp.dirname(environment.model_template.model_template_path),
            "hpo_config.yaml",
        )
    )

    if hpopt_cfg.get("resume", False):
        task.set_resume_path_to_config(
            hpo_weight_path
        )  # prepare finetune stage to resume

    if args.load_weights:
        environment.model.configuration.configurable_parameters = hyper_parameters

    return task


def get_cuda_device_list():
    """Retuns the list of avaiable cuda devices"""
    if torch.cuda.is_available():
        hpo_env = os.environ.copy()
        cuda_visible_devices = hpo_env.get("CUDA_VISIBLE_DEVICES", None)

        if cuda_visible_devices is None:
            return list(range(0, torch.cuda.device_count()))

        return [int(i) for i in cuda_visible_devices.split(",")]
    return None


def run_hpo_trainer(
    hp_config,
    hyper_parameters,
    model_template,
    dataset_paths,
    task_type,
):
    """Run each training of each trial with given hyper parameters"""
    if dataset_paths is None:
        raise ValueError("Dataset is not defined.")

    # User argument Parameters are applied to HPO trial
    default_hp = create(model_template.hyper_parameters.data)
    _set_dict_to_parameter_group(default_hp, hyper_parameters)
    hyper_parameters = default_hp

    # set epoch and warm-up stage depending on given epoch
    if _is_cls_framework_task(task_type):
        hyper_parameters.learning_parameters.max_num_epochs = hp_config["iterations"]
    elif _is_det_framework_task(task_type):
        if "bracket" not in hp_config:
            hyper_parameters.learning_parameters.learning_rate_warmup_iters = int(
                hyper_parameters.learning_parameters.learning_rate_warmup_iters
                * hp_config["iterations"]
                / hyper_parameters.learning_parameters.num_iters
            )
        hyper_parameters.learning_parameters.num_iters = hp_config["iterations"]
    elif _is_seg_framework_task(task_type):
        if "bracket" not in hp_config:
            eph_comp = [
                hyper_parameters.learning_parameters.learning_rate_fixed_iters,
                hyper_parameters.learning_parameters.learning_rate_warmup_iters,
                hyper_parameters.learning_parameters.num_iters,
            ]

            eph_comp = list(
                map(lambda x: x * hp_config["iterations"] / sum(eph_comp), eph_comp)
            )

            for val in sorted(
                list(range(len(eph_comp))),
                key=lambda k: eph_comp[k] - int(eph_comp[k]),
                reverse=True,
            )[: hp_config["iterations"] - sum(map(int, eph_comp))]:
                eph_comp[val] += 1

            hyper_parameters.learning_parameters.learning_rate_fixed_iters = int(
                eph_comp[0]
            )
            hyper_parameters.learning_parameters.learning_rate_warmup_iters = int(
                eph_comp[1]
            )
            hyper_parameters.learning_parameters.num_iters = int(eph_comp[2])
        else:
            hyper_parameters.learning_parameters.num_iters = hp_config["iterations"]

    # set hyper-parameters and print them
    HpoManager.set_hyperparameter(hyper_parameters, hp_config["params"])
    print(f"hyper parameter of current trial : {hp_config['params']}")

    dataset_class = get_dataset_class(task_type)
    dataset = dataset_class(
        train_subset={
            "ann_file": dataset_paths.get("train_ann_file", None),
            "data_root": dataset_paths.get("train_data_root", None),
        },
        val_subset={
            "ann_file": dataset_paths.get("val_ann_file", None),
            "data_root": dataset_paths.get("val_data_root", None),
        },
    )

    train_env = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=generate_label_schema(dataset, task_type),
        model_template=model_template,
    )

    # load fixed initial weight
    train_env.model = read_model(
        train_env.get_model_configuration(),
        osp.join(osp.dirname(hp_config["file_path"]), "weights.pth"),
        None,
    )

    train_env.model_template.hpo = {
        "hp_config": hp_config,
        "metric": hp_config["metric"],
    }

    task_class = get_impl_class(train_env.model_template.entrypoints.base)
    hpo_impl_class = get_train_wrapper_task(task_class, task_type)
    task = hpo_impl_class(task_environment=train_env)
    task.prepare_hpo(hp_config)

    dataset = HpoDataset(dataset, hp_config)

    output_model = ModelEntity(
        dataset,
        train_env.get_model_configuration(),
    )

    # make callback to report score to hpopt every epoch
    train_param = TrainParameters(
        False, HpoCallback(hp_config, hp_config["metric"], task), None
    )
    train_param.train_on_empty_model = None

    task.train(dataset=dataset, output_model=output_model, train_parameters=train_param)

    hpopt.finalize_trial(hp_config)

    # remove model weight except best model weight
    best_model_weight = _get_best_model_weight_path(
        _get_hpo_dir(hp_config), str(hp_config["trial_id"]), task_type
    )
    best_model_weight = osp.realpath(best_model_weight)
    for dirpath, _, filenames in os.walk(_get_hpo_trial_workdir(hp_config)):
        for filename in filenames:
            full_name = osp.join(dirpath, filename)
            if (not osp.islink(full_name)) and full_name != best_model_weight:
                os.remove(full_name)


def exec_hpo_trainer(arg_file_name, alloc_gpus):
    """Execute new process to train model for ASHA's trial"""

    gpu_ids = ",".join(str(val) for val in alloc_gpus)
    trainer_file_name = osp.abspath(__file__)
    hpo_env = os.environ.copy()
    if torch.cuda.device_count() > 0:
        hpo_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    hpo_env["PYTHONPATH"] = f"{osp.dirname(__file__)}/../../:" + hpo_env.get(
        "PYTHONPATH", ""
    )

    subprocess.run(
        ["python3", trainer_file_name, arg_file_name],
        shell=False,
        env=hpo_env,
        check=False,
    )
    time.sleep(10)


def get_train_wrapper_task(impl_class, task_type):
    """get task wrapper for the HPO with given task type"""

    class HpoTrainTask(impl_class):
        """wrapper class for the HPO"""

        def __init__(self, task_environment):
            super().__init__(task_environment)
            self._task_type = task_type

        def set_resume_path_to_config(self, resume_path):
            """set path for the resume to the config of the each task framework"""
            if _is_cls_framework_task(self._task_type):
                self._cfg.model.resume = resume_path
                self._cfg.test.save_initial_metric = True
            elif _is_det_framework_task(self._task_type):
                self._config.resume_from = resume_path
            elif _is_seg_framework_task(self._task_type):
                self._config.resume_from = resume_path

        def prepare_hpo(self, hp_config):
            """update config of the each task framework for the HPO"""
            if _is_cls_framework_task(self._task_type):
                # pylint: disable=attribute-defined-outside-init
                self._scratch_space = _get_hpo_trial_workdir(hp_config)
                self._cfg.data.save_dir = self._scratch_space
                self._cfg.model.save_all_chkpts = True
            elif _is_det_framework_task(self._task_type) or _is_seg_framework_task(
                self._task_type
            ):
                self._config.work_dir = _get_hpo_trial_workdir(hp_config)
                self._config.checkpoint_config["max_keep_ckpts"] = (
                    hp_config["iterations"] + 10
                )
                self._config.checkpoint_config["interval"] = 1

    return HpoTrainTask


def _convert_parameter_group_to_dict(parameter_group):
    groups = getattr(parameter_group, "groups", None)
    parameters = getattr(parameter_group, "parameters", None)

    total_arr = []
    for val in [groups, parameters]:
        if val is not None:
            total_arr.extend(val)
    if not total_arr:
        return parameter_group

    ret = {}
    for key in total_arr:
        val = _convert_parameter_group_to_dict(getattr(parameter_group, key))
        if not (isclass(val) or isinstance(val, Enum)):
            ret[key] = val

    return ret


def _set_dict_to_parameter_group(origin_hp, hp_config):
    """
    Set given hyper parameter to hyper parameter in environment
    aligning with "ConfigurableParameters".
    """
    for key, val in hp_config.items():
        if not isinstance(val, dict):
            setattr(origin_hp, key, val)
        else:
            _set_dict_to_parameter_group(getattr(origin_hp, key), val)


def _load_hpopt_config(file_path):
    """load HPOpt config file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            hpopt_cfg = yaml.safe_load(f)
    except FileNotFoundError as err:
        print("hpo_config.yaml file should be in directory where template.yaml is.")
        raise err

    return hpopt_cfg


def _get_best_model_weight_path(hpo_dir: str, trial_num: str, task_type: TaskType):
    """Return best model weight from HPO trial directory"""
    if _is_cls_framework_task(task_type):
        best_weight_path = osp.realpath(osp.join(hpo_dir, str(trial_num), "best.pth"))
    elif _is_det_framework_task(task_type) or _is_seg_framework_task(task_type):
        best_trial_path = osp.join(
            hpo_dir,
            trial_num,
            "checkpoints_round_0",
        )
        for file_name in os.listdir(best_trial_path):
            if "best" in file_name:
                best_weight_path = osp.join(best_trial_path, file_name)
                break
    elif _is_anomaly_framework_task(task_type):
        # TODO need to implement later
        best_weight_path = ""
    else:
        best_weight_path = ""

    return best_weight_path


def _get_hpo_dir(hp_config):
    """Return HPO work directory path from hp_config"""
    return osp.dirname(hp_config["file_path"])


def _get_hpo_trial_workdir(hp_config):
    """Return HPO trial work directory path from hp_config"""
    return osp.join(_get_hpo_dir(hp_config), str(hp_config["trial_id"]))


class HpoCallback(UpdateProgressCallback):
    """Callback class to report score to hpopt"""

    def __init__(self, hp_config, metric, hpo_task):
        super().__init__()
        self.hp_config = hp_config
        self.metric = metric
        self.hpo_task = hpo_task

    def __call__(self, progress: float, score: Optional[float] = None):
        if score is not None:
            current_iters = -1
            if score > 1.0:
                current_iters = int(score)
                score = float(score - current_iters)
            elif score < 0.0:
                current_iters = int(-1.0 * score - 1.0)
                score = 1.0
            print(f"[DEBUG-HPO] score = {score} at iteration {current_iters}")
            if (
                hpopt.report(
                    config=self.hp_config, score=score, current_iters=current_iters
                )
                == hpopt.Status.STOP
            ):
                self.hpo_task.cancel_training()


class HpoManager:
    """Manage overall HPO process"""

    def __init__(
        self, environment, dataset, dataset_paths, expected_time_ratio, hpo_save_path
    ):
        self.environment = environment
        self.dataset_paths = dataset_paths
        self.work_dir = hpo_save_path
        self.deleted_hp = {}

        if environment.model is None:
            impl_class = get_impl_class(environment.model_template.entrypoints.base)
            task = impl_class(task_environment=environment)
            model = ModelEntity(
                dataset,
                environment.get_model_configuration(),
            )
            task.save_model(model)
            save_model_data(model, self.work_dir)
        else:
            save_model_data(environment.model, self.work_dir)

        hpopt_cfg = _load_hpopt_config(
            osp.join(
                osp.dirname(self.environment.model_template.model_template_path),
                "hpo_config.yaml",
            )
        )

        self.algo = hpopt_cfg.get("search_algorithm", "smbo")
        self.metric = hpopt_cfg.get("metric", "mAP")

        num_available_gpus = torch.cuda.device_count()

        self.num_gpus_per_trial = 1

        train_dataset_size = len(dataset.get_subset(Subset.TRAINING))
        val_dataset_size = len(dataset.get_subset(Subset.VALIDATION))

        task_type = self.environment.model_template.task_type

        # make batch size range lower than train set size
        env_hp = self.environment.get_hyper_parameters()
        batch_size_name = None
        if (
            _is_cls_framework_task(task_type)
            or _is_det_framework_task(task_type)
            or _is_seg_framework_task(task_type)
        ):
            batch_size_name = "learning_parameters.batch_size"
        elif _is_anomaly_framework_task(task_type):
            batch_size_name = "dataset.train_batch_size"
        if batch_size_name is not None:
            if batch_size_name in hpopt_cfg["hp_space"]:
                batch_range = hpopt_cfg["hp_space"][batch_size_name]["range"]
                if batch_range[1] > train_dataset_size:
                    batch_range[1] = train_dataset_size

                # If trainset size is lower than min batch size range,
                # fix batch size to trainset size
                if batch_range[0] > batch_range[1]:
                    print(
                        "Train set size is lower than batch size range."
                        "Batch size is fixed to train set size."
                    )
                    del hpopt_cfg["hp_space"][batch_size_name]
                    self.deleted_hp[batch_size_name] = train_dataset_size

        # prepare default hyper parameters
        default_hyper_parameters = {}
        for key in hpopt_cfg["hp_space"].keys():
            splited_key = key.split(".")
            target = env_hp
            for val in splited_key:
                target = getattr(target, val, None)
                if target is None:
                    break
            if target is not None:
                default_hyper_parameters[key] = target

        hpopt_arguments = dict(
            search_alg="bayes_opt" if self.algo == "smbo" else self.algo,
            search_space=HpoManager.generate_hpo_search_space(hpopt_cfg["hp_space"]),
            early_stop=hpopt_cfg.get("early_stop", None),
            resume=self.check_resumable(),
            save_path=self.work_dir,
            max_iterations=hpopt_cfg.get("max_iterations"),
            subset_ratio=hpopt_cfg.get("subset_ratio"),
            num_full_iterations=HpoManager.get_num_full_iterations(self.environment),
            full_dataset_size=train_dataset_size,
            expected_time_ratio=expected_time_ratio,
            non_pure_train_ratio=val_dataset_size
            / (train_dataset_size + val_dataset_size),
            batch_size_name=batch_size_name,
            default_hyper_parameters=default_hyper_parameters,
            metric=self.metric,
            mode=hpopt_cfg.get("mode", "max"),
        )

        if self.algo == "smbo":
            hpopt_arguments["num_init_trials"] = hpopt_cfg.get("num_init_trials")
            hpopt_arguments["num_trials"] = hpopt_cfg.get("num_trials")
        elif self.algo == "asha":
            hpopt_arguments["num_brackets"] = hpopt_cfg.get("num_brackets")
            hpopt_arguments["reduction_factor"] = hpopt_cfg.get("reduction_factor")
            hpopt_arguments["min_iterations"] = hpopt_cfg.get("min_iterations")
            hpopt_arguments["num_trials"] = hpopt_cfg.get("num_trials")
            hpopt_arguments["num_workers"] = (
                num_available_gpus // self.num_gpus_per_trial
            )

            # Prevent each trials from being stopped during warmup stage
            batch_size = default_hyper_parameters.get(batch_size_name)
            if "min_iterations" not in hpopt_cfg and batch_size is not None:
                if _is_cls_framework_task(task_type):
                    with open(
                        osp.join(
                            osp.dirname(
                                self.environment.model_template.model_template_path
                            ),
                            "main_model.yaml",
                        ),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        model_yaml = yaml.safe_load(f)
                    if "warmup" in model_yaml["train"]:
                        hpopt_arguments["min_iterations"] = ceil(
                            model_yaml["train"]["warmup"]
                            * batch_size
                            / train_dataset_size
                        )
                elif _is_det_framework_task(task_type) or _is_seg_framework_task(
                    task_type
                ):
                    hpopt_arguments["min_iterations"] = ceil(
                        env_hp.learning_parameters.learning_rate_warmup_iters
                        / ceil(train_dataset_size / batch_size)
                    )

        HpoManager.remove_empty_keys(hpopt_arguments)

        print(f"[OTE_CLI] [DEBUG-HPO] hpopt args for create hpopt = {hpopt_arguments}")

        self.hpo = hpopt.create(**hpopt_arguments)

    def check_resumable(self):
        """
        Check if HPO could be resumed from the previous result.
        If previous results are found, ask the user if resume or start from scratch.
        """
        resume_flag = False

        hpo_previous_status = hpopt.get_previous_status(self.work_dir)

        if hpo_previous_status in [
            hpopt.Status.PARTIALRESULT,
            hpopt.Status.COMPLETERESULT,
        ]:
            print(f"Previous HPO results are found in {self.work_dir}.")
            retry_count = 0

            while True:
                if hpo_previous_status == hpopt.Status.PARTIALRESULT:
                    user_input = input(
                        "Do you want to resume HPO? [\033[1mY\033[0m/n]: "
                    )
                else:
                    user_input = input(
                        "Do you want to reuse previously found "
                        "hyper-parameters? [\033[1mY\033[0m/n]: "
                    )

                input_len = len(user_input)

                if input_len < 1:
                    # It means that an user accepted the default choice.
                    resume_flag = True
                    break

                if input_len == 1:
                    user_input = user_input.lower()
                    if user_input in ["y", "n"]:
                        if user_input == "y":
                            resume_flag = True
                        break

                retry_count += 1
                if retry_count >= 5:
                    print(
                        "Your inputs are invalid. "
                        "The whole program is terminating..."
                    )
                    sys.exit()

                print("You should type 'y' or 'n'. Nothing means 'y'.")
        return resume_flag

    def run(self):
        """Execute HPO according to configuration"""
        task_type = self.environment.model_template.task_type
        proc_list = []
        gpu_alloc_list = []
        num_workers = 1
        if self.algo == "asha":
            num_workers = torch.cuda.device_count() // self.num_gpus_per_trial

        while True:
            num_active_workers = 0
            for proc, gpu in zip(reversed(proc_list), reversed(gpu_alloc_list)):
                if proc.is_alive():
                    num_active_workers += 1
                else:
                    proc.close()
                    proc_list.remove(proc)
                    gpu_alloc_list.remove(gpu)

            if num_active_workers == num_workers:
                time.sleep(10)

            while num_active_workers < num_workers:
                hp_config = self.hpo.get_next_sample()

                if hp_config is None:
                    # Wait for unfinished trials
                    for proc in proc_list:
                        if proc.is_alive():
                            proc.join()
                    break

                for key, val in self.deleted_hp.items():
                    hp_config["params"][key] = val

                hp_config["metric"] = self.metric

                hpo_work_dir = osp.abspath(_get_hpo_trial_workdir(hp_config))

                # Clear hpo_work_dir
                if osp.exists(hpo_work_dir):
                    shutil.rmtree(hpo_work_dir)
                Path(hpo_work_dir).mkdir(parents=True)

                _kwargs = {
                    "hp_config": hp_config,
                    "hyper_parameters": _convert_parameter_group_to_dict(
                        self.environment.get_hyper_parameters()
                    ),
                    "model_template": self.environment.model_template,
                    "dataset_paths": self.dataset_paths,
                    "task_type": task_type,
                }

                pickle_path = HpoManager.safe_pickle_dump(
                    hpo_work_dir, f"hpo_trial_{hp_config['trial_id']}", _kwargs
                )

                alloc_gpus = self.__alloc_gpus(gpu_alloc_list)

                p = multiprocessing.Process(
                    target=exec_hpo_trainer,
                    args=(
                        pickle_path,
                        alloc_gpus,
                    ),
                )
                proc_list.append(p)
                gpu_alloc_list.append(alloc_gpus)
                p.start()
                num_active_workers += 1

            # All trials are done.
            if num_active_workers == 0:
                break

        best_config = self.hpo.get_best_config()
        for key, val in self.deleted_hp.items():
            best_config[key] = val

        # TODO: is it needed here?
        # # finetune stage resumes hpo trial, so warmup isn't needed
        # if task_type == TaskType.DETECTION:
        #     best_config["learning_parameters.learning_rate_warmup_iters"] = 0
        # if task_type == TaskType.SEGMENTATION:
        #     best_config["learning_parameters.learning_rate_fixed_iters"] = 0
        #     best_config["learning_parameters.learning_rate_warmup_iters"] = 0

        hyper_parameters = self.environment.get_hyper_parameters()
        HpoManager.set_hyperparameter(hyper_parameters, best_config)

        self.hpo.print_results()

        print("Best Hyper-parameters")
        print(best_config)

        # get weight to pass for resume
        hpo_weight_path = _get_best_model_weight_path(
            self.hpo.save_path, str(self.hpo.hpo_status["best_config_id"]), task_type
        )

        return hyper_parameters, hpo_weight_path

    def __alloc_gpus(self, gpu_alloc_list):
        gpu_list = get_cuda_device_list()
        # CPU-mode
        if torch.cuda.device_count() == 0:
            gpu_list = [0]

        alloc_gpus = []
        flat_gpu_alloc_list = sum(gpu_alloc_list, [])
        for idx in gpu_list:
            if idx not in flat_gpu_alloc_list:
                alloc_gpus.append(idx)
                if len(alloc_gpus) == self.num_gpus_per_trial:
                    break

        if len(alloc_gpus) < self.num_gpus_per_trial:
            raise ValueError("No available GPU!!")

        return alloc_gpus

    @staticmethod
    def generate_hpo_search_space(hp_space_dict):
        """Generate search space from user's input"""
        search_space = {}
        for key, val in hp_space_dict.items():
            search_space[key] = hpopt.SearchSpace(val["param_type"], val["range"])
        return search_space

    @staticmethod
    def remove_empty_keys(arg_dict):
        """Remove keys with null values in the arg_dict dictionary"""
        del_candidate = []
        for key, val in arg_dict.items():
            if val is None:
                del_candidate.append(key)
        for val in del_candidate:
            arg_dict.pop(val)
        return del_candidate

    @staticmethod
    def get_num_full_iterations(environment):
        """Get the number of full iterations for the specified environment"""
        num_full_iterations = 0

        task_type = environment.model_template.task_type
        params = environment.get_hyper_parameters()
        if _is_cls_framework_task(task_type):
            learning_parameters = params.learning_parameters
            num_full_iterations = learning_parameters.max_num_epochs
        elif _is_det_framework_task(task_type) or _is_seg_framework_task(task_type):
            learning_parameters = params.learning_parameters
            num_full_iterations = learning_parameters.num_iters
        elif _is_anomaly_framework_task(task_type):
            trainer = params.trainer
            num_full_iterations = trainer.max_epochs

        return num_full_iterations

    @staticmethod
    def safe_pickle_dump(dir_path, file_name, data):
        """Dump a pickle file with minimal file permission"""
        pickle_path = osp.join(dir_path, f"{file_name}.pickle")

        oldmask = os.umask(0o077)
        with open(pickle_path, "wb") as pfile:
            pickle.dump(data, pfile)
        os.umask(oldmask)

        return pickle_path

    @staticmethod
    def set_hyperparameter(origin_hp, hp_config):
        """
        Set given hyper parameter to hyper parameter in environment
        aligning with "ConfigurableParameters".
        """

        for param_key, param_val in hp_config.items():
            param_key = param_key.split(".")

            target = origin_hp
            for val in param_key[:-1]:
                target = getattr(target, val)
            setattr(target, param_key[-1], param_val)


class HpoDataset:
    """
    Wrapper class for DatasetEntity of dataset.
    It's used to make subset during HPO.
    """

    def __init__(self, fullset, config=None, indices=None):
        self.fullset = fullset
        self.indices = indices
        self.subset_ratio = 1 if config is None else config["subset_ratio"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, indx) -> dict:
        return self.fullset[self.indices[indx]]

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.fullset, name)

    def get_subset(self, subset: Subset):
        """
        Get subset according to subset_ratio if trainin dataset is requeseted.
        """

        dataset = self.fullset.get_subset(subset)
        if subset != Subset.TRAINING or self.subset_ratio > 0.99:
            return dataset

        indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(42)
        )
        indices = indices.tolist()  # type: ignore
        indices = indices[: int(len(dataset) * self.subset_ratio)]

        return HpoDataset(dataset, config=None, indices=indices)


class HpoUnpickler(pickle.Unpickler):
    """Safe unpickler for HPO"""

    __safe_builtins = {
        "range",
        "complex",
        "set",
        "frozenset",
        "slice",
        "dict",
    }

    __safe_collections = {"OrderedDict", "defaultdict"}

    __allowed_classes = {
        "datetime": {
            "datetime",
            "timezone",
            "timedelta",
        },
        "networkx.classes.graph": {
            "Graph",
        },
        "networkx.classes.multidigraph": {
            "MultiDiGraph",
        },
        "ote_sdk.configuration.enums.config_element_type": {
            "ConfigElementType",
        },
        "ote_sdk.entities.label_schema": {
            "LabelTree",
            "LabelGroup",
            "LabelGroupType",
            "LabelSchemaEntity",
            "LabelGraph",
        },
        "ote_sdk.entities.id": {
            "ID",
        },
        "ote_sdk.entities.label": {
            "Domain",
            "LabelEntity",
        },
        "ote_sdk.entities.color": {
            "Color",
        },
        "ote_sdk.entities.model_template": {
            "ModelTemplate",
            "TaskFamily",
            "TaskType",
            "InstantiationType",
            "TargetDevice",
            "DatasetRequirements",
            "HyperParameterData",
            "EntryPoints",
            "ExportableCodePaths",
        },
    }

    def find_class(self, module_name, class_name):
        # Only allow safe classes from builtins.
        if (
            module_name in ["builtins", "__builtin__"]
            and class_name in self.__safe_builtins
        ):
            return getattr(builtins, class_name)
        if module_name == "collections" and class_name in self.__safe_collections:
            return getattr(collections, class_name)
        for allowed_module_name, val in self.__allowed_classes.items():
            if module_name == allowed_module_name and class_name in val:
                module = importlib.import_module(module_name)
                return getattr(module, class_name)

        # Forbid everything else.
        raise pickle.UnpicklingError(
            f"global '{module_name}.{class_name}' is forbidden"
        )


def main():
    """Run run_hpo_trainer with a pickle file"""
    hp_config = None
    sys.path[0] = ""  # to prevent importing nncf from this directory

    try:
        with open(sys.argv[1], "rb") as pfile:
            kwargs = HpoUnpickler(pfile).load()
            hp_config = kwargs["hp_config"]

            run_hpo_trainer(**kwargs)
    except RuntimeError as err:
        if str(err).startswith("CUDA out of memory"):
            hpopt.reportOOM(hp_config)


if __name__ == "__main__":
    main()
