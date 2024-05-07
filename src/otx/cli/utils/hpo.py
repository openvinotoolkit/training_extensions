"""Utils for HPO with hpopt."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
import re
import shutil
import time
from copy import deepcopy
from enum import Enum
from functools import partial
from inspect import isclass
from math import floor
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import yaml

from otx.algorithms.common.utils import is_xpu_available
from otx.api.configuration.helper import create
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters, UpdateProgressCallback
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_model, save_model_data
from otx.core.data.adapter import get_dataset_adapter
from otx.hpo import HyperBand, TrialStatus, run_hpo_loop
from otx.hpo.hpo_base import HpoBase
from otx.utils.logger import get_logger

logger = get_logger()


def _check_hpo_enabled_task(task_type):
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


class TaskManager:
    """Task utility class to give common interface from different task.

    Args:
        task_type (TaskType): otx task type
    """

    def __init__(self, task_type: TaskType):
        self._task_type = task_type

    @property
    def task_type(self):
        """Task_type property."""
        return self._task_type

    def is_mmcv_framework_task(self) -> bool:
        """Check task is run on mmcv.

        Returns:
            bool: whether task is run on mmcv
        """
        return self.is_cls_framework_task() or self.is_det_framework_task() or self.is_seg_framework_task()

    def is_cls_framework_task(self) -> bool:
        """Check that task is run on mmcls framework.

        Returns:
            bool: whether task is run on mmcls
        """
        return self._task_type == TaskType.CLASSIFICATION

    def is_det_framework_task(self) -> bool:
        """Check that task is one of a task run on mmdet framework.

        Returns:
            bool: whether task is run on mmdet
        """
        return self._task_type in [
            TaskType.DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.ROTATED_DETECTION,
        ]

    def is_seg_framework_task(self) -> bool:
        """Check that task is run on mmseg framework.

        Returns:
            bool: whether tasks is run on mmseg
        """
        return self._task_type == TaskType.SEGMENTATION

    def is_anomaly_framework_task(self) -> bool:
        """Check taht task is run on anomalib.

        Returns:
            bool: whether task is run on anomalib
        """
        return self._task_type in [
            TaskType.ANOMALY_CLASSIFICATION,
            TaskType.ANOMALY_DETECTION,
            TaskType.ANOMALY_SEGMENTATION,
        ]

    def get_batch_size_name(self) -> str:
        """Give an proper batch size name depending on framework.

        Returns:
            str: batch size name
        """
        if self.is_mmcv_framework_task():
            batch_size_name = "learning_parameters.batch_size"
        elif self.is_anomaly_framework_task():
            batch_size_name = "learning_parameters.train_batch_size"
        else:
            raise RuntimeError(f"There is no information about {self._task_type} batch size name")

        return batch_size_name

    def get_epoch_name(self) -> str:
        """Give an proper epoch name depending on framework.

        Returns:
            str: epoch name
        """
        if self.is_mmcv_framework_task():
            epoch_name = "num_iters"
        elif self.is_anomaly_framework_task():
            epoch_name = "max_epochs"
        else:
            raise RuntimeError(f"There is no information about {self._task_type} epoch name")

        return epoch_name

    def copy_weight(self, src: Union[str, Path], det: Union[str, Path]):
        """Copy all model weights from work directory.

        Args:
            src (Union[str, Path]): path where model weights are saved
            det (Union[str, Path]): path to save model weights
        """
        src = Path(src)
        det = Path(det)
        if self.is_mmcv_framework_task():
            for weight_candidate in src.rglob("*epoch*.pth"):
                if not (weight_candidate.is_symlink() or (det / weight_candidate.name).exists()):
                    shutil.copy(weight_candidate, det)
        # TODO need to implement after anomaly task supports resume

    def get_latest_weight(self, workdir: Union[str, Path]) -> Optional[str]:
        """Get latest model weight from all weights.

        Args:
            workdir (Union[str, Path]): path where model weights are saved

        Returns:
            Optional[str]: latest model weight path. If not found, than return None value.
        """
        latest_weight = None
        workdir = Path(workdir)
        if self.is_mmcv_framework_task():
            pattern = re.compile(r"(\d+)\.pth")
            current_latest_epoch = -1
            latest_weight = None

            for weight_name in workdir.rglob("epoch_*.pth"):
                ret = pattern.search(str(weight_name))
                if ret is not None:
                    epoch = int(ret.group(1))
                    if current_latest_epoch < epoch:
                        current_latest_epoch = epoch
                        latest_weight = str(weight_name)
        # TODO need to implement after anomaly task supports resume

        return latest_weight


class TaskEnvironmentManager:
    """OTX environment utility class to set or get a value from environment class.

    Args:
        environment (TaskEnvironment): OTX task environment
    """

    def __init__(self, environment: TaskEnvironment):
        self._environment = environment
        self.task = TaskManager(environment.model_template.task_type)

    @property
    def environment(self):
        """Environment property."""
        return self._environment

    def get_task(self) -> TaskType:
        """Get task type of environment.

        Returns:
            TaskType: task type
        """
        return self._environment.model_template.task_type

    def get_model_template(self):
        """Get model template."""
        return self._environment.model_template

    def get_model_template_path(self) -> str:
        """Get model template path.

        Returns:
            str: path of model template
        """
        return self._environment.model_template.model_template_path

    def set_hyper_parameter_using_str_key(self, hyper_parameter: Dict[str, Any]):
        """Set hyper parameter to environment using string key hyper_parameter.

        Set hyper parameter to environment. Argument `hyper_parameter` is a dictionary which has string key.
        For example, hyper_parameter has a key "a.b.c", then value is set at env_hp.a.b.c.

        Args:
            hyper_parameter (Dict[str, Any]): hyper parameter to set which has a string format
        """
        env_hp = self._environment.get_hyper_parameters()  # type: ignore

        for param_key, param_val in hyper_parameter.items():
            splited_param_key = param_key.split(".")

            target = env_hp
            for val in splited_param_key[:-1]:
                target = getattr(target, val)
            setattr(target, splited_param_key[-1], param_val)

    def get_dict_type_hyper_parameter(self) -> Dict[str, Any]:
        """Get dictionary type hyper parmaeter of environment.

        Returns:
            Dict[str, Any]: dictionary type hyper parameter of environment
        """
        learning_parameters = self._environment.get_hyper_parameters().learning_parameters  # type: ignore
        learning_parameters = self._convert_parameter_group_to_dict(learning_parameters)
        hyper_parameter = {f"learning_parameters.{key}": val for key, val in learning_parameters.items()}
        return hyper_parameter

    def _convert_parameter_group_to_dict(self, parameter_group) -> Dict[str, Any]:
        """Convert parameter group to dictionary.

        Args:
            parameter_group : parameter gruop

        Returns:
            Dict[str, Any]: parameter group converted to dictionary
        """
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
            val = self._convert_parameter_group_to_dict(getattr(parameter_group, key))
            if not (isclass(val) or isinstance(val, Enum)):
                ret[key] = val

        return ret

    def get_max_epoch(self) -> int:
        """Get max epoch from environment.

        Returns:
            int: max epoch of environment
        """
        return getattr(
            self._environment.get_hyper_parameters().learning_parameters, self.task.get_epoch_name()  # type: ignore
        )

    def save_initial_weight(self, save_path: Union[Path, str]) -> bool:
        """Save an initial model weight.

        Args:
            save_path (Union[str, Path]): path to save initial model weight

        Returns:
            bool: whether model weight is saved successfully
        """
        save_path = Path(save_path)
        dir_path = save_path.parent
        if self._environment.model is None:
            # if task isn't anomaly, then save model weight during first trial
            if self.task.is_anomaly_framework_task():
                task = self.get_train_task()
                model = self.get_new_model_entity()
                task.save_model(model)
                save_model_data(model, str(dir_path))
                (dir_path / "weights.pth").rename(save_path)
                return True
        else:
            save_model_data(self._environment.model, str(dir_path))
            (dir_path / "weights.pth").rename(save_path)
            return True
        return False

    def get_train_task(self):
        """Get OTX train task instance.

        Returns:
           OTX task: OTX train task instance
        """
        impl_class = get_impl_class(self._environment.model_template.entrypoints.base)
        return impl_class(task_environment=self._environment)

    def get_batch_size_name(self) -> str:
        """Get proper batch size name depending on task.

        Returns:
            str: batch size name
        """
        return self.task.get_batch_size_name()

    def load_model_weight(self, model_weight_path: str, dataset: DatasetEntity):
        """Set model weight on environment to load the weight during training.

        Args:
            model_weight_path (str): model weight to load during training
            dataset (DatasetEntity): dataset for training a model
        """
        self._environment.model = read_model(self._environment.get_model_configuration(), model_weight_path, dataset)

    def resume_model_weight(self, model_weight_path: str, dataset: DatasetEntity):
        """Set model weight on environment to resume the weight during training.

        Args:
            model_weight_path (str): model weight to resume during training
            dataset (DatasetEntity): dataset for training a model
        """
        self.load_model_weight(model_weight_path, dataset)
        self._environment.model.model_adapters["resume"] = True  # type: ignore

    def get_new_model_entity(self, dataset=None) -> ModelEntity:
        """Get new model entity using environment.

        Args:
            dataset (Optional[DatasetEntity]): OTX dataset

        Returns:
            ModelEntity: new model entity
        """
        return ModelEntity(
            dataset,
            self._environment.get_model_configuration(),
        )

    def set_epoch(self, epoch: int):
        """Set epoch on environment.

        Args:
            epoch (int): epoch to set
        """
        hyper_parameter = {f"learning_parameters.{self.task.get_epoch_name()}": epoch}
        self.set_hyper_parameter_using_str_key(hyper_parameter)


class HpoRunner:
    """Class which is in charge of preparing and running HPO.

    Args:
        environment (TaskEnvironment): OTX environment
        train_dataset_size (int): train dataset size
        val_dataset_size (int): validation dataset size
        hpo_workdir (Union[str, Path]): work directory for HPO
        hpo_time_ratio (int, optional): time ratio to use for HPO compared to training time. Defaults to 4.
        progress_updater_callback (Optional[Callable[[Union[int, float]], None]]): callback to update progress
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        environment: TaskEnvironment,
        train_dataset_size: int,
        val_dataset_size: int,
        hpo_workdir: Union[str, Path],
        hpo_time_ratio: int = 4,
        progress_updater_callback: Optional[Callable[[Union[int, float]], None]] = None,
    ):
        if train_dataset_size <= 0:
            raise ValueError(f"train_dataset_size should be bigger than 0. Your value is {train_dataset_size}")
        if val_dataset_size <= 0:
            raise ValueError(f"val_dataset_size should be bigger than 0. Your value is {val_dataset_size}")
        if hpo_time_ratio < 1:
            raise ValueError(f"hpo_time_ratio shouldn't be smaller than 1. Your value is {hpo_time_ratio}")

        self._environment = TaskEnvironmentManager(environment)
        self._hpo_workdir: Path = Path(hpo_workdir)
        self._hpo_time_ratio = hpo_time_ratio
        self._hpo_config: Dict = self._set_hpo_config()
        self._train_dataset_size = train_dataset_size
        self._val_dataset_size = val_dataset_size
        self._fixed_hp: Dict[str, Any] = {}
        self._initial_weight_name = "initial_weight.pth"
        self._progress_updater_callback = progress_updater_callback

        self._align_batch_size_search_space_to_dataset_size()

    def _set_hpo_config(self):
        hpo_config_path = Path(self._environment.get_model_template_path()).parent / "hpo_config.yaml"
        with hpo_config_path.open("r") as f:
            hpopt_cfg = yaml.safe_load(f)

        return hpopt_cfg

    def _align_batch_size_search_space_to_dataset_size(self):
        batch_size_name = self._environment.get_batch_size_name()

        if batch_size_name in self._hpo_config["hp_space"]:
            if "range" in self._hpo_config["hp_space"][batch_size_name]:
                max_val = self._hpo_config["hp_space"][batch_size_name]["range"][1]
                min_val = self._hpo_config["hp_space"][batch_size_name]["range"][0]
                step = 1
                if self._hpo_config["hp_space"][batch_size_name]["param_type"] in ["quniform", "qloguniform"]:
                    step = self._hpo_config["hp_space"][batch_size_name]["range"][2]
                if max_val > self._train_dataset_size:
                    max_val = self._train_dataset_size
                    self._hpo_config["hp_space"][batch_size_name]["range"][1] = max_val
            else:
                max_val = self._hpo_config["hp_space"][batch_size_name]["max"]
                min_val = self._hpo_config["hp_space"][batch_size_name]["min"]
                step = self._hpo_config["hp_space"][batch_size_name].get("step", 1)

                if max_val > self._train_dataset_size:
                    max_val = self._train_dataset_size
                    self._hpo_config["hp_space"][batch_size_name]["max"] = max_val

            # If trainset size is lower than min batch size range,
            # fix batch size to trainset size
            reason_to_fix_bs = ""
            if min_val >= max_val:
                reason_to_fix_bs = "Train set size is equal or lower than batch size range."
            elif max_val - min_val < step:
                reason_to_fix_bs = "Difference between min and train set size is lesser than step."
            if reason_to_fix_bs:
                logger.info(f"{reason_to_fix_bs} Batch size is fixed to train set size.")
                del self._hpo_config["hp_space"][batch_size_name]
                self._fixed_hp[batch_size_name] = self._train_dataset_size
                self._environment.set_hyper_parameter_using_str_key(self._fixed_hp)

    def run_hpo(self, train_func: Callable, data_roots: Dict[str, Dict]) -> Union[Dict[str, Any], None]:
        """Run HPO and provides optimized hyper parameters.

        Args:
            train_func (Callable): training model function
            data_roots (Dict[str, Dict]): dataset path of each dataset type

        Returns:
            Union[Dict[str, Any], None]: Optimized hyper parameters. If there is no best hyper parameter, return None.
        """
        self._environment.save_initial_weight(self._get_initial_model_weight_path())
        hpo_algo = self._get_hpo_algo()

        if self._progress_updater_callback is not None:
            progress_updater_thread = Thread(target=self._update_hpo_progress, args=[hpo_algo], daemon=True)
            progress_updater_thread.start()

        remove_unused_model_weight = Thread(
            target=self._remove_unused_weight, args=[hpo_algo, self._hpo_workdir], daemon=True
        )
        remove_unused_model_weight.start()

        if torch.cuda.is_available():
            resource_type = "gpu"
        elif is_xpu_available():
            resource_type = "xpu"
        else:
            resource_type = "cpu"

        run_hpo_loop(
            hpo_algo,
            partial(
                train_func,
                model_template=self._environment.get_model_template(),
                data_roots=data_roots,
                task_type=self._environment.get_task(),
                hpo_workdir=self._hpo_workdir,
                initial_weight_name=self._initial_weight_name,
                metric=self._hpo_config["metric"],
            ),
            resource_type,  # type: ignore
        )
        best_config = hpo_algo.get_best_config()
        if best_config is not None:
            self._restore_fixed_hp(best_config["config"])
        hpo_algo.print_result()

        return best_config

    def _restore_fixed_hp(self, hyper_parameter: Dict[str, Any]):
        for key, val in self._fixed_hp.items():
            hyper_parameter[key] = val

    def _get_hpo_algo(self):
        hpo_algo_type = self._hpo_config.get("search_algorithm", "asha")

        if hpo_algo_type == "asha":
            hpo_algo = self._prepare_asha()
        elif hpo_algo_type == "smbo":
            hpo_algo = self._prepare_smbo()
        else:
            raise ValueError(f"Supported HPO algorithms are asha and smbo. your value is {hpo_algo_type}.")

        return hpo_algo

    def _prepare_asha(self):
        if is_xpu_available():
            asynchronous_sha = torch.xpu.device_count() != 1
        else:
            asynchronous_sha = torch.cuda.device_count() != 1

        args = {
            "search_space": self._hpo_config["hp_space"],
            "save_path": str(self._hpo_workdir),
            "maximum_resource": self._hpo_config.get("maximum_resource"),
            "minimum_resource": self._hpo_config.get("minimum_resource"),
            "mode": self._hpo_config.get("mode", "max"),
            "num_workers": 1,
            "num_full_iterations": self._environment.get_max_epoch(),
            "full_dataset_size": self._train_dataset_size,
            "non_pure_train_ratio": self._val_dataset_size / (self._train_dataset_size + self._val_dataset_size),
            "metric": self._hpo_config.get("metric", "mAP"),
            "expected_time_ratio": self._hpo_time_ratio,
            "prior_hyper_parameters": self._get_default_hyper_parameters(),
            "asynchronous_bracket": True,
            "asynchronous_sha": asynchronous_sha,
        }

        logger.debug(f"ASHA args = {args}")

        return HyperBand(**args)

    def _prepare_smbo(self):
        raise NotImplementedError

    def _get_default_hyper_parameters(self):
        default_hyper_parameters = {}
        hp_from_env = self._environment.get_dict_type_hyper_parameter()

        for key, val in hp_from_env.items():
            if key in self._hpo_config["hp_space"]:
                default_hyper_parameters[key] = val

        if not default_hyper_parameters:
            return None
        return default_hyper_parameters

    def _get_initial_model_weight_path(self):
        return self._hpo_workdir / self._initial_weight_name

    def _update_hpo_progress(self, hpo_algo: HpoBase):
        """Function for a thread to report a HPO progress regularly.

        Args:
            hpo_algo (HpoBase): HPO algorithm class
        """

        while True:
            if hpo_algo.is_done():
                break
            self._progress_updater_callback(hpo_algo.get_progress() * 100)
            time.sleep(1)

    def _remove_unused_weight(self, hpo_algo: HpoBase, hpo_work_dir: Path):
        """Function for a thread to report a HPO progress regularly.

        Args:
            hpo_algo (HpoBase): HPO algorithm instance.
            hpo_work_dir (Path): HPO work directory.
        """

        while not hpo_algo.is_done():
            finished_trials = hpo_algo.get_inferior_trials()
            for trial in finished_trials:
                dir_to_remove = hpo_work_dir / "weight" / str(trial.id)
                if dir_to_remove.exists():
                    shutil.rmtree(dir_to_remove)
            time.sleep(1)


def run_hpo(
    hpo_time_ratio: int,
    output: Path,
    environment: TaskEnvironment,
    dataset: DatasetEntity,
    data_roots: Dict[str, Dict],
    progress_updater_callback: Optional[Callable[[Union[int, float]], None]] = None,
) -> Optional[TaskEnvironment]:
    """Run HPO and load optimized hyper parameter and best HPO model weight.

    Args:
        hpo_time_ratio(int): expected ratio of total time to run HPO to time taken for full fine-tuning
        output(Path): directory where HPO output is saved
        environment (TaskEnvironment): otx task environment
        dataset (DatasetEntity): dataset to use for training
        data_roots (Dict[str, Dict]): dataset path of each dataset type
        progress_updater_callback (Optional[Callable[[Union[int, float]], None]]): callback to update progress
    """
    task_type = environment.model_template.task_type
    if not _check_hpo_enabled_task(task_type):
        logger.warning(
            "Currently supported task types are classification, detection, segmentation and anomaly"
            f"{task_type} is not supported yet."
        )
        return environment

    if "TORCHELASTIC_RUN_ID" in os.environ:
        logger.warning("OTX is trained by torchrun. HPO isn't available.")
        return environment

    hpo_save_path = (output / "hpo").absolute()
    hpo_runner = HpoRunner(
        environment,
        len(dataset.get_subset(Subset.TRAINING)),
        len(dataset.get_subset(Subset.VALIDATION)),
        hpo_save_path,
        hpo_time_ratio,
        progress_updater_callback,
    )

    logger.info("started hyper-parameter optimization")
    best_config = hpo_runner.run_hpo(run_trial, data_roots)
    logger.info("completed hyper-parameter optimization")

    env_manager = TaskEnvironmentManager(environment)
    best_hpo_weight = None

    if best_config is not None:
        env_manager.set_hyper_parameter_using_str_key(best_config["config"])
        best_hpo_weight = get_best_hpo_weight(hpo_save_path, best_config["id"])
        if best_hpo_weight is None:
            logger.warning("Can not find the best HPO weight. Best HPO wegiht won't be used.")
        else:
            logger.debug(f"{best_hpo_weight} will be loaded as best HPO weight")
            env_manager.load_model_weight(best_hpo_weight, dataset)

    _remove_unused_model_weights(hpo_save_path, best_hpo_weight)
    return env_manager.environment


def _remove_unused_model_weights(hpo_save_path: Path, best_hpo_weight: Optional[str] = None):
    for weight in hpo_save_path.rglob("*.pth"):
        if best_hpo_weight is not None and str(weight) == best_hpo_weight:
            continue
        weight.unlink()


def get_best_hpo_weight(hpo_dir: Union[str, Path], trial_id: Union[str, Path]) -> Optional[str]:
    """Get best model weight path of the HPO trial.

    Args:
        hpo_dir (Union[str, Path]): HPO work directory path
        trial_id (Union[str, Path]): trial id

    Returns:
        Optional[str]: best HPO model weight
    """
    hpo_dir = Path(hpo_dir)
    trial_output_files = list(hpo_dir.rglob(f"{trial_id}.json"))
    if not trial_output_files:
        return None
    trial_output_file = trial_output_files[0]

    with trial_output_file.open("r") as f:
        trial_output = json.load(f)

    best_epochs = []
    best_score = None
    for eph, score in trial_output["score"].items():
        if best_score is None:
            best_score = score
            best_epochs.append(eph)
        elif best_score < score:
            best_score = score
            best_epochs = [eph]
        elif best_score == score:
            best_epochs.append(eph)

    best_weight = None
    for best_epoch in best_epochs:
        best_weight_path = list(hpo_dir.glob(f"weight/{trial_id}/*epoch*{best_epoch}*"))
        if best_weight_path:
            best_weight = str(best_weight_path[0])

    return best_weight


class Trainer:
    """Class which prepares and trains a model given hyper parameters.

    Args:
        hp_config (Dict[str, Any]): hyper parameter to use on training
        report_func (Callable): function to report score
        model_template: model template
        data_roots (Dict[str, Dict]): dataset path of each dataset type
        task_type (TaskType): OTX task type
        hpo_workdir (Union[str, Path]): work directory for HPO
        initial_weight_name (str): initial model weight name for each trials to load
        metric (str): metric name
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes

    def __init__(
        self,
        hp_config: Dict[str, Any],
        report_func: Callable,
        model_template,
        data_roots: Dict[str, Dict],
        task_type: TaskType,
        hpo_workdir: Union[str, Path],
        initial_weight_name: str,
        metric: str,
    ):
        self._hp_config = hp_config
        self._report_func = report_func
        self._model_template = model_template
        self._data_roots = data_roots
        self._task = TaskManager(task_type)
        self._hpo_workdir: Path = Path(hpo_workdir)
        self._initial_weight_name = initial_weight_name
        self._metric = metric
        self._epoch = floor(self._hp_config["configuration"]["iterations"])
        del self._hp_config["configuration"]["iterations"]

    def run(self):
        """Run each training of each trial with given hyper parameters."""
        hyper_parameters = self._prepare_hyper_parameter()
        dataset_adapter = self._prepare_dataset_adapter()

        dataset = dataset_adapter.get_otx_dataset()
        dataset = HpoDataset(dataset, self._hp_config)

        label_schema = dataset_adapter.get_label_schema()

        environment = self._prepare_environment(hyper_parameters, label_schema)
        self._set_hyper_parameter(environment)

        need_to_save_initial_weight = False
        resume_weight_path = self._get_resume_weight_path()
        if resume_weight_path is not None:
            ret = re.search(r"(\d+)\.pth", resume_weight_path)
            if ret is not None:
                resume_epoch = int(ret.group(1))
                if self._epoch <= resume_epoch:  # given epoch is already done
                    self._report_func(0, 0, done=True)
                    return
            environment.resume_model_weight(resume_weight_path, dataset)
        else:
            initial_weight = self._load_fixed_initial_weight()
            if initial_weight is not None:
                environment.load_model_weight(str(initial_weight), dataset)
            else:
                need_to_save_initial_weight = True

        task = environment.get_train_task()
        if need_to_save_initial_weight:
            self._add_initial_weight_saving_hook(task)

        output_model = environment.get_new_model_entity(dataset)
        score_report_callback = self._prepare_score_report_callback(task)
        task.train(dataset=dataset, output_model=output_model, train_parameters=score_report_callback)
        self._finalize_trial(task)

    def _prepare_hyper_parameter(self):
        return create(self._model_template.hyper_parameters.data)

    def _prepare_dataset_adapter(self):
        dataset_adapter = get_dataset_adapter(
            self._task.task_type,
            self._model_template.hyper_parameters.parameter_overrides["algo_backend"]["train_type"]["default_value"],
            train_data_roots=self._data_roots["train_subset"]["data_roots"],
            val_data_roots=self._data_roots["val_subset"]["data_roots"] if "val_subset" in self._data_roots else None,
            unlabeled_data_roots=self._data_roots["unlabeled_subset"]["data_roots"]
            if "unlabeled_subset" in self._data_roots
            else None,
        )

        return dataset_adapter

    def _set_hyper_parameter(self, environment: TaskEnvironmentManager):
        environment.set_hyper_parameter_using_str_key(self._hp_config["configuration"])
        if self._task.is_mmcv_framework_task():
            environment.set_hyper_parameter_using_str_key({"learning_parameters.auto_decrease_batch_size": "None"})
            environment.set_hyper_parameter_using_str_key({"learning_parameters.auto_adapt_batch_size": "None"})
        environment.set_epoch(self._epoch)

    def _prepare_environment(self, hyper_parameters, label_schema):
        enviroment = TaskEnvironment(
            model=None,
            hyper_parameters=hyper_parameters,
            label_schema=label_schema,
            model_template=self._model_template,
        )

        return TaskEnvironmentManager(enviroment)

    def _get_resume_weight_path(self):
        trial_work_dir = self._get_weight_dir_path()
        if not trial_work_dir.exists():
            return None
        return self._task.get_latest_weight(trial_work_dir)

    def _load_fixed_initial_weight(self):
        initial_weight_path = self._get_initial_weight_path()
        if initial_weight_path.exists():
            return initial_weight_path
        return None

    def _add_initial_weight_saving_hook(self, task):
        initial_weight_path = self._get_initial_weight_path()
        task.update_override_configurations(
            {
                "custom_hooks": [
                    dict(
                        type="SaveInitialWeightHook",
                        save_path=initial_weight_path.parent,
                        file_name=initial_weight_path.name,
                    )
                ]
            }
        )

    def _prepare_score_report_callback(self, task) -> TrainParameters:
        return TrainParameters(False, HpoCallback(self._report_func, self._metric, self._epoch, task))

    def _get_initial_weight_path(self) -> Path:
        return self._hpo_workdir / self._initial_weight_name

    def _finalize_trial(self, task):
        self._report_func(0, 0, done=True)

        weight_dir_path = self._get_weight_dir_path()
        weight_dir_path.mkdir(parents=True, exist_ok=True)
        self._task.copy_weight(task.project_path, weight_dir_path)
        necessary_weights = [
            self._task.get_latest_weight(weight_dir_path),
            get_best_hpo_weight(self._hpo_workdir, self._hp_config["id"]),
        ]
        while None in necessary_weights:
            necessary_weights.remove(None)
        for each_model_weight in weight_dir_path.iterdir():
            for necessary_weight in necessary_weights:
                if each_model_weight.samefile(necessary_weight):
                    break
            else:
                each_model_weight.unlink()

    def _get_weight_dir_path(self) -> Path:
        return self._hpo_workdir / "weight" / self._hp_config["id"]


def run_trial(
    hp_config: Dict[str, Any],
    report_func: Callable,
    model_template,
    data_roots: Dict[str, Dict],
    task_type: TaskType,
    hpo_workdir: Union[str, Path],
    initial_weight_name: str,
    metric: str,
):
    """Function to train a model given hyper parameters.

    Args:
        hp_config (Dict[str, Any]): hyper parameter to use on training
        report_func (Callable): function to report score
        model_template: model template
        data_roots (Dict[str, Dict]): dataset path of each dataset type
        task_type (TaskType): OTX task type
        hpo_workdir (Union[str, Path]): work directory for HPO
        initial_weight_name (str): initial model weight name for each trials to load
        metric (str): metric name
    """
    # pylint: disable=too-many-arguments
    trainer = Trainer(
        hp_config, report_func, model_template, data_roots, task_type, hpo_workdir, initial_weight_name, metric
    )
    trainer.run()


class HpoCallback(UpdateProgressCallback):
    """Callback class to report score to HPO.

    Args:
        report_func (Callable): function to report score
        metric (str): metric name
        max_epoch (int): max_epoch
        task: OTX train task
    """

    def __init__(self, report_func: Callable, metric: str, max_epoch: int, task):
        if max_epoch <= 0:
            raise ValueError(f"max_epoch should be bigger than 0. Current value is {max_epoch}.")

        super().__init__()
        self._report_func = report_func
        self.metric = metric
        self._max_epoch = max_epoch
        self._task = task

    def __call__(self, progress: Union[int, float], score: Optional[float] = None):
        """When callback is called, report a score to HPO algorithm."""
        if score is not None:
            epoch = round(self._max_epoch * progress / 100)
            logger.debug(f"In hpo callback : {score} / {progress} / {epoch}")
            if self._report_func(score=score, progress=epoch) == TrialStatus.STOP:
                self._task.cancel_training()

    def __deepcopy__(self, memo):
        """Prevent repot_func from deepcopied."""
        args = [self.metric, self._max_epoch, self._task]
        copied_args = deepcopy(args, memo)
        return self.__class__(self._report_func, *copied_args)


class HpoDataset:
    """Wrapper class for DatasetEntity of dataset. It's used to make subset during HPO.

    Args:
        fullset: full dataset
        config (Optional[Dict[str, Any]], optional): hyper parameter trial config
        indices (Optional[List[int]]): dataset index. Defaults to None.
    """

    def __init__(self, fullset, config: Optional[Dict[str, Any]] = None, indices: Optional[List[int]] = None):
        self.fullset = fullset
        self.indices = indices
        if config is not None:
            subset_ratio = config["train_environment"]["subset_ratio"]
            self.subset_ratio = 1 if subset_ratio is None else subset_ratio

    def __len__(self) -> int:
        """Get length of subset."""
        if self.indices is None:
            return len(self.fullset)
        return len(self.indices)

    def __getitem__(self, indx) -> dict:
        """Get dataset at index."""
        if self.indices is None:
            return self.fullset[indx]
        return self.fullset[self.indices[indx]]

    def __getattr__(self, name):
        """When trying to get other attributes, not dataset, get values from fullset."""
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.fullset, name)

    def get_subset(self, subset: Subset):
        """Get subset according to subset_ratio if training dataset is requested.

        Args:
            subset (Subset): which subset to get

        Returns:
            HpoDataset: subset wrapped by HpoDataset
        """
        dataset = self.fullset.get_subset(subset)
        if subset != Subset.TRAINING or self.subset_ratio > 0.99:
            return dataset

        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
        indices = indices.tolist()  # type: ignore
        indices = indices[: int(len(dataset) * self.subset_ratio)]

        return HpoDataset(dataset, config=None, indices=indices)
