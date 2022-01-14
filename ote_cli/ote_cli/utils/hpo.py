"""
Utils for HPO with hpopt
"""

import builtins
import collections
import importlib
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import time
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
from ote_cli.utils.io import generate_label_schema

try:
    import hpopt
except ImportError:
    hpopt = None


def check_hpopt_available():
    """Check whether hpopt is avaiable"""

    if hpopt is None:
        return False
    return True


def run_hpo(args, environment, dataset, task_type):
    """Update the environment with better hyper-parameters found by HPO"""
    if check_hpopt_available():
        if task_type not in {TaskType.CLASSIFICATION, TaskType.DETECTION}:
            print(
                "Currently supported task types are classification and detection."
                f"{task_type} is not supported yet."
            )
            return

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
        hyper_parameters = hpo.run()

        environment.set_hyper_parameters(hyper_parameters)

        if args.load_weights:
            environment.model.confiugration.configurable_parameters = hyper_parameters


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
    model,
    hyper_parameters,
    model_template,
    dataset_paths,
    task_type,
):
    """Run each training of each trial with given hyper parameters"""

    if isinstance(hyper_parameters, dict):
        current_params = {}
        for val in hyper_parameters["parameters"]:
            current_params[val] = hyper_parameters[val]
        hyper_parameters = create(model_template.hyper_parameters.data)
        HpoManager.set_hyperparameter(hyper_parameters, current_params)

    if dataset_paths is None:
        raise ValueError("Dataset is not defined.")

    impl_class = get_dataset_class(task_type)
    dataset = impl_class(
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
        model=model,
        hyper_parameters=hyper_parameters,
        label_schema=generate_label_schema(dataset, task_type),
        model_template=model_template,
    )

    hyper_parameters = train_env.get_hyper_parameters()

    # set epoch
    if task_type == TaskType.CLASSIFICATION:
        (hyper_parameters.learning_parameters.max_num_epochs) = hp_config["iterations"]
    elif task_type in (TaskType.DETECTION, TaskType.SEGMENTATION):
        hyper_parameters.learning_parameters.num_iters = hp_config["iterations"]

    # set hyper-parameters and print them
    HpoManager.set_hyperparameter(hyper_parameters, hp_config["params"])
    print(f"hyper parameter of current trial : {hp_config['params']}")

    train_env.set_hyper_parameters(hyper_parameters)
    train_env.model_template.hpo = {
        "hp_config": hp_config,
        "metric": hp_config["metric"],
    }

    impl_class = get_impl_class(train_env.model_template.entrypoints.base)
    task = impl_class(task_environment=train_env)

    dataset = HpoDataset(dataset, hp_config)
    if train_env.model:
        train_env.model.train_dataset = dataset
        train_env.model.confiugration.configurable_parameters = hyper_parameters

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


class HpoCallback(UpdateProgressCallback):
    """Callback class to report score to hpopt"""

    def __init__(self, hp_config, metric, hpo_task):
        super().__init__()
        self.hp_config = hp_config
        self.metric = metric
        self.hpo_task = hpo_task

    def __call__(self, progress: float, score: Optional[float] = None):
        if score is not None:
            if hpopt.report(config=self.hp_config, score=score) == hpopt.Status.STOP:
                self.hpo_task.cancel_training()


class HpoManager:
    """Manage overall HPO process"""

    def __init__(
        self, environment, dataset, dataset_paths, expected_time_ratio, hpo_save_path
    ):
        self.environment = environment
        self.dataset_paths = dataset_paths
        self.work_dir = hpo_save_path

        try:
            with open(
                osp.join(
                    osp.dirname(self.environment.model_template.model_template_path),
                    "hpo_config.yaml",
                ),
                "r",
                encoding="utf-8",
            ) as f:
                hpopt_cfg = yaml.safe_load(f)
        except FileNotFoundError as err:
            print("hpo_config.yaml file should be in directory where template.yaml is.")
            raise err

        self.algo = hpopt_cfg.get("search_algorithm", "smbo")
        self.metric = hpopt_cfg.get("metric", "mAP")

        num_available_gpus = torch.cuda.device_count()

        if num_available_gpus == 0 and self.algo == "asha":
            print(
                "There is no available CUDA devices. The search algorithm of HPO uses smbo instead of asha."
            )

        if num_available_gpus == 1 and self.algo == "asha":
            print(
                "There is only one CUDA devices. The search algorithm of HPO uses smbo instead of asha."
            )

        if num_available_gpus <= 1:
            self.algo = "smbo"

        self.num_gpus_per_trial = 1

        train_dataset_size = len(dataset.get_subset(Subset.TRAINING))
        val_dataset_size = len(dataset.get_subset(Subset.VALIDATION))

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
            batch_size_name="learning_parameters.batch_size",
        )

        if self.algo == "smbo":
            hpopt_arguments["num_init_trials"] = hpopt_cfg.get("num_init_trials")
            hpopt_arguments["num_trials"] = hpopt_cfg.get("num_trials")
        elif self.algo == "asha":
            hpopt_arguments["num_brackets"] = hpopt_cfg.get("num_brackets")
            hpopt_arguments["min_iterations"] = hpopt_cfg.get("min_iterations")
            hpopt_arguments["reduction_factor"] = hpopt_cfg.get("reduction_factor")
            hpopt_arguments["num_trials"] = hpopt_cfg.get("num_trials")
            hpopt_arguments["num_workers"] = (
                num_available_gpus // self.num_gpus_per_trial
            )

        HpoManager.remove_empty_keys(hpopt_arguments)

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

                hp_config["metric"] = self.metric

                hpo_work_dir = osp.abspath(
                    osp.join(
                        osp.dirname(hp_config["file_path"]),
                        str(hp_config["trial_id"]),
                    )
                )

                # Clear hpo_work_dir
                if osp.exists(hpo_work_dir):
                    shutil.rmtree(hpo_work_dir)
                Path(hpo_work_dir).mkdir(parents=True)

                _kwargs = {
                    "hp_config": hp_config,
                    "model": self.environment.model,
                    "hyper_parameters": vars(
                        self.environment.get_hyper_parameters().learning_parameters
                    ),
                    "model_template": self.environment.model_template,
                    "dataset_paths": self.dataset_paths,
                    "task_type": self.environment.model_template.task_type,
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

        hyper_parameters = self.environment.get_hyper_parameters()
        HpoManager.set_hyperparameter(hyper_parameters, best_config)

        self.hpo.print_results()

        print("Best Hyper-parameters")
        print(best_config)

        return hyper_parameters

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
            search_space[key] = hpopt.search_space(val["param_type"], val["range"])
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
        learning_parameters = environment.get_hyper_parameters().learning_parameters
        if task_type == TaskType.CLASSIFICATION:
            num_full_iterations = learning_parameters.max_num_epochs
        elif task_type in (TaskType.DETECTION, TaskType.SEGMENTATION):
            num_full_iterations = learning_parameters.num_iters

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
