"""Configuration Manager ."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import IDataset
from omegaconf import OmegaConf

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.helper import create
from otx.api.entities.model_template import ModelTemplate, parse_model_template
from otx.cli.registry import Registry as OTXRegistry
from otx.cli.utils.config import configure_dataset, override_parameters
from otx.cli.utils.errors import (
    CliException,
    ConfigValueError,
    FileNotExistError,
    NotSupportedError,
)
from otx.cli.utils.importing import get_otx_root_path
from otx.cli.utils.multi_gpu import is_multigpu_child_process
from otx.cli.utils.parser import gen_param_help, gen_params_dict_from_args
from otx.core.data.manager.dataset_manager import DatasetManager
from otx.utils.logger import get_logger
from otx.utils.utils import add_suffix_to_filename

logger = get_logger()

DEFAULT_MODEL_TEMPLATE_ID = {
    "CLASSIFICATION": "Custom_Image_Classification_EfficinetNet-B0",
    "DETECTION": "Custom_Object_Detection_Gen3_ATSS",
    "INSTANCE_SEGMENTATION": "Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50",
    "ROTATED_DETECTION": "Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50",
    "SEGMENTATION": "Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR",
    "ACTION_CLASSIFICATION": "Custom_Action_Classification_X3D",
    "ACTION_DETECTION": "Custom_Action_Detection_X3D_FAST_RCNN",
    "VISUAL_PROMPTING": "Visual_Prompting_SAM_ViT_B",
    "ANOMALY_CLASSIFICATION": "ote_anomaly_classification_padim",
    "ANOMALY_DETECTION": "ote_anomaly_detection_padim",
    "ANOMALY_SEGMENTATION": "ote_anomaly_segmentation_padim",
}

AUTOSPLIT_SUPPORTED_FORMAT = [
    "imagenet",
    "coco",
    "cityscapes",
    "voc",
]

TASK_TYPE_TO_SUPPORTED_FORMAT = {
    "CLASSIFICATION": ["imagenet", "datumaro"],
    "DETECTION": ["coco", "voc", "yolo"],
    "SEGMENTATION": ["cityscapes", "common_semantic_segmentation", "voc", "ade20k2017", "ade20k2020"],
    "ACTION_CLASSIFICATION": ["multi-cvat"],
    "ACTION_DETECTION": ["multi-cvat"],
    "VISUAL_PROMPTING": ["coco", "voc", "common_semantic_segmentation"],
    "ANOMALY_CLASSIFICATION": ["mvtec"],
    "ANOMALY_DETECTION": ["mvtec"],
    "ANOMALY_SEGMENTATION": ["mvtec"],
    "INSTANCE_SEGMENTATION": ["coco", "voc"],
    "ROTATED_DETECTION": ["coco", "voc"],
}

TASK_TYPE_TO_SUB_DIR_NAME = {
    "Incremental": "",
    "Semisupervised": "semisl",
    "Selfsupervised": "selfsl",
}


def set_workspace(task: str, root: str = None, name: str = "otx-workspace"):
    """Set workspace path according to arguments."""
    path = f"{root}/{name}-{task}" if root else f"./{name}-{task}"
    return path


class ConfigManager:  # pylint: disable=too-many-instance-attributes
    """Auto configuration manager that could set the proper configuration.

    Currently, it only supports the small amount of functions.
    * Data format detection
    * Task type detection
    * Write the data to the workspace
    * Write the data configuration to the workspace

    However, it will supports lots of things in the near future.
    * Automatic train type detection (Supervised, Self, Semi)
    * Automatic resource allocation (num_workers, HPO)

    """

    def __init__(self, args, workspace_root: Optional[str] = None, mode: str = "train"):
        # Currently, Datumaro.auto_split() can support below 3 tasks
        # Classification, Detection, Segmentation
        self.otx_root = get_otx_root_path()
        self.workspace_root = Path(workspace_root) if workspace_root else Path(".")
        self.mode = mode
        self.rebuild: bool = False
        self.create_date: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.args = args
        self.template = args.template
        self.task_type: str = ""
        self.train_type: str = ""
        self.model: str = ""

        self.dataset_manager = DatasetManager()
        self.data_format: str = ""
        self.data_config: DefaultDict[str, dict] = defaultdict(dict)

    @property
    def data_config_file_path(self) -> Path:
        """The path of the data configuration yaml to use for the task.

        Raises:
            FileNotFoundError: If data is received as args from otx train and the file does not exist, Error.

        Returns:
            Path: Path of target data configuration file.
        """
        if "data" in self.args and self.args.data:
            if Path(self.args.data).exists():
                return Path(self.args.data)
            raise FileNotExistError(f"Not found: {self.args.data}")
        return self.workspace_root / "data.yaml"

    @property
    def output_path(self) -> Path:
        """The path of output directory for workspace.

        Returns:
            Path: Path of output directory.
        """
        if "output" in self.args and self.args.output:
            output_path = Path(self.args.output)
        else:
            output_path = self.workspace_root / "outputs" / f"{self.create_date}_{self.mode}"
        if not output_path.exists():
            output_path.mkdir(exist_ok=True, parents=True)
        return output_path

    def check_workspace(self) -> bool:
        """Check that the class's workspace_root is an actual workspace folder.

        Returns:
            bool: true for workspace else false
        """
        has_template_yaml = (self.workspace_root / "template.yaml").exists()
        has_data_yaml = self.data_config_file_path.exists()
        return has_template_yaml and has_data_yaml

    def configure_template(self, model: str = None) -> None:
        """Update the template appropriate for the situation."""
        if self.check_workspace():
            # Workspace -> template O
            self.template = parse_model_template(str(self.workspace_root / "template.yaml"))
            if self.mode == "build" and self._check_rebuild():
                self.rebuild = True
                model = model if model else self.template.name
                self.template = self._get_template(str(self.task_type), model=model)
                self.train_type = self._get_train_type()
            else:
                # FIXME: Inside the workspace, ignore the --train-type args.
                self.train_type = self._get_train_type(ignore_args=True)
        elif self.template and Path(self.template).exists():
            # No workspace -> template O
            self.template = parse_model_template(self.template)
            self.train_type = self._get_train_type()
        else:
            task_type = self.task_type
            if not task_type and not model:
                if self.mode in ["train", "build"]:
                    if not hasattr(self.args, "train_data_roots"):
                        raise ConfigValueError("Can't find the argument 'train_data_roots'")
                    task_type = self.auto_task_detection(self.args.train_data_roots)
                else:
                    raise ConfigValueError("No appropriate template or task-type was found.")
            self.template = self._get_template(task_type, model=model)
            self.train_type = self._get_train_type()
        self.task_type = self.template.task_type
        self.model = self.template.name

    def _check_rebuild(self):
        """Checking for Rebuild status."""
        if self.args.task and str(self.template.task_type) != self.args.task.upper():
            raise NotSupportedError("Task Update is not yet supported.")
        result = False
        if self.args.model and self.template.name != self.args.model.upper():
            print(f"[*] Rebuild model: {self.template.name} -> {self.args.model.upper()}")
            result = True
        template_train_type = self._get_train_type(ignore_args=True)
        if self.args.train_type and template_train_type != self.args.train_type:
            self.train_type = self.args.train_type
            print(f"[*] Rebuild train-type: {template_train_type} -> {self.train_type}")
            result = True
        return result

    def configure_data_config(self, update_data_yaml: bool = True) -> None:
        """Configure data_config according to the situation and create data.yaml."""
        data_yaml_path = self.data_config_file_path
        data_yaml = configure_dataset(self.args, data_yaml_path=data_yaml_path)
        if self.mode in ("train", "build", "optimize"):
            use_auto_split = data_yaml["data"]["train"]["data-roots"] and not data_yaml["data"]["val"]["data-roots"]
            # FIXME: Hardcoded for Self-Supervised Learning
            if use_auto_split and str(self.train_type).upper() != "SELFSUPERVISED":
                splitted_dataset = self.auto_split_data(
                    data_yaml["data"]["train"]["data-roots"], str(self.task_type), self.args.train_ann_files
                )
                default_data_folder_name = "splitted_dataset"
                data_yaml = self._get_arg_data_yaml()
                self._save_data(splitted_dataset, default_data_folder_name, data_yaml)
        if (str(self.task_type).upper() == "VISUAL_PROMPTING") and (self.mode == "train"):
            # TODO (sungchul): find proper way to update data_yaml
            # data_yaml is related to OmegaConf.to_yaml and it doesn't support defaultdict
            if "options" not in data_yaml:
                data_yaml["options"] = {}
            data_yaml["options"]["use_mask"] = getattr(self.args, "params.learning_parameters.dataset.use_mask", False)
        if update_data_yaml:
            self._export_data_cfg(data_yaml, str(data_yaml_path))
            print(f"[*] Update data configuration file to: {str(data_yaml_path)}")
        self.update_data_config(data_yaml)

    def _get_train_type(self, ignore_args: bool = False) -> str:
        """Check and return the train_type received as input args.

        If value passed to args.train_type -> return this train type.
        Configure train type if None in args.
        If ignore_args passed -> use value in model template
        """

        if not ignore_args:
            if hasattr(self.args, "train_type") and self.mode in ("build", "train", "optimize"):
                self._configure_train_type()
                if self.train_type not in TASK_TYPE_TO_SUB_DIR_NAME:
                    raise NotSupportedError(f"{self.train_type} is not currently supported by otx.")
                return self.train_type

        # if ignore_args -> use train type from template file
        algo_backend = self.template.hyper_parameters.parameter_overrides.get("algo_backend", False)
        if algo_backend:
            train_type = algo_backend.get("train_type", {"default_value": "Incremental"})
            return train_type.get("default_value", "Incremental")
        return "Incremental"

    def auto_task_detection(self, data_roots: str) -> str:
        """Detect task type automatically."""
        if not data_roots:
            raise CliException("Workspace must already exist or one of {task or model or train-data-roots} must exist.")
        self.data_format = self.dataset_manager.get_data_format(data_roots)
        return self._get_task_type_from_data_format(self.data_format)

    def _configure_train_type(self):
        """Auto train type detection.

        If self.args.train_type is not None -> use args.train_type
        If train_data_roots contains only set of images -> Self-SL
        If unlabeled-data-roots were passed to CLI -> use Semi-SL
        If unlabeled_images presented in dataset structure and it is sufficient to start Semi-SL -> Semi-SL
        Overwise set Incremental training type.
        """

        def _count_imgs_in_dir(dir, recursive=False):
            """Count number of images in directory recursively."""
            import glob

            valid_suff = ["jpg", "png", "jpeg", "gif"]
            num_valid_imgs = 0
            for files in glob.iglob(f"{dir}/**", recursive=recursive):
                suff = files.split(".")[-1]
                if suff.lower() in valid_suff:
                    num_valid_imgs += 1

            return num_valid_imgs

        def _check_semisl_requirements(unlabeled_dir):
            """Check if quantity of unlabeled images is sufficient for Semi-SL learning."""
            if unlabeled_dir is None:
                return False

            if not os.path.isdir(unlabeled_dir) or not os.listdir(unlabeled_dir):
                raise ValueError(
                    "unlabeled-data-roots isn't a directory, it doesn't exist or it is empty. "
                    "Please, check command line and directory path."
                )

            all_unlabeled_images = _count_imgs_in_dir(unlabeled_dir, recursive=True)
            # check if number of unlabeled images is more than relative thershold
            if all_unlabeled_images > 1:
                return unlabeled_dir

            logger.warning(
                "WARNING: There are none or too litle images to start Semi-SL training. "
                "It should be more than relative threshold (at least 7% of labeled images) "
                "Start Supervised training instead."
            )

        # if user explicitly passed train type via args
        if self.args.train_type is not None:
            self.train_type = self.args.train_type
            return

        if self.mode == "build" and self.args.train_data_roots is None:
            # Case, when we want to build environment with tempate without dataset path
            # Set train_type to Incremental by default
            self.train_type = "Incremental"
            return

        if (
            self.args.train_data_roots is None
            or not os.path.isdir(self.args.train_data_roots)
            or not os.listdir(self.args.train_data_roots)
        ):
            raise ValueError(
                "train-data-roots isn't a directory, it doesn't exist or it is empty. "
                "Please, check command line and directory path."
            )

        if _count_imgs_in_dir(self.args.train_data_roots):
            # If train folder with images only was passed to args
            # Then we start self-supervised training
            print("[*] Selfsupervised training type detected")
            self.train_type = "Selfsupervised"
            return

        # if user explicitly passed unlabeled images folder
        valid_unlabeled_path = _check_semisl_requirements(self.args.unlabeled_data_roots)
        if valid_unlabeled_path:
            print(f"[*] Semisupervised training type detected with unlabeled data: {valid_unlabeled_path}")
            self.train_type = "Semisupervised"
            return

        self.train_type = "Incremental"

    def _get_task_type_from_data_format(self, data_format: str) -> str:
        """Detect task type.

        For some datasets (i.e. COCO, VOC, MVTec), can't be fully automated.
        Because those datasets have several format at the same time.
        (i.e. for the COCO case, object detection and instance segmentation annotations coexist)
        In this case, the task_type will be selected to default value.

        For action tasks, currently action_classification is default.

        If Datumaro supports the Kinetics, AVA datasets, MVTec, _is_cvat_format(), _is_mvtec_format()
        functions will be deleted.
        """

        for task_key, data_value in TASK_TYPE_TO_SUPPORTED_FORMAT.items():
            if data_format in data_value:
                self.task_type = task_key
                print(f"[*] Detected task type: {self.task_type}")
                return task_key
        raise ConfigValueError(f"Can't find proper task. we are not support {data_format} format, yet.")

    def auto_split_data(self, data_roots: str, task: str, ann_file: Optional[str] = None):
        """Automatically Split train data --> train/val dataset."""
        self.data_format = self.dataset_manager.get_data_format(data_roots)
        dataset = self.dataset_manager.import_dataset(data_root=data_roots, data_format=self.data_format)
        train_dataset = self.dataset_manager.get_train_dataset(dataset)
        if ann_file is not None:
            train_dataset = self.dataset_manager.import_dataset(ann_file, data_format=self.data_format, subset="train")
        val_dataset = self.dataset_manager.get_val_dataset(dataset)
        splitted_dataset = None
        if self.data_format in AUTOSPLIT_SUPPORTED_FORMAT:
            if val_dataset is None:
                splitted_dataset = self.dataset_manager.auto_split(
                    task=task,
                    dataset=train_dataset,
                    split_ratio=[("train", 0.8), ("val", 0.2)],
                )
            else:
                print(f"[*] Found validation data in your dataset in {data_roots}. It'll be used as validation data.")
                splitted_dataset = {"train": train_dataset, "val": val_dataset}
        else:
            print(f"[*] Current auto-split can't support the {self.data_format} format.")
        return splitted_dataset

    def _get_arg_data_yaml(self):
        # TODO: This should modify data yaml format to data_config format.
        """Save the splitted dataset and data.yaml to the workspace."""
        data_yaml = self._create_empty_data_cfg()
        if self.mode in ("train", "optimize"):
            if self.args.train_data_roots:
                data_yaml["data"]["train"]["data-roots"] = self.args.train_data_roots
            if self.args.train_ann_files:
                data_yaml["data"]["train"]["ann-files"] = self.args.train_ann_files
            if self.args.val_data_roots:
                data_yaml["data"]["val"]["data-roots"] = self.args.val_data_roots
            if self.args.val_ann_files:
                data_yaml["data"]["val"]["ann-files"] = self.args.val_ann_files
            if self.args.unlabeled_data_roots:
                data_yaml["data"]["unlabeled"]["data-roots"] = self.args.unlabeled_data_roots
            if self.args.unlabeled_file_list:
                data_yaml["data"]["unlabeled"]["file-list"] = self.args.unlabeled_file_list
        elif self.mode == "test":
            if self.args.test_data_roots:
                data_yaml["data"]["test"]["data-roots"] = self.args.test_data_roots
            if self.args.test_ann_files:
                data_yaml["data"]["test"]["ann-files"] = self.args.test_ann_files
        return data_yaml

    def _save_data(
        self,
        splitted_dataset: Dict[str, IDataset],
        default_data_folder_name: str,
        data_config: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> None:
        """Save the data for the classification task.

        Args:
            splitted_dataset (dict): A dictionary containing split datasets
            default_data_folder_name (str): the name of splitted dataset folder
            data_config (dict): dictionary that has information about data path
        """
        for phase, dataset in splitted_dataset.items():
            dst_dir_path = self.workspace_root / default_data_folder_name / phase
            data_config["data"][phase]["data-roots"] = str(dst_dir_path.absolute())
            # Convert Datumaro class: DatasetFilter(IDataset) --> Dataset
            if isinstance(dataset, Dataset):
                datum_dataset = dataset
            else:
                datum_dataset = Dataset.from_extractors(dataset)
            # Write the data
            # TODO: consider the way that reduces disk stroage
            # Currently, saving all images to the workspace.
            # It might needs quite large disk storage.
            self.dataset_manager.export_dataset(
                dataset=datum_dataset, output_dir=str(dst_dir_path), data_format=self.data_format, save_media=True
            )

        if data_config["data"]["unlabeled"]["data-roots"] is not None:
            data_config["data"]["unlabeled"]["data-roots"] = str(
                Path(data_config["data"]["unlabeled"]["data-roots"]).absolute()
            )
        if data_config["data"]["unlabeled"]["file-list"] is not None:
            data_config["data"]["unlabeled"]["file-list"] = str(
                Path(data_config["data"]["unlabeled"]["file-list"]).absolute()
            )

    def _create_empty_data_cfg(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Create default dictionary to represent the dataset."""
        data_config: Dict[str, Dict[str, Any]] = {"data": {}}
        for subset in ["train", "val", "test"]:
            data_subset = {"ann-files": None, "data-roots": None}
            data_config["data"][subset] = data_subset
        data_config["data"]["unlabeled"] = {"file-list": None, "data-roots": None}
        return data_config

    def _export_data_cfg(self, data_cfg: Dict[str, Dict[str, Dict[str, Any]]], output_path: str) -> None:
        """Export the data configuration file to output_path."""
        Path(output_path).write_text(OmegaConf.to_yaml(data_cfg), encoding="utf-8")

    def get_hyparams_config(self, override_param: Optional[List] = None) -> ConfigurableParameters:
        """Separates the input params received from args and updates them.."""
        hyper_parameters = self.template.hyper_parameters.data
        type_hint = gen_param_help(hyper_parameters)
        updated_hyper_parameters = gen_params_dict_from_args(
            self.args, override_param=override_param, type_hint=type_hint
        )
        override_parameters(updated_hyper_parameters, hyper_parameters)
        return create(hyper_parameters)

    def get_dataset_config(self, subsets: List[str], hyper_parameters: Optional[ConfigurableParameters] = None) -> dict:
        """Returns dataset_config in a format suitable for each subset.

        Args:
            subsets (list, str): Defaults to ["train", "val", "unlabeled"].
            hyper_parameters (ConfigurableParameters): Set of hyper parameters.

        Returns:
            dict: dataset_config
        """
        if str(self.train_type).upper() == "INCREMENTAL" and "unlabeled" in subsets:
            subsets.remove("unlabeled")
        dataset_config: Dict[str, Any] = {
            "task_type": self.task_type,
            "train_type": self.train_type,
            "encryption_key": self.encryption_key,
        }
        for subset in subsets:
            if f"{subset}_subset" in self.data_config:
                if self.data_config[f"{subset}_subset"]["data_roots"]:
                    dataset_config.update({f"{subset}_data_roots": self.data_config[f"{subset}_subset"]["data_roots"]})
                if "ann_files" in self.data_config[f"{subset}_subset"]:
                    dataset_config.update({f"{subset}_ann_files": self.data_config[f"{subset}_subset"]["ann_files"]})
                if "file_list" in self.data_config[f"{subset}_subset"]:
                    dataset_config.update({f"{subset}_file_list": self.data_config[f"{subset}_subset"]["file_list"]})
        if "options" in self.data_config:
            dataset_config.update(self.data_config["options"])
        if hyper_parameters is not None:
            dataset_config["cache_config"] = {}
            algo_backend = getattr(hyper_parameters, "algo_backend", None)
            if algo_backend:
                storage_cache_scheme = getattr(algo_backend, "storage_cache_scheme", None)
                if storage_cache_scheme is not None:
                    storage_cache_scheme = str(storage_cache_scheme)
                dataset_config["cache_config"]["scheme"] = storage_cache_scheme

            learning_parameters = getattr(hyper_parameters, "learning_parameters", None)
            if learning_parameters:
                num_workers = getattr(learning_parameters, "num_workers", 0)
                dataset_config["cache_config"]["num_workers"] = num_workers

        if str(self.task_type).upper() == "SEGMENTATION" and str(self.train_type).upper() == "SELFSUPERVISED":
            # FIXME: manually set a path to save pseudo masks in workspace
            train_type_rel_path = TASK_TYPE_TO_SUB_DIR_NAME[self.train_type]
            train_type_dir = self.workspace_root / train_type_rel_path
            dataset_config["pseudo_mask_dir"] = train_type_dir / "detcon_mask"
        return dataset_config

    def update_data_config(self, data_yaml: dict) -> None:
        # TODO: This also requires uniformity in the format.
        """Convert the data yaml format to the data_config format consumed by the task.

        Args:
            data_yaml (dict): data.yaml format
        """
        if "data-roots" in data_yaml["data"]["train"]:
            self.data_config["train_subset"] = {"data_roots": data_yaml["data"]["train"]["data-roots"]}
            if "ann-files" in data_yaml["data"]["train"]:
                self.data_config["train_subset"]["ann_files"] = data_yaml["data"]["train"]["ann-files"]
        if "data-roots" in data_yaml["data"]["val"]:
            self.data_config["val_subset"] = {"data_roots": data_yaml["data"]["val"]["data-roots"]}
            if "ann-files" in data_yaml["data"]["val"]:
                self.data_config["val_subset"]["ann_files"] = data_yaml["data"]["val"]["ann-files"]
        if "data-roots" in data_yaml["data"]["test"]:
            self.data_config["test_subset"] = {"data_roots": data_yaml["data"]["test"]["data-roots"]}
            if "ann-files" in data_yaml["data"]["test"]:
                self.data_config["test_subset"]["ann_files"] = data_yaml["data"]["test"]["ann-files"]
        if "unlabeled" in data_yaml["data"] and data_yaml["data"]["unlabeled"]["data-roots"]:
            self.data_config["unlabeled_subset"] = {
                "data_roots": data_yaml["data"]["unlabeled"]["data-roots"],
                "file_list": data_yaml["data"]["unlabeled"]["file-list"],
            }
        # FIXME: Hardcoded for Self-Supervised Learning
        if self.mode in ("train", "optimize") and str(self.train_type).upper() == "SELFSUPERVISED":
            self.data_config["val_subset"] = {"data_roots": None}

        if str(self.task_type).upper() == "VISUAL_PROMPTING":
            self.data_config["options"]["use_mask"] = data_yaml["options"]["use_mask"]

    def _get_template(self, task_type: str, model: Optional[str] = None) -> ModelTemplate:
        """Returns the appropriate template for each situation.

        Args:
            task_type (str): The task_type registered in the registry. Used for filtering.
            model (str, optional): The task_type registered in the registry. Used for filtering. Defaults to None.

        Returns:
            ModelTemplate: Selected model template.
        """
        otx_registry = OTXRegistry(self.otx_root).filter(task_type=task_type if task_type else None)
        if model:
            template_lst = [temp for temp in otx_registry.templates if temp.name.lower() == model.lower()]
            if not template_lst:
                raise NotSupportedError(
                    f"[*] {model} is not a type supported by OTX {task_type}."
                    f"\n[*] Please refer to 'otx find --template --task {task_type}'"
                )
            template = template_lst[0]
        else:
            template = otx_registry.get(DEFAULT_MODEL_TEMPLATE_ID[task_type.upper()])
        return template

    def build_workspace(self, new_workspace_path: Optional[str] = None) -> None:
        """Create OTX workspace with Template configs from task type.

        This function provides a user-friendly OTX workspace and provides more intuitive
        and create customizable templates to help users use all the features of OTX.

        Args:
            new_workspace_path (Optional[str]): Workspace dir name for build
        """

        # Create OTX-workspace
        if is_multigpu_child_process():
            return
        # Check whether the workspace is existed or not
        if self.check_workspace() and not self.rebuild:
            return
        if self.rebuild:
            print(f"[*] \t- Rebuild: model-{self.model} / train type-{self.train_type}")
        if new_workspace_path:
            self.workspace_root = Path(new_workspace_path)
        elif not self.check_workspace():
            self.workspace_root = Path(set_workspace(task=self.task_type))
        self.workspace_root.mkdir(exist_ok=True, parents=True)
        print(f"[*] Workspace Path: {self.workspace_root}")
        print(f"[*] Load Model Template ID: {self.template.model_template_id}")
        print(f"[*] Load Model Name: {self.template.name}")

        template_dir = Path(self.template.model_template_path).parent

        # Copy task base configuration file
        task_configuration_path = template_dir / self.template.hyper_parameters.base_path
        shutil.copyfile(task_configuration_path, str(self.workspace_root / "configuration.yaml"))
        # Load Model Template
        template_config = OmegaConf.load(self.template.model_template_path)
        template_config.hyper_parameters.base_path = "./configuration.yaml"

        # Configuration of Train Type value
        train_type_rel_path = TASK_TYPE_TO_SUB_DIR_NAME[self.train_type]

        # FIXME: Hardcoded solution for supcon
        enable_supcon = gen_params_dict_from_args(self.args).get("learning_parameters", {})
        enable_supcon = enable_supcon.get("enable_supcon", {"value": False})
        if enable_supcon.get("value", False):
            train_type_rel_path = "supcon"

        model_dir = template_dir.absolute() / train_type_rel_path
        if not model_dir.exists():
            raise NotSupportedError(f"[*] {self.train_type} is not a type supported by OTX {self.task_type}")
        train_type_dir = self.workspace_root / train_type_rel_path
        train_type_dir.mkdir(exist_ok=True)

        # Update Hparams
        if (model_dir / "hparam.yaml").exists():
            template_config = OmegaConf.merge(template_config, OmegaConf.load(str(model_dir / "hparam.yaml")))

        # Copy config files
        config_files = [
            (model_dir, "model.py", train_type_dir),
            (model_dir, "model_multilabel.py", train_type_dir),
            (model_dir, "data_pipeline.py", train_type_dir),
            (template_dir, "tile_pipeline.py", self.workspace_root),
            (template_dir, "deployment.py", self.workspace_root),
            (template_dir, "hpo_config.yaml", self.workspace_root),
            (template_dir, "model_hierarchical.py", self.workspace_root),
        ]
        for target_dir, file_name, dest_dir in config_files:
            self._copy_config_files(target_dir, file_name, dest_dir)

            # check xpu file exists
            xpu_file = add_suffix_to_filename(target_dir / file_name, "_xpu")
            if xpu_file.exists():
                self._copy_config_files(xpu_file.parent, xpu_file.name, dest_dir)

        (self.workspace_root / "template.yaml").write_text(OmegaConf.to_yaml(template_config))

        # Copy deployment_tile_classifier for Instance Segmentation
        if (model_dir / "deployment_tile_classifier.py").exists():
            shutil.copyfile(
                str(model_dir / "deployment_tile_classifier.py"),
                str(train_type_dir / "deployment_tile_classifier.py"),
            )
            print(f"[*] \t- Updated: {str(train_type_dir / 'deployment_tile_classifier.py')}")

        # Copy compression_config.json
        if (model_dir / "compression_config.json").exists():
            shutil.copyfile(
                str(model_dir / "compression_config.json"),
                str(train_type_dir / "compression_config.json"),
            )
            print(f"[*] \t- Updated: {str(train_type_dir / 'compression_config.json')}")

        # copy PTQ config
        if (model_dir / "ptq_optimization_config.py").exists():
            shutil.copyfile(
                str(model_dir / "ptq_optimization_config.py"),
                str(train_type_dir / "ptq_optimization_config.py"),
            )
            print(f"[*] \t- Updated: {str(train_type_dir / 'ptq_optimization_config.py')}")

        if not (self.workspace_root / "data.yaml").exists():
            data_yaml = self._get_arg_data_yaml()
            self._export_data_cfg(data_yaml, str((self.workspace_root / "data.yaml")))

        self.template = parse_model_template(str(self.workspace_root / "template.yaml"))

    def _copy_config_files(self, target_dir: Path, file_name: str, dest_dir: Path) -> None:
        """Copy Configuration files for workspace."""
        if (target_dir / file_name).exists():
            if file_name.endswith(".py"):
                try:
                    from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
                        OTXConfig,
                    )

                    config = OTXConfig.fromfile(str(target_dir / file_name))
                    self._patch_cli_configs(config)
                    config.dump(str(dest_dir / file_name))
                except Exception as exc:
                    raise CliException(f"{self.task_type} requires mmcv-full to be installed.") from exc
            elif file_name.endswith((".yml", ".yaml")):
                config = OmegaConf.load(str(target_dir / file_name))
                (dest_dir / file_name).write_text(OmegaConf.to_yaml(config))
            print(f"[*] \t- Updated: {str(dest_dir / file_name)}")

    def _patch_cli_configs(self, config):
        """Patch for CLI configurations."""
        if config.get("ignore", None):
            # FIXME: In the CLI, there is currently no case for using the ignore label.
            # so the workspace's model patches ignore to False.
            config.ignore = False
            print("In the CLI, Update ignore to false in model configuration.")
        if hasattr(config, "deterministic") and hasattr(self.args, "deterministic"):
            config.deterministic = self.args.deterministic
        if hasattr(config, "seed") and hasattr(self.args, "seed") and self.args.seed:
            config.seed = self.args.seed

    @property
    def encryption_key(self):
        """Get encryption key from CLI argument or OS environment variables. If it is not specified, return None."""
        key_from_args = getattr(self.args, "encryption_key", None)
        key_from_envs = os.environ.get("ENCRYPTION_KEY", None)

        if key_from_args is not None and key_from_envs is not None:
            raise ValueError(
                "You have to choose either one of the two, whether encryption_key is "
                "specified as a CLI argument (--encryption-key=<key>) or specified in "
                "an environment variable (ENCRYPTION_KEY=<key>). "
            )

        if key_from_args is not None:
            return key_from_args
        if key_from_envs is not None:
            return key_from_envs

        return None
