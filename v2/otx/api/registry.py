from otx import OTXConstants
from otx.utils.logger import get_logger

logger = get_logger()


class __Registry():
    def __init__(self, config_path: str):
        # TODO: retrieve task configurations from builtin config path + given user's config path
        self._configs = dict(
            classification=dict(
                classifier=dict(
                    path=f"{config_path}/tasks/classification/cls-incr-classifier.yaml",
                ),
                multilabel=dict(
                    path="path/to/multi-label-cls-config.yaml",
                ),
                hierachical=dict(
                    path="path/to/hier-label-cls-config.yaml",
                ),
            ),
            anomaly=dict(
                classification_stfpm=dict(
                    path=f"{config_path}/tasks/anomaly/classification/stfpm.yaml",
                ),
                classification_draem=dict(
                    path=f"{config_path}/tasks/anomaly/classification/draem.yaml",
                )
            ),
        )

        # TODO: retrieve models from builtin config path
        self._models = dict(
            classification=dict(
                classifier=[
                    f"{config_path}/models/classification/image_classifier.yaml",
                    f"{config_path}/models/classification/sam_image_classifier.yaml",
                ],
                multilabel=[
                    "path/to/multilabel/classification/model.yaml",
                ],
                hierachical=[
                    "path/to/hierachical/classification/model.yaml",
                ],
            ),
            anomaly=dict(
                classification_stfpm=f"{config_path}/models/anomaly/stfpm.yaml",
                classification_draem=f"{config_path}/models/anomaly/draem.yaml",
            )
        )

        self._backbones = dict(
            classifier=dict(
                effcientnet=f"{config_path}/models/backbones/efficientnet.yaml",
                mobilenet_v3=f"{config_path}/models/backbones/mobilenet_v3.yaml",
            ),
            classification_stfpm=dict(
                resnet18=f"{config_path}/models/backbones/resnet18.yaml"
            ),
            classification_draem=None,
        )
        # # TODO: need to define support map between a task and models
        # self._supported_map: Dict[str, str] = dict()

    @property
    def configs(self):
        return self._configs

    @property
    def models(self):
        return self._models

    @property
    def backbones(self):
        return self._backbones


__registry = __Registry(OTXConstants.CONFIG_PATH)


def find_task_types():
    types = []
    for k, _ in __registry.configs.items():
        types.append(k)
    return types


def find_tasks(type: str):
    tasks = []
    for _, item in __registry.configs[type].items():
        tasks.append(item["path"])
    return tasks


def find_models(task_yaml: str):
    for task_type, algos in __registry.configs.items():
        for algo, algo_cfg in algos.items():
            if task_yaml == algo_cfg["path"]:
                return __registry.models[task_type][algo]
    logger.error(f"cannot find compatible models for {task_yaml}")
    return None


def find_backbones(model_yaml: str):
    for task_type, models in __registry.models.items():
        for algo, yaml_list in models.items():
            for yaml in yaml_list:
                if model_yaml == yaml:
                    return __registry.backbones[algo]
    logger.error(f"cannot find compatible backbones for {model_yaml}")
    return None
