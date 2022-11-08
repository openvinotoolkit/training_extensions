import os
from abc import abstractmethod
from typing import Dict

from otx.api.dataset import Dataset
from otx.core.dataset import IDatasetAdapter
from otx.core.job import IJob
from otx.core.model import IModel
from otx.utils import import_and_get_class_from_path
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class Task:
    def __init__(self, config: Config):
        logger.info(f"config = {config}")
        self.config = config

        self.jobs: Dict[IJob]
        self.model_adapter: IModel
        self.dataset_adapter: IDatasetAdapter

        if not hasattr(self.config, "gpu_ids"):
            gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            logger.info(f"CUDA_VISIBLE_DEVICES = {gpu_ids}")
            if gpu_ids is not None:
                if isinstance(gpu_ids, str):
                    self.config.gpu_ids = range(len(gpu_ids.split(",")))
                else:
                    raise ValueError(f"not supported type for gpu_ids: {type(gpu_ids)}")
            else:
                self.config.gpu_ids = range(1)

        if not hasattr(self.config, "work_dir"):
            self.config.work_dir = "./workspace"

        self._init_jobs(self.config.jobs)
        self._init_model_adapter(self.config.model)
        self._init_dataset_adapter(self.config.data)

    def _init_jobs(self, jobs):
        self.jobs = {}
        for job in jobs:
            for spec in job.spec:
                job_cls = import_and_get_class_from_path(job.adapter)
                self.jobs[spec] = job_cls(spec, self.config[spec])
        logger.info(f"initialized jobs = {self.jobs}")

    def _run_job(self, spec, *args, **kwargs):
        return self.jobs[spec].run(*args, task_config=self.config, **kwargs)

    @abstractmethod
    def _init_model_adapter(self, model_cfg):
        raise NotImplementedError()

    @abstractmethod
    def _init_dataset_adapter(self, data_cfg):
        raise NotImplementedError()

    @abstractmethod
    def train(self, dataset: Dataset, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def eval(self, dataset: Dataset, metric, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def infer(self, dataset: Dataset, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def export(self, type: str, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def optimize(self, type: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def create(task_yaml: str):
        if os.path.exists(task_yaml):
            config = Config.fromfile(task_yaml)
            task_cls = import_and_get_class_from_path(config.algo.impl)
            if task_cls is None:
                logger.error(f"cannot find a supported task for this algo configuration: {config.algo}")
                return None
            else:
                return task_cls(config)
        else:
            logger.error(f"file {task_yaml} is not existed!")
        return None
