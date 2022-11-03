from otx.algorithms.base import BaseTask
from otx.utils.logger import get_logger

logger = get_logger()


class ClsTask(BaseTask):
    def eval(self, dataset, metric, **kwargs):
        logger.info("*** task.eval() ***")
        spec = kwargs.get("spec", "eval")

        logger.info("=== configure task ===")
        self.jobs[spec].configure(self.config,
            model_cfg=self.adapters["model"].config,
            data_cfg=self.adapters["data"].config,
            training=False,
            model_ckpt=self.adapters["model"].ckpt,
        )

        logger.info("=== update config ===")
        self.update_model_cfg(self.config.model, overwrite=True)
        self.update_data_cfg(self.config.data, overwrite=True)

        logger.info("=== prepare model ===")
        self.model = self._get_model()
        dataset = kwargs.get("dataset", self.adapters["data"].get_subset("test"))
        infer_results = self.infer(dataset, **kwargs)
        return self._run_job("eval", metric, infer_results, **kwargs, **self.config[spec])

    def infer(self, dataset, **kwargs):
        logger.info("*** task.infer() ***")
        spec = kwargs.get("spec", "infer")
        self.model = self._get_model()
        dataset = kwargs.get("dataset", self.adapters["data"].get_test_dataset())
        return self._run_job("infer", model, dataset, **kwargs, **self.config[spec])

    def export(self, type, **kwargs):
        logger.info("*** task.export() ***")
        spec = kwargs.get("spec", "export")
        model = self._get_model()
        return self._run_job("export", model, **kwargs, **self.config[spec])

    def optimize(self, type, **kwargs):
        logger.info("*** task.optimize() ***")
        spec = kwargs.get("spec", "optimize")
        self.optimized_model = "optimized model"
        self.model_status = ModelStatus.OPTIMIZED
        return True


class ClsIncrClassification(ClsTask):
    def train(self, dataset, **kwargs):
        logger.info("*** task.train() ***")
        spec = kwargs.get("spec", "train")

        logger.info("=== configure task ===")
        self.jobs[spec].configure(self.config,
            model_cfg=self.model_adapter.config,
            data_cfg=self.dataset_adapter.config,
        )

        logger.info("=== update config ===")
        self.update_model_cfg(self.config.model, overwrite=True)
        self.update_data_cfg(self.config.data, overwrite=True)

        logger.info("=== prepare model ===")
        self.model = self._get_model()
        logger.info("=== prepare dataset ===")
        train_dataset = kwargs.pop("train_dataset", self.dataset_adapter.get_subset("train"))
        val_dataset = kwargs.pop("val_dataset", self.dataset_adapter.get_subset("val"))
        datasets = dict(
            train=train_dataset,
            val=val_dataset,
        )
        results = self._run_job(spec, self.model, datasets, **kwargs, **self.config[spec])
        ckpt = results.get("final_ckpt")
        if ckpt is not None:
            self.adapters["model"].ckpt = ckpt
            self.model_status = ModelStatus.TRAINED