# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import traceback

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


class ExporterMixin(object):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):  # noqa: C901
        self._init_logger()
        logger.info("exporting the model")
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            logger.warning(f"Supported modes are {self.mode} but '{mode}' is given.")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        logger.info("export!")

        #  from torch.jit._trace import TracerWarning
        #  import warnings
        #  warnings.filterwarnings("ignore", category=TracerWarning)
        precision = kwargs.pop("precision", "FP32")
        if precision not in ("FP32", "FP16", "INT8"):
            raise NotImplementedError
        logger.info(f"Model will be exported with precision {precision}")
        model_name = cfg.get("model_name", "model")

        # TODO: handle complicated pipeline
        # If test dataset is a wrapper dataset
        # pipeline may not include load transformation which is assumed to be included afterwards
        # Here, we assume simple wrapper datasets where pipeline of the wrapper is just a consecutive one.
        if cfg.data.test.get("dataset", None) or cfg.data.test.get("datasets", None):
            dataset = cfg.data.test.get("dataset", cfg.data.test.get("datasets", [None])[0])
            assert dataset is not None
            pipeline = dataset.get("pipeline", [])
            pipeline += cfg.data.test.get("pipeline", [])
            cfg.data.test.pipeline = pipeline

        model_builder = kwargs.get("model_builder")
        try:
            deploy_cfg = kwargs.get("deploy_cfg", None)
            if deploy_cfg is not None:
                self.mmdeploy_export(
                    cfg.work_dir,
                    model_builder,
                    precision,
                    cfg,
                    deploy_cfg,
                    model_name,
                )
            else:
                self.naive_export(cfg.work_dir, model_builder, precision, cfg, model_name)
        except Exception as ex:
            # output_model.model_status = ModelStatus.FAILED
            # raise RuntimeError('Optimization was unsuccessful.') from ex
            return {
                "outputs": None,
                "msg": f"exception {type(ex)}: {ex}\n\n{traceback.format_exc()}",
            }

        return {
            "outputs": {
                "bin": os.path.join(cfg.work_dir, f"{model_name}.bin"),
                "xml": os.path.join(cfg.work_dir, f"{model_name}.xml"),
                "partitioned": [
                    {
                        "bin": os.path.join(cfg.work_dir, name.replace(".xml", ".bin")),
                        "xml": os.path.join(cfg.work_dir, name),
                    }
                    for name in os.listdir(cfg.work_dir)
                    if name.endswith(".xml")
                    and name != f"{model_name}.xml"
                    and name.replace(".xml", ".bin") in os.listdir(cfg.work_dir)
                ],
            },
            "msg": "",
        }

    @staticmethod
    def mmdeploy_export(
        output_dir,
        model_builder,
        precision,
        cfg,
        deploy_cfg,
        model_name="model",
    ):
        from otx.algorithms.common.adapters.mmdeploy.apis import MMdeployExporter

        if precision == "FP16":
            deploy_cfg.backend_config.mo_options.flags.append("--compress_to_fp16")
        MMdeployExporter.export2openvino(output_dir, model_builder, cfg, deploy_cfg, model_name=model_name)

    @staticmethod
    def naive_export(output_dir, model_builder, precision, cfg, model_name="model"):
        raise NotImplementedError()
