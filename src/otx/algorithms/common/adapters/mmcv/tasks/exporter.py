"""Base Exporter for OTX tasks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import traceback

from otx.utils.logger import get_logger

logger = get_logger()


class Exporter:
    """Exporter class for OTX export."""

    def run(self, cfg, **kwargs):  # noqa: C901
        """Run export procedure."""
        logger.info("export!")

        precision = kwargs.pop("precision", "FP32")
        export_type = kwargs.pop("type", "OPENVINO")
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
        for pipeline in cfg.data.test.pipeline:
            if pipeline.get("transforms", None):
                transforms = pipeline.transforms
                for transform in transforms:
                    if transform.type == "Collect":
                        for collect_key in transform["keys"]:
                            if collect_key != "img":
                                transform["keys"].remove(collect_key)

        model_builder = kwargs.get("model_builder")
        try:
            deploy_cfg = kwargs.get("deploy_cfg", None)
            if deploy_cfg is not None:
                self.mmdeploy_export(
                    cfg.work_dir,
                    model_builder,
                    precision,
                    export_type,
                    cfg,
                    deploy_cfg,
                    model_name,
                )
            else:
                self.naive_export(cfg.work_dir, model_builder, precision, export_type, cfg, model_name)
        except RuntimeError as ex:
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
                "onnx": os.path.join(cfg.work_dir, f"{model_name}.onnx"),
                "partitioned": [
                    {
                        f"{os.path.splitext(name)[0]}": {
                            "bin": os.path.join(cfg.work_dir, name.replace(".onnx", ".bin")),
                            "xml": os.path.join(cfg.work_dir, name.replace(".onnx", ".xml")),
                            "onnx": os.path.join(cfg.work_dir, name),
                        }
                    }
                    for name in os.listdir(cfg.work_dir)
                    if name.endswith(".onnx") and name != f"{model_name}.onnx"
                ],
            },
            "msg": "",
        }

    @staticmethod
    def mmdeploy_export(
        output_dir,
        model_builder,
        precision,
        export_type,
        cfg,
        deploy_cfg,
        model_name="model",
    ):
        """Export procedure using mmdeploy backend."""
        from otx.algorithms.common.adapters.mmdeploy.apis import MMdeployExporter

        if precision == "FP16":
            deploy_cfg.backend_config.mo_options.flags.append("--compress_to_fp16")
        MMdeployExporter.export2backend(output_dir, model_builder, cfg, deploy_cfg, export_type, model_name=model_name)

    @staticmethod
    def naive_export(output_dir, model_builder, precision, export_type, cfg, model_name="model"):
        """Export using pytorch backend."""
        raise NotImplementedError()
