"""Export task for OTX Segmentation with MMSEG."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv.runner import wrap_fp16_model

from otx.algorithms.common.adapters.mmcv.tasks.exporter import Exporter
from otx.algorithms.common.adapters.mmdeploy.utils import sync_batchnorm_2_batchnorm
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.utils.builder import build_segmentor

logger = get_logger()


class SegmentationExporter(Exporter):
    """Exporter for OTX Segmentation using mmsegmentation training backend."""

    def run(self, cfg, **kwargs):  # noqa: C901
        """Run exporter stage."""

        precision = kwargs.get("precision", "FP32")
        model_builder = kwargs.get("model_builder", build_segmentor)

        def model_builder_helper(*args, **kwargs):
            model = model_builder(*args, **kwargs)
            # TODO: handle various input size
            model = sync_batchnorm_2_batchnorm(model, 2)

            if precision == "FP16":
                wrap_fp16_model(model)
            elif precision == "INT8":
                from nncf.torch.nncf_network import NNCFNetwork

                assert isinstance(model, NNCFNetwork)

            return model

        kwargs["model_builder"] = model_builder_helper

        return super().run(cfg, **kwargs)

    @staticmethod
    def naive_export(output_dir, model_builder, precision, export_type, cfg, model_name="model"):
        """Export using pytorch backend."""
        from mmseg.apis.inference import LoadImage
        from mmseg.datasets.pipelines import Compose

        from otx.algorithms.common.adapters.mmdeploy.apis import NaiveExporter

        def get_fake_data(cfg, orig_img_shape=(128, 128, 3)):
            pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
            pipeline = Compose(pipeline)
            data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
            data = pipeline(data)
            return data

        fake_data = get_fake_data(cfg)
        opset_version = 11

        NaiveExporter.export2backend(
            output_dir,
            model_builder,
            cfg,
            fake_data,
            precision=precision,
            model_name=model_name,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            export_type=export_type,
        )
