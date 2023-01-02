# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import warnings

import torch
from mmcv.runner import load_checkpoint
from mmseg.apis import export_model
from mmseg.models import build_segmentor

from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .stage import SegStage

logger = get_logger()


@STAGES.register_module()
class SegExporter(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        self._init_logger()
        logger.info("exporting the model")
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            logger.warning(f"mode for this stage {mode}")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        output_path = os.path.join(cfg.work_dir, "export")
        os.makedirs(output_path, exist_ok=True)
        model = build_segmentor(cfg.model)
        if model_ckpt:
            logger.info(f"model checkpoint load: {model_ckpt}")
            load_checkpoint(model=model, filename=model_ckpt, map_location="cpu")

        try:
            from torch.jit._trace import TracerWarning

            warnings.filterwarnings("ignore", category=TracerWarning)
            if torch.cuda.is_available():
                model = model.cuda(cfg.gpu_ids[0])
            else:
                model = model.cpu()
            precision = kwargs.pop("precision", "FP32")
            logger.info(f"Model will be exported with precision {precision}")

            export_model(
                model, cfg, output_path, target="openvino", output_logits=True, input_format="bgr", precision=precision
            )
        except Exception as ex:
            # output_model.model_status = ModelStatus.FAILED
            # raise RuntimeError('Optimization was unsuccessful.') from ex
            return {"outputs": None, "msg": f"exception {type(ex)}"}
        bin_file = [f for f in os.listdir(output_path) if f.endswith(".bin")][0]
        xml_file = [f for f in os.listdir(output_path) if f.endswith(".xml")][0]
        logger.info("Exporting completed")
        return {
            "outputs": {"bin": os.path.join(output_path, bin_file), "xml": os.path.join(output_path, xml_file)},
            "msg": "",
        }
