"""Task of OTX Detection using mmdetection training backend."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from contextlib import nullcontext
from typing import List, Optional, Union

from mmcv.utils import Config, ConfigDict, get_git_hash
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmrotate import __version__

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    BaseRecordingForwardHook,
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmcv.utils import build_data_parallel
from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.datasets import ImageTilingDataset
from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.subset import Subset

logger = get_logger()


class MMRotateTask(MMDetectionTask):
    """Task of OTX Detection using mmrotate training backend."""

    def record_info_to_checkpoint_meta(self, cfg: Config, classes: List[str]):
        """Record info to checkpoint meta.

        Args:
            cfg (Config): detection configuration
            classes (list): list of dataset classes
        """
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmrotate_version=__version__ + get_git_hash()[:7],
                CLASSES=classes,
            )

    # def _infer_model(
    #     self,
    #     dataset: DatasetEntity,
    #     inference_parameters: Optional[InferenceParameters] = None,
    # ):
    #     """Main infer function."""
    #     original_subset = dataset[0].subset
    #     for item in dataset:
    #         item.subset = Subset.TESTING
    #     self._data_cfg = ConfigDict(
    #         data=ConfigDict(
    #             train=ConfigDict(
    #                 otx_dataset=None,
    #                 labels=self._labels,
    #             ),
    #             test=ConfigDict(
    #                 otx_dataset=dataset,
    #                 labels=self._labels,
    #             ),
    #         )
    #     )

    #     dump_features = True
    #     dump_saliency_map = not inference_parameters.is_evaluation if inference_parameters else True

    #     self._init_task(dataset)

    #     cfg = self.configure(False, None)
    #     logger.info("infer!")

    #     # Data loader
    #     mm_dataset = build_dataset(cfg.data.test)
    #     samples_per_gpu = cfg.data.test_dataloader.get("samples_per_gpu", 1)
    #     # If the batch size and the number of data are not divisible, the metric may score differently.
    #     # To avoid this, use 1 if they are not divisible.
    #     samples_per_gpu = samples_per_gpu if len(mm_dataset) % samples_per_gpu == 0 else 1
    #     dataloader = build_dataloader(
    #         mm_dataset,
    #         samples_per_gpu=samples_per_gpu,
    #         workers_per_gpu=cfg.data.test_dataloader.get("workers_per_gpu", 0),
    #         num_gpus=len(cfg.gpu_ids),
    #         dist=cfg.distributed,
    #         seed=cfg.get("seed", None),
    #         shuffle=False,
    #     )

    #     # Target classes
    #     if "task_adapt" in cfg:
    #         target_classes = cfg.task_adapt.final
    #         if len(target_classes) < 1:
    #             raise KeyError(
    #                 f"target_classes={target_classes} is empty check the metadata from model ckpt or recipe "
    #                 "configuration"
    #             )
    #     else:
    #         target_classes = mm_dataset.CLASSES

    #     # Model
    #     model = self.build_model(cfg, fp16=cfg.get("fp16", False))
    #     model.CLASSES = target_classes
    #     model.eval()
    #     feature_model = model.model_t if self._train_type == TrainType.Semisupervised else model
    #     model = build_data_parallel(model, cfg, distributed=False)

    #     # InferenceProgressCallback (Time Monitor enable into Infer task)
    #     time_monitor = None
    #     if cfg.get("custom_hooks", None):
    #         time_monitor = [hook.time_monitor for hook in cfg.custom_hooks if hook.type == "OTXProgressHook"]
    #         time_monitor = time_monitor[0] if time_monitor else None
    #     if time_monitor is not None:
    #         # pylint: disable=unused-argument
    #         def pre_hook(module, inp):
    #             time_monitor.on_test_batch_begin(None, None)

    #         def hook(module, inp, outp):
    #             time_monitor.on_test_batch_end(None, None)

    #         model.register_forward_pre_hook(pre_hook)
    #         model.register_forward_hook(hook)

    #     # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
    #     # TODO[EUGENE]: FIGURE OUT HOW DetClassProbabilityMapHook WORKS
    #     if not dump_saliency_map:
    #         saliency_hook: Union[nullcontext, BaseRecordingForwardHook] = nullcontext()
    #     else:
    #         raw_model = feature_model
    #         if raw_model.__class__.__name__ == "NNCFNetwork":
    #             raw_model = raw_model.get_nncf_wrapped_model()
    #         saliency_hook = ActivationMapHook(feature_model)

    #     if not dump_features:
    #         feature_vector_hook: Union[nullcontext, BaseRecordingForwardHook] = nullcontext()
    #     else:
    #         feature_vector_hook = FeatureVectorHook(feature_model)

    #     eval_predictions = []
    #     # pylint: disable=no-member
    #     with feature_vector_hook:
    #         with saliency_hook:
    #             eval_predictions = single_gpu_test(model, dataloader)
    #             if isinstance(feature_vector_hook, nullcontext):
    #                 feature_vectors = [None] * len(mm_dataset)
    #             else:
    #                 feature_vectors = feature_vector_hook.records
    #             if isinstance(saliency_hook, nullcontext):
    #                 saliency_maps = [None] * len(mm_dataset)
    #             else:
    #                 saliency_maps = saliency_hook.records

    #     for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
    #         cfg.evaluation.pop(key, None)

    #     # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
    #     while hasattr(mm_dataset, "dataset") and not isinstance(mm_dataset, ImageTilingDataset):
    #         mm_dataset = mm_dataset.dataset

    #     if isinstance(mm_dataset, ImageTilingDataset):
    #         feature_vectors = [feature_vectors[i] for i in range(mm_dataset.num_samples)]
    #         saliency_maps = [saliency_maps[i] for i in range(mm_dataset.num_samples)]
    #         eval_predictions = mm_dataset.merge(eval_predictions)

    #     metric = None
    #     if inference_parameters and inference_parameters.is_evaluation:
    #         if isinstance(mm_dataset, ImageTilingDataset):
    #             metric = mm_dataset.dataset.evaluate(eval_predictions, **cfg.evaluation)
    #         else:
    #             metric = mm_dataset.evaluate(eval_predictions, **cfg.evaluation)
    #         metric = metric["mAP"] if isinstance(cfg.evaluation.metric, list) else metric[cfg.evaluation.metric]

    #     assert len(eval_predictions) == len(feature_vectors) == len(saliency_maps), (
    #         "Number of elements should be the same, however, number of outputs are "
    #         f"{len(eval_predictions)}, {len(feature_vectors)}, and {len(saliency_maps)}"
    #     )

    #     results = dict(
    #         outputs=dict(
    #             classes=target_classes,
    #             detections=eval_predictions,
    #             metric=metric,
    #             feature_vectors=feature_vectors,
    #             saliency_maps=saliency_maps,
    #         )
    #     )

    #     # TODO: InferenceProgressCallback register
    #     output = results["outputs"]
    #     metric = output["metric"]
    #     predictions = output["detections"]
    #     assert len(output["detections"]) == len(output["feature_vectors"]) == len(output["saliency_maps"]), (
    #         "Number of elements should be the same, however, number of outputs are "
    #         f"{len(output['detections'])}, {len(output['feature_vectors'])}, and {len(output['saliency_maps'])}"
    #     )
    #     prediction_results = zip(predictions, output["feature_vectors"], output["saliency_maps"])
    #     # FIXME. This is temporary solution.
    #     # All task(e.g. classification, segmentation) should change item's type to Subset.TESTING
    #     # when the phase is inference.
    #     for item in dataset:
    #         item.subset = original_subset
    #     return prediction_results, metric
