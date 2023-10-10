from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.algorithms.common.adapters.mmcv.utils import (
    remove_from_configs_by_type,
    update_config,
)

logger = get_logger()

#NOTE,
# For the current models, each task showed below characteristics (roughly)
#   *Batch size
#       Classification: 64, Detection: 8, Instance seg: 4, Semantic seg 8
#   *Iter time
#       Classification: 0.15s, Detection: 0.35s, Instance seg: 0.45s, Semantic seg 0.25s
# So, TASK_TO_SIMPLE_CONFIG is set to end the training within 5 min.

TASK_TO_RUNNER_CONFIG={
    "classification":{
        "iters_per_epoch": 75,
        "max_epochs": 25
    },
    "detection":{
        "iters_per_epoch": 30,
        "max_epochs": 28
    },
    "instance_segmentation":{
        "iters_per_epoch": 60,
        "max_epochs": 11
    },
    "semantic_segmentation":{
        "iters_per_epoch": 30,
        "max_epochs": 40
    },
}

def sample_dataset(dataset, iters_per_epoch, batch_size):
    sampled_number = iters_per_epoch * batch_size
    sampled_dataset = dataset[:sampled_number]
    return DatasetEntity(
        items=sampled_dataset
    )

def enable_simple_incr_learning(cfg, task):
    logger.info("Simple incremental model is enabled.")
    logger.info("- If there is AdaptiveRepeatDataHook, it will be disabled.")
    remove_from_configs_by_type(cfg.custom_hooks, "AdaptiveRepeatDataHook")
    batch_size = cfg.data.train_dataloader.samples_per_gpu
    
    dataset = cfg.data.train.otx_dataset 
    dataset_len = len(dataset)
    current_iters_per_epoch = dataset_len // batch_size
     
    task_config = TASK_TO_RUNNER_CONFIG[task]
    preset_iters_per_epoch = task_config["iters_per_epoch"]
    preset_max_epochs = task_config["max_epochs"]
    
    if current_iters_per_epoch > preset_iters_per_epoch: 
        logger.info(
            "- Iterations per epoch will be changed, "
            f"{current_iters_per_epoch} -> {preset_iters_per_epoch}"
        )
        sampled_dataset = sample_dataset(dataset, preset_iters_per_epoch, batch_size)
        cfg.data.train.otx_dataset = sampled_dataset
    else:
        logger.info(
            "- Iterations per epoch will not be changed, " 
            f"current: {current_iters_per_epoch}"
        )
    
    logger.info(
        "- Max epochs will be changed, "
        f"{cfg.runner.max_epochs} -> {preset_max_epochs}"
    )
    cfg.runner.max_epochs = preset_max_epochs
    