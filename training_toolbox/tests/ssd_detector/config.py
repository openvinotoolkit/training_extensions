from os import path
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
from ssd_detector.readers.object_detector_json import ObjectDetectorJson

current_dir = path.dirname(path.realpath(__file__))
root_dir = path.normpath(path.join(current_dir, "../../../"))
model_dir = path.normpath(path.join(current_dir, '../models'))

class train:
  annotation_path = path.join(root_dir, "data/test/annotations_train.json")
  batch_size = 4
  cache_type = "ENCODED"
  fill_with_current_image_mean = True
  keep_checkpoint_every_n_hours = 6
  max_steps = None
  random_seed = 666
  save_checkpoints_steps = 300
  save_summary_steps = 1
  steps = 300

  class execution:
    CUDA_VISIBLE_DEVICES = ""  # Do not use GPU

    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 4
    transformer_parallel_calls = 1
    transformer_prefetch_size = 1



class eval:
  annotation_path = {
    "train": path.join(root_dir, "data/test/annotations_train.json"),
    "test": path.join(root_dir, "data/test/annotations_test.json")
  }
  batch_size = 1
  datasets = ["train", "test"]
  save_images_step = 1
  vis_num = 1

  class execution:
    CUDA_VISIBLE_DEVICES = ""

    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 4
    transformer_parallel_calls = 1
    transformer_prefetch_size = 1


class infer:
  batch_size = 1
  out_subdir = "predictions"

  class execution:
    CUDA_VISIBLE_DEVICES = ""

    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 4
    transformer_parallel_calls = 1
    transformer_prefetch_size = 1

classes = ObjectDetectorJson.get_classes_from_coco_annotation(train.annotation_path)
input_shape = [128, 128, 3]


def optimizer(learning_rate):
  import tensorflow as tf
  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
  return optimizer


detector_params = {
  "data_format": "NHWC",
  "depth_multiplier": 0.35,
  "initial_weights_path": path.join(root_dir, "data/test/model_ckpt/model.ckpt"),
  "learning_rate": 0.001,
  "mobilenet_version": "v2",
  "num_classes": len(classes),
  "optimizer": optimizer,
  "priors_rule": "object_detection_api",
  "collect_priors_summary": False
}

