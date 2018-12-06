from collections import namedtuple
from os import path
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
from ssd_detector.readers.object_detector_json import ObjectDetectorJson

CURRENT_DIR = path.dirname(path.realpath(__file__))
ROOT_DIR = path.normpath(path.join(CURRENT_DIR, "../../../"))
MODEL_DIR = path.normpath(path.join(CURRENT_DIR, '../models'))

Train = namedtuple('Train',
                   'annotation_path,'
                   'batch_size,'
                   'cache_type,'
                   'fill_with_current_image_mean,'
                   'keep_checkpoint_every_n_hours,'
                   'max_steps,'
                   'random_seed,'
                   'save_checkpoints_steps,'
                   'save_summary_steps,'
                   'steps,'
                   'execution')

Eval = namedtuple('Eval',
                  'annotation_path,'
                  'batch_size,'
                  'datasets,'
                  'save_images_step,'
                  'vis_num,'
                  'execution')

Infer = namedtuple('Infer',
                   'batch_size,'
                   'out_subdir,'
                   'execution')

Execution = namedtuple('Execution',
                       'CUDA_VISIBLE_DEVICES,'
                       'intra_op_parallelism_threads,'
                       'inter_op_parallelism_threads,'
                       'transformer_parallel_calls,'
                       'transformer_prefetch_size')

# pylint: disable=invalid-name
train = Train(
  annotation_path=path.join(ROOT_DIR, "data/test/annotations_train.json"),
  batch_size=4,
  cache_type="NONE",
  fill_with_current_image_mean=True,
  keep_checkpoint_every_n_hours=6,
  max_steps=None,
  random_seed=666,
  save_checkpoints_steps=300,
  save_summary_steps=1,
  steps=300,
  execution=Execution(
    CUDA_VISIBLE_DEVICES="",  # Do not use GPU
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    transformer_parallel_calls=1,
    transformer_prefetch_size=1)
)

# pylint: disable=invalid-name
eval = Eval(
  annotation_path={
    "train": path.join(ROOT_DIR, "data/test/annotations_train.json"),
    "test": path.join(ROOT_DIR, "data/test/annotations_test.json")
  },
  batch_size=1,
  datasets=["train", "test"],
  save_images_step=1,
  vis_num=1,
  execution=Execution(
    CUDA_VISIBLE_DEVICES="",
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    transformer_parallel_calls=1,
    transformer_prefetch_size=1)
)


# pylint: disable=invalid-name
infer = Infer(
  batch_size=1,
  out_subdir="predictions",
  execution=Execution(
    CUDA_VISIBLE_DEVICES="",
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    transformer_parallel_calls=1,
    transformer_prefetch_size=1)
)

# pylint: disable=invalid-name
classes = ObjectDetectorJson.get_classes_from_coco_annotation(train.annotation_path)
input_shape = [128, 128, 3]


def optimizer(learning_rate):
  import tensorflow as tf
  return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)


# pylint: disable=invalid-name
detector_params = {
  "data_format": "NHWC",
  "depth_multiplier": 0.35,
  "initial_weights_path": path.join(ROOT_DIR, "data/test/model_ckpt/model.ckpt"),
  "learning_rate": 0.001,
  "mobilenet_version": "v2",
  "num_classes": len(classes),
  "optimizer": optimizer,
  "priors_rule": "object_detection_api",
  "collect_priors_summary": False
}
