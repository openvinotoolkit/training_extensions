import os
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
from ssd_detector.readers.object_detector_json import ObjectDetectorJson

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "../../.."))

# See more details about parameters in TensorFlow documentation tf.estimator
class train:
  annotation_path = "instances_train2017.json"  # Path to the annotation file
  cache_type = "ENCODED"  # Type of data to save in memory, possible options: 'FULL', 'ENCODED', 'NONE'

  batch_size = 32                    # Number of images in the batch
  steps = 50000000                   # Number of steps for which to train model
  max_steps = None                   # Number of total steps for which to train model
  save_checkpoints_steps = 1000      # Number of training steps when checkpoint should be saved
  keep_checkpoint_every_n_hours = 6  # Checkpoint should be saved forever after every n hours
  save_summary_steps = 100           # Number of steps when the summary information should be saved
  random_seed = 666                  # Random seed

  fill_with_current_image_mean = False  # Parameter of data transformer

  class execution:
    CUDA_VISIBLE_DEVICES = "0"             # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True                    # Option which attempts to allocate only as much GPU memory based on runtime allocations

    intra_op_parallelism_threads = 2
    inter_op_parallelism_threads = 8
    transformer_parallel_calls = 4  # Number of parallel threads in data transformer/augmentation
    transformer_prefetch_size = 8   # Number of batches to prefetch


class eval:
  annotation_path = {
    "test": os.path.join(CURRENT_DIR, "instances_val2017.json")
  }  # Dictionary with paths to annotations and its short names which will be displayed in the TensorBoard
  datasets = ["test"]  # List of names from annotation_path dictionary on which evaluation will be launched
  vis_num = 12                  # Select random images for visualization in the TensorBoard
  save_images_step = 2          # Save images every 2-th evaluation
  batch_size = 8                # Number of images in the batch

  class execution:
    CUDA_VISIBLE_DEVICES = "0"             # Environment variable to control CUDA device used for evaluation
    per_process_gpu_memory_fraction = 0.5  # Fix extra memory allocation issue
    allow_growth = True                    # Option which attempts to allocate only as much GPU memory based on runtime allocations

    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 1
    transformer_parallel_calls = 1  # Number of parallel threads in data transformer/augmentation
    transformer_prefetch_size = 1   # Number of batches to prefetch


class infer:
  out_subdir = "predictions"  # Name of folder in model directory where output json files with detections will be saved
  batch_size = 32             # Number of images in the batch

  class execution:
    CUDA_VISIBLE_DEVICES = "0"             # Environment variable to control cuda device used for training
    per_process_gpu_memory_fraction = 0.5  # Fix extra memory allocation issue
    allow_growth = True                    # Option which attempts to allocate only as much GPU memory based on runtime allocations

    intra_op_parallelism_threads = 2
    inter_op_parallelism_threads = 8
    transformer_parallel_calls = 4  # Number of parallel threads in data transformer/augmentation
    transformer_prefetch_size = 8   # Number of batches to prefetch


input_shape = (256, 256, 3)  # Input shape of the model (width, height, channels)
classes = ObjectDetectorJson.get_classes_from_coco_annotation(os.path.join(CURRENT_DIR, train.annotation_path))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'model')  # Path to the folder where all training and evaluation artifacts will be located
if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR)


def learning_rate_schedule():  # Function which controls learning rate during training
  import tensorflow as tf
  return tf.train.exponential_decay(
    learning_rate=0.004,
    global_step=tf.train.get_or_create_global_step(),
    decay_steps=800000,
    decay_rate=0.95,
    staircase=True)


def optimizer(learning_rate):
  import tensorflow as tf
  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, decay=0.9, epsilon=1.0)
  return optimizer


detector_params = {
  "num_classes": len(classes),  # Number of classes to detect
  "priors_rule": "object_detection_api",    # Prior boxes rule for SSD, possible options: 'caffe', 'object_detection_api', 'custom'
  "mobilenet_version": "v2",                # Version of mobilenet backbone, possible options: 'v1', 'v2'
  "initial_weights_path": "",               # Path to initial weights
  "depth_multiplier": 1.0,                  # MobileNet channels multiplier
  "weight_regularization": 4e-5,            # L2 weight regularization
  "learning_rate": learning_rate_schedule,  # Learning rate
  "optimizer": optimizer,                   # Optimizer
  "collect_priors_summary": False,          # Option to collect priors summary for further analysis
}
