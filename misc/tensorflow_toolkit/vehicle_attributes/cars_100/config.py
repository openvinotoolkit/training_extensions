# Copyright (C) 2019 Intel Corporation
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

input_shape = (72, 72, 3)  # (height, width, channels)
model_dir = 'model'

class train:
  batch_size = 32
  steps = 2000000

  random_seed = 666

  save_checkpoints_steps = 1000      # Number of training steps when checkpoint should be saved
  keep_checkpoint_every_n_hours = 1  # Checkpoint should be saved forever after every n hours
  save_summary_steps = 100           # Number of steps when the summary information should be saved

  num_parallel_calls = 4
  prefetch_size = 4

  annotation_path = '../../../data/cars_100/cars_100_train.json'
  use_pretrained_weights = True
  pretrained_ckpt = 'vehicle-attributes-barrier-0103/model.ckpt-2000000'

  class execution:
    CUDA_VISIBLE_DEVICES = "0"
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations

    intra_op_parallelism_threads = 2
    inter_op_parallelism_threads = 8
    transformer_parallel_calls = 4  # Number of parallel threads in data transformer/augmentation
    transformer_prefetch_size = 8   # Number of batches to prefetch

class eval:
  batch_size = 32

  annotation_path = '../../../data/cars_100/cars_100_test.json'

  class execution:
    CUDA_VISIBLE_DEVICES = "0"
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations

    intra_op_parallelism_threads = 2
    inter_op_parallelism_threads = 8
    transformer_parallel_calls = 4  # Number of parallel threads in data transformer/augmentation
    transformer_prefetch_size = 8   # Number of batches to prefetch

class infer:
  annotation_path = '../../../data/cars_100/cars_100_test.json'

  class execution:
    CUDA_VISIBLE_DEVICES = "0"
    intra_op_parallelism_threads = 0

def optimizer(learning_rate):
  import tensorflow as tf
  return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)

resnet_params = {
  "learning_rate": 0.001,                                 # Learning rate
  "optimizer": optimizer,                                 # Optimizer
  "pretrained_ckpt": train.pretrained_ckpt,               # Trained model
  "use_pretrained_weights": train.use_pretrained_weights  # Use pretrained model weights
}
