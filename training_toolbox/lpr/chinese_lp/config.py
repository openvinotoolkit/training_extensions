import os
input_shape = (24, 94, 3)  # (height, width, channels)
use_h_concat = False
use_oi_concat = False

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'model')  # Path to the folder where all training and evaluation artifacts will be located
if not os.path.exists(model_dir):
  os.makedirs(model_dir)


class train:
  train_list_file_path = 'train' #path to annotation file with training data in per line format: <path_to_image_with_license_plate label>
  val_list_file_path =  'val'  #path to annotation file with validation data in per line format: <path_to_image_with_license_plate label>

  batch_size = 32
  val_batch_size = 128
  steps = 250000
  learning_rate = 0.001
  grad_noise_scale = 0.001
  opt_type = 'Adam'

  start_iter = 10000
  snap_iter = 100
  display_iter = 10
  val_iter = 100
  val_steps = 0  # 0 to run on full dataset

  rnn_cells_num = 128

  apply_basic_aug = False
  apply_stn_aug = True
  apply_blur_aug = False

  need_to_save_weights = True
  need_to_save_log = True

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


class eval:
  file_list_path = 'test' #path to annotation file with validation data in per line format: <path_to_image_with_license_plate label>
  checkpoint = ''
  max_lp_length = 20
  rnn_cells_num = 128

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations



class infer:
  file_list_path = 'test_infer' #path to text file with list of images in per line format: <path_to_image_with_license_plate>
  checkpoint = ''
  max_lp_length = 20
  rnn_cells_num = 128

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations
