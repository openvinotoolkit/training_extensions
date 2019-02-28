# Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose

This repository contains training code for the paper [Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://arxiv.org/pdf/1811.12004.pdf). This work heavily optimizes the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) approach to reach real-time inference on CPU with negliable accuracy drop. It detects a skeleton (which consists of keypoints and connections between them) to identify human poses for every person inside the image. The pose may contain up to 18 keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles. On COCO 2017 Keypoint Detection validation set this code achives 40% AP for the single scale inference (no flip or any post-processing done). The result can be reproduced using this repository.

## Requirements

+ Ubuntu 16.04
+ Python 3.6
+ PyTorch 0.4.1 (should also work with 1.0, but not tested)

## Prerequisities

1. Download COCO 2017 dataset: [http://cocodataset.org/#download](http://cocodataset.org/#download) (train, val, annotations) and unpack it to `<COCO_HOME>` folder.
2. Create virtual environment `bash init_venv.sh`

## Training

Training consists of 3 steps (given AP values for full validation dataset):
* Training from MobileNet weights. Expected AP after this step is ~38%.
* Training from weights, obtained from previous step. Expected AP after this step is ~39%.
* Training from weights, obtained from previous step and increased number of refinement stages to 3 in network. Expected AP after this step is ~40% (for the network with 1 refinement stage, two next are discarded).

1. Download pre-trained MobileNet v1 weights `mobilenet_sgd_68.848.pth.tar` from: [https://github.com/marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet) (sgd option). To load this weights properly, change the 9-10 lines in `modules/load_state.py` to:

```
k = target_key
if k.find('model') != -1:
    k = k.replace('model', 'module.model')
if k in source_state and source_state[k].size() == target_state[target_key].size():
    new_target_state[target_key] = source_state[k]
```

2. Convert train annotations in internal format. Run `python scripts/prepare_train_labels.py --labels <COCO_HOME>/annotations/person_keypoints_train2017.json`. It will produce `prepared_train_annotation.pkl` with converted in internal format annotations.

   [OPTIONAL] For fast validation it is recommended to make *subset* of validation dataset. Run `python scripts/make_val_subset.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json`. It will produce `val_subset.json` with annotations just for 250 random images (out of 5000).

3. To train from MobileNet weights, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path mobilenet_sgd_68.848.pth.tar --weights-only`

4. Next, to train from checkpoint from previous step, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path-to>/checkpoint_iter_420000.pth.tar --weights-only`

5. Finally, to train from checkpoint from previous step and 3 refinement stages in network, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path-to>/checkpoint_iter_280000.pth.tar --weights-only --num-refinement-stages 3`. We took checkpoint after 370000 iterations as the final one.

We did not perform the best checkpoint selection at any step, so similar result may be achieved after less number of iterations.

#### Known issue

We observe this error with maximum number of open files (`ulimit -n`) equals to 1024:

```
  File "train.py", line 164, in <module>
    args.log_after, args.val_labels, args.val_images_folder, args.val_output_name, args.checkpoint_after, args.val_after)
  File "train.py", line 77, in train
    for _, batch_data in enumerate(train_loader):
  File "/<path>/python3.6/site-packages/torch/utils/data/dataloader.py", line 330, in __next__
    idx, batch = self._get_batch()
  File "/<path>/python3.6/site-packages/torch/utils/data/dataloader.py", line 309, in _get_batch
    return self.data_queue.get()
  File "/<path>/python3.6/multiprocessing/queues.py", line 337, in get
    return _ForkingPickler.loads(res)
  File "/<path>/python3.6/site-packages/torch/multiprocessing/reductions.py", line 151, in rebuild_storage_fd
    fd = df.detach()
  File "/<path>/python3.6/multiprocessing/resource_sharer.py", line 58, in detach
    return reduction.recv_handle(conn)
  File "/<path>/python3.6/multiprocessing/reduction.py", line 182, in recv_handle
    return recvfds(s, 1)[0]
  File "/<path>/python3.6/multiprocessing/reduction.py", line 161, in recvfds
    len(ancdata))
RuntimeError: received 0 items of ancdata
```

To get rid of it, increase the limit to bigger number, e.g. 65536, run in the terminal: `ulimit -n 65536`

## Validation

1. Run `python val.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json --images-folder <COCO_HOME>/val2017 --checkpoint-path <CHECKPOINT>`

## Pre-trained model

The model expects normalized image (mean=[128, 128, 128], scale=[1/256, 1/256, 1/256]) in planar BGR format.
Pre-trained on COCO model is available at: https://download.01.org/openvinotoolkit/open_model_zoo/training_toolbox_pytorch/checkpoint_iter_370000.pth.tar, it has 40% of AP on COCO validation set (38.6% of AP on the val *subset*).

#### Conversion to OpenVINO format

1. Convert PyTorch model to ONNX format: run script in terminal `python scripts/convert_to_onnx.py --checkpoint-path <CHECKPOINT>`. It produces `human-pose-estimation.onnx`.
2. Convert ONNX model to OpenVINO format with Model Optimizer: run in terminal `python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model human-pose-estimation.onnx --input data --mean_values data[128.0,128.0,128.0] --scale_values data[256] --output stage_1_output_0_pafs,stage_1_output_1_heatmaps`. This produces model `human-pose-estimation.xml` and weights `human-pose-estimation.bin` in single-precision floating-point format (FP32).

## Demo

To run the demo download Intel&reg; OpenVINO&trade; Toolkit [https://software.intel.com/en-us/openvino-toolkit/choose-download](https://software.intel.com/en-us/openvino-toolkit/choose-download), install it and [build the samples](https://software.intel.com/en-us/articles/OpenVINO-InferEngine) (*Inferring Your Model with the Inference Engine Samples* part). Then run `<SAMPLES_BIN_FOLDER>/human_pose_estimation_demo -m <path_to>/human-pose-estimation.xml -i <path_to_video_file>` for the inference on `CPU`.

## Fine-tuning

* The annotations have to be in [COCO format](http://cocodataset.org/#format-data).
* If number of keypoints differs from this model (18), then network structure and result parsing code (e.g. `modules/keypoints.py`) adjustments are required.
* The output order of keypoints is consistent to [OpenPose format](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25) (from 1 to 18).
* Fine-tuning steps are the same as steps 2-3 for training, just pass path to the pre-trained checkpoint and own data.

#### Example

We will perform fine-tuning on the first half of validation dataset and report the performance on the second part (see image indices in `data/val2017_2nd_half_indices.txt` for the reference) after 200 iterations (~6 epoches).

Validate before fine-tuning:
* `python val.py --labels val2017_2nd_part.json --output-name detections.json --images-folder <COCO_HOME>/val2017 --checkpoint-path https://download.01.org/openvinotoolkit/open_model_zoo/training_toolbox_pytorch/models/hpe/checkpoint_iter_370000.pth.tar`. AP before fine-tuning is: 39%.

Perform fine-tuning for 200 iterations:
* `python train.py --train-images-folder <COCO_HOME>/val2017/ --prepared-train-labels prepared_val2017_1st_part_annotation.pkl --val-labels val2017_2nd_part.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path https://download.01.org/openvinotoolkit/open_model_zoo/training_toolbox_pytorch/checkpoint_iter_370000.pth.tar --weights-only --checkpoint-after 200 --val-after 100 --log-after 12 --base-lr 0.00000444444`. Expected AP after 200 iterations of fine-tuning is: 39.1%-39.2%.


## Citation:

If this helps your research, please cite the paper:

```
@inproceedings{osokin2018lightweight_openpose,
    author={Osokin, Daniil},
    title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
    booktitle = {arXiv preprint arXiv:1811.12004},
    year = {2018}
}
```

