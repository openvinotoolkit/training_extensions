# Object Detection sample
This sample demonstrates DL model compression capabailites for object detection task.

## Features:
- Vanilla SSD300 / SSD512 (+ Batch Normalization), MobileNetSSD-300
- VOC2007 / VOC2012, COCO datasets
- Configuration file examples for sparsity and quantization
- Export to ONNX compatible with OpenVINO (compatible with pre-shipped CPU extensions detection layers)
- DataParallel and DistributedDataParallel modes
- Tensorboard output

## Quantize FP32 pretrained model
This scenario demonstrates quantization with fine-tuning of SSD300 on VOC dataset.

#### Dataset preparation
- Download and extract VOC2007 and VOC2012 train/val and test data + devkit from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) and [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit)

#### Run object detection sample
- If you did not install the package then add the repository root folder to the `PYTHONPATH` environment variable
- Navigate to the `examples/object_detection` folder
- Run the following command to start compression with fine-tuning on GPUs:
`python main.py -m train --config configs/ssd300_vgg_int8_voc.json --data <path_to_dataset> --log-dir=../../results/quantization/ssd300_int8`
It may take a few epochs to get the baseline accuracy results.
- Use `--multiprocessing-distributed` flag to run in the distributed mode.
- Use `--resume` flag with the path to a previously saved model to resume training.

#### Validate your model checkpoint
To estimate the test scores of your model checkpoint use the following command:
`python main.py -m test --config=configs/ssd300_vgg_int8_voc.json --data <path_to_dataset> --resume <path_to_trained_model_checkpoint>`
If you want to validate an FP32 model checkpoint, make sure the compression algorithm settings are empty in the configuration file or `pretrained=True` is set.

#### Export compressed model
To export trained model to ONNX format use the following command:
`python main.py -m test --config configs/ssd300_vgg_int8_voc.json --data <path_to_dataset> --resume <path_to_compressed_model_checkpoint> --to-onnx=../../results/ssd300_int8.onnx`

#### Export to OpenVINO Intermediate Representation (IR)

To export a model to OpenVINO IR and run it using Intel Deep Learning Deployment Toolkit please refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

### Results

|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|Config path|PyTorch Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|SSD300-BN|None|VOC12+07|78.28|examples/object_detection/configs/ssd300_vgg_voc.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/ssd300_vgg_voc.pth|
|SSD300-BN|INT8|VOC12+07|78.07|examples/object_detection/configs/ssd300_vgg_voc_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/ssd300_vgg_voc_int8.pth|
|SSD300-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07|78.01|examples/object_detection/configs/ssd300_vgg_voc_magnitude_sparsity_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/ssd300_vgg_voc_magnitude_sparsity_int8.pth|
|SSD512-BN|None|VOC12+07|80.26|examples/object_detection/configs/ssd512_vgg_voc.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/ssd512_vgg_voc.pth|
|SSD512-BN|INT8|VOC12+07|80.02|examples/object_detection/configs/ssd512_vgg_voc_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/ssd512_vgg_voc_int8.pth|
|SSD512-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07|79.98|examples/object_detection/configs/ssd512_vgg_voc_magnitude_sparsity_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/ssd512_vgg_voc_magnitude_sparsity_int8.pth|
