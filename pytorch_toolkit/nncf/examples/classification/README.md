# Image classification sample
This sample demonstrates DL model compression in case of the image classification problem. The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs, training and validation steps. The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features:
- Torchvision models (ResNets, VGG, Inception, etc.) and datasets (ImageNet, CIFAR 10, CIFAR 100) support.
- Custom models support.
- Configuration file examples for sparsity, quantization, and quantization  with sparsity.
- Export to ONNX that is supported by OpenVINO toolkit.
- DataParallel and DistributedDataParallel modes.
- Tensorboard-compatible output.

## Quantize FP32 pre-trained model
This scenario demonstrates quantization with fine-tuning of MobileNet v2 on ImageNet dataset.

#### Dataset preparation
To prepare ImageNet dataset please refer to the following [tutorial](https://github.com/pytorch/examples/tree/master/imagenet).

#### Run classification sample
- If you did not install the package then add the repository root folder to the `PYTHONPATH` environment variable.
- Navigate to the `examples/classification` folder.
- Run the following command to start compression with fine-tuning on GPUs:
`python main.py -m train --config configs/quantization/mobilenetV2_imagenet_int8.json --data /data/imagenet/ --log-dir=../../results/quantization/mobilenetV2_int8/`
It may take a few epochs to get the baseline accuracy results.
- Use `--multiprocessing-distributed` flag to run in the distributed mode.
- Use `--resume` flag with the path to a previously saved model to resume training.

#### Validate your model checkpoint
To estimate the test scores of your model checkpoint use the following command:
`python main.py -m test --config=configs/quantization/mobilenetV2_imagenet_int8.json --resume <path_to_trained_model_checkpoint>`
If you want to validate an FP32 model checkpoint, make sure the compression algorithm settings are empty in the configuration file or `pretrained=True` is set.

#### Export compressed model
To export trained model to ONNX format use the following command:
`python main.py -m test --config=configs/quantization/mobilenetV2_imagenet_int8.json --resume=../../results/quantization/mobilenetV2_int8/6/checkpoints/epoch_1.pth --to-onnx=../../results/mobilenetV2_int8.onnx`

#### Export to OpenVINO Intermediate Representation (IR)

To export a model to OpenVINO IR and run it using Intel Deep Learning Deployment Toolkit please refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

### Results for INT8 quantization

| Model | Dataset | FP32 baseline | Compressed model accuracy | Config path | Checkpoint |
| :-- | :-: | :-: | :-: | :-: | :-: |
| ResNet-50 INT8 quantized | ImageNet | 76.13 | 76.49 | examples/classification/configs/quantization/resnet50_imagenet_int8.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_int8.pth) |
| ResNet-50 INT8 w/ 60% of sparsity (RB) | ImageNet | 76.13 | 75.2 | examples/classification/configs/sparsity_quantization/resnet50_imagenet_sparsity_int8.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_sparsity_int8.pth) |
| Inception v3 INT8 quantized | ImageNet | 77.32 | 78.36 | examples/classification/configs/quantization/inceptionV3_imagenet_int8.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/inception_v3_imagenet_int8.pth) |
| Inception v3 INT8 w/ 60% of sparsity (RB) | ImageNet | 77.32 | 77.05 | examples/classification/configs/sparsity_quantization/inceptionV3_imagenet_sparsity_int8.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/inceptionV3_imagenet_sparsity_int8.pth) |
| MobileNet v2 INT8 quantized | ImageNet | 71.8 | 71.33 | examples/classification/configs/quantization/mobilenetV2_imagenet_int8.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/mobilenetv2_imagenet_int8.pth) |
| MobileNet v2 INT8 w/ 51% of sparsity (RB) | ImageNet | 71.8 | 70.84 | examples/classification/configs/sparsity_quantization/mobilenetV2_imagenet_sparsity_int8.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/mobilenetv2_imagenet_sparse_int8.pth) |
| SqueezeNet v1.1 INT8 quantized | ImageNet | 58.19 | 58.16 | examples/classification/configs/quantization/squeezenet1_1_imagenet_int8.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/squeezenet1_1_imagenet_int8.pth) |


#### Binarization

As an example of NNCF convolution binarization capabilities, you may use the configs in `examples/classification/configs/binarization` to binarize ResNet18. Use the same steps/command line parameters as for quantization (for best results, specify `--pretrained`), except for the actual binarization config path.


### Results for binarization
| Model | Weight binarization type | Activation binarization type | Dataset | FP32 baseline | Compressed model accuracy | Config path | Checkpoint |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ResNet-18 | XNOR | Scale/threshold | ImageNet | 69.75 | 61.71 | examples/classification/configs/binarization/resnet18_imagenet_bin_xnor.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet18_imagenet_binarization_xnor.pth) |
| ResNet-18 | DoReFa | Scale/threshold | ImageNet | 69.75 | 61.58 | examples/classification/configs/binarization/resnet18_imagenet_bin_dorefa.json |  [Link](https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet18_imagenet_binarization_dorefa.pth) |
