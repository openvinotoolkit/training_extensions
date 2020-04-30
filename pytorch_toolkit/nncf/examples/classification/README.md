# Image Classification Sample

This sample demonstrates a DL model compression in case of an image-classification problem. The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs, training and validation steps. The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features

- Torchvision models (ResNets, VGG, Inception, etc.) and datasets (ImageNet, CIFAR 10, CIFAR 100) support
- Custom models support
- Configuration file examples for sparsity, quantization, and quantization with sparsity
- Export to ONNX that is supported by the OpenVINO™ toolkit
- DataParallel and DistributedDataParallel modes
- Tensorboard-compatible output

## Quantize FP32 Pretrained Model

This scenario demonstrates quantization with fine-tuning of MobileNet v2 on the ImageNet dataset.

#### Dataset Preparation

To prepare the ImageNet dataset, refer to the following [tutorial](https://github.com/pytorch/examples/tree/master/imagenet).

#### Run Classification Sample

- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/classification` folder.
- Run the following command to start compression with fine-tuning on GPUs:
    ```
    python main.py -m train --config configs/quantization/mobilenet_v2_imagenet_int8.json --data /data/imagenet/ --log-dir=../../results/quantization/mobilenet_v2_int8/
    ```
    It may take a few epochs to get the baseline accuracy results.
- Use the `--multiprocessing-distributed` flag to run in the distributed mode.
- Use the `--resume` flag with the path to a previously saved model to resume training.

#### Validate Your Model Checkpoint

To estimate the test scores of your model checkpoint, use the following command:
```
python main.py -m test --config=configs/quantization/mobilenet_v2_imagenet_int8.json --resume <path_to_trained_model_checkpoint>
```
To validate an FP32 model checkpoint, make sure the compression algorithm settings are empty in the configuration file or `pretrained=True` is set.

#### Export Compressed Model

To export trained model to the ONNX format, use the following command:
```
python main.py -m test --config=configs/quantization/mobilenet_v2_imagenet_int8.json --resume=../../results/quantization/mobilenet_v2_int8/6/checkpoints/epoch_1.pth --to-onnx=../../results/mobilenet_v2_int8.onnx
```

#### Export to OpenVINO™ Intermediate Representation (IR)

To export a model to the OpenVINO IR and run it using the Intel® Deep Learning Deployment Toolkit, refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

### Results for quantization

|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|Config path|PyTorch Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|76.13|examples/classification/configs/quantization/resnet50_imagenet.json|-|
|ResNet-50|INT8|ImageNet|76.05|examples/classification/configs/quantization/resnet50_imagenet_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_int8.pth|
|ResNet-50|Mixed, 44.8% INT8 / 55.2% INT4|ImageNet|76.3|examples/classification/configs/quantization/resnet50_imagenet_mixed_int_hawq.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_int4_int8.pth|
|ResNet-50|INT8 + Sparsity 61% (RB)|ImageNet|75.28|examples/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_rb_sparsity_int8.pth|
|ResNet-50|Filter pruning, 30%, magnitude criterion|ImageNet|75.7|examples/classification/configs/pruning/resnet50_pruning_magnitude.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_filter_pruning_magnitude.pth|
|ResNet-50|Filter pruning, 30%, geometric median criterion|ImageNet|75.7|examples/classification/configs/pruning/resnet50_pruning_geometric_median.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_filter_pruning_geomean.pth|
|Inception V3|None|ImageNet|77.32|examples/classification/configs/quantization/inception_v3_imagenet.json|-|
|Inception V3|INT8|ImageNet|76.92|examples/classification/configs/quantization/inception_v3_imagenet_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/inception_v3_imagenet_int8.pth|
|Inception V3|INT8 + Sparsity 61% (RB)|ImageNet|76.98|examples/classification/configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/inception_v3_imagenet_rb_sparsity_int8.pth|
|MobileNet V2|None|ImageNet|71.81|examples/classification/configs/quantization/mobilenet_v2_imagenet.json|-|
|MobileNet V2|INT8|ImageNet|71.34|examples/classification/configs/quantization/mobilenet_v2_imagenet_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/mobilenet_v2_imagenet_int8.pth|
|MobileNet V2|Mixed, 46.6% INT8 / 53.4% INT4|ImageNet|70.89|examples/classification/configs/quantization/mobilenet_v2_imagenet_mixed_int_hawq.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/mobilenet_v2_imagenet_int4_int8.pth|
|MobileNet V2|INT8 + Sparsity 52% (RB)|ImageNet|70.99|examples/classification/configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/mobilenet_v2_imagenet_rb_sparsity_int8.pth|
|SqueezeNet V1.1|None|ImageNet|58.18|examples/classification/configs/quantization/squeezenet1_1_imagenet.json|-|
|SqueezeNet V1.1|INT8|ImageNet|58.02|examples/classification/configs/quantization/squeezenet1_1_imagenet_int8.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/squeezenet1_1_imagenet_int8.pth|
|SqueezeNet V1.1|Mixed, 54.7% INT8 / 45.3% INT4|ImageNet|58.84|examples/classification/configs/quantization/squeezenet1_1_imagenet_mixed_int_hawq.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/squeezenet1_1_imagenet_int4_int8.pth|
|ResNet-18|None|ImageNet|69.76|examples/classification/configs/binarization/resnet18_imagenet.json|-|
|ResNet-18|Filter pruning, 30%, magnitude criterion|ImageNet|68.69|examples/classification/configs/pruning/resnet18_pruning_magnitude.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet18_imagenet_filter_pruning_magnitude.pth|
|ResNet-18|Filter pruning, 30%, geometric median criterion|ImageNet|68.97|examples/classification/configs/pruning/resnet18_pruning_geometric_median.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet18_imagenet_filter_pruning_geomean.pth|
|ResNet-34|None|ImageNet|73.31|examples/classification/configs/pruning/resnet34_imagenet.json|-|
|ResNet-34|Filter pruning, 30%, magnitude criterion|ImageNet|72.54|examples/classification/configs/pruning/resnet34_pruning_magnitude.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet34_imagenet_filter_pruning_magnitude.pth|
|ResNet-34|Filter pruning, 30%, geometric median criterion|ImageNet|72.60|examples/classification/configs/pruning/resnet34_pruning_geometric_median.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet34_imagenet_filter_pruning_geomean.pth|


#### Binarization

As an example of NNCF convolution binarization capabilities, you may use the configs in `examples/classification/configs/binarization` to binarize ResNet18. Use the same steps/command line parameters as for quantization (for best results, specify `--pretrained`), except for the actual binarization config path.

### Results for binarization
|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|Config path|PyTorch Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-18|None|ImageNet|69.76|examples/classification/configs/binarization/resnet18_imagenet.json|-|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|61.61|examples/classification/configs/binarization/resnet18_imagenet_binarization_xnor.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet18_imagenet_binarization_xnor.pth|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|61.59|examples/classification/configs/binarization/resnet18_imagenet_binarization_dorefa.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet18_imagenet_binarization_dorefa.pth|


### Results for filter pruning
|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|Config path|PyTorch Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|76.13|examples/classification/configs/quantization/resnet50_imagenet.json|-|
|ResNet-50|Filter pruning, 30%, magnitude criterion|ImageNet|75.7|examples/classification/configs/pruning/resnet50_pruning_magnitude.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_filter_pruning_magnitude.pth|
|ResNet-50|Filter pruning, 30%, geometric median criterion|ImageNet|75.7|examples/classification/configs/pruning/resnet50_pruning_geometric_median.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet50_imagenet_filter_pruning_geomean.pth|
|ResNet-18|None|ImageNet|69.76|examples/classification/configs/binarization/resnet18_imagenet.json|-|
|ResNet-18|Filter pruning, 30%, magnitude criterion|ImageNet|68.69|examples/classification/configs/pruning/resnet18_pruning_magnitude.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet18_imagenet_filter_pruning_magnitude.pth|
|ResNet-18|Filter pruning, 30%, geometric median criterion|ImageNet|68.97|examples/classification/configs/pruning/resnet18_pruning_geometric_median.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet18_imagenet_filter_pruning_geomean.pth|
|ResNet-34|None|ImageNet|73.31|examples/classification/configs/pruning/resnet34_imagenet.json|-|
|ResNet-34|Filter pruning, 30%, magnitude criterion|ImageNet|72.54|examples/classification/configs/pruning/resnet34_pruning_magnitude.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet34_imagenet_filter_pruning_magnitude.pth|
|ResNet-34|Filter pruning, 30%, geometric median criterion|ImageNet|72.60|examples/classification/configs/pruning/resnet34_pruning_geometric_median.json|https://download.01.org/opencv/openvino_training_extensions/models/nncf/resnet34_imagenet_filter_pruning_geomean.pth|

