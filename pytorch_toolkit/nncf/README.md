# Neural Network Compression Framework (NNCF)

This module contains a PyTorch\*-based framework and samples for neural networks compression. The framework is organized as a Python\* package that can be built and used in a standalone mode. The framework architecture is unified to make it easy to add different compression methods. The samples demonstrate the usage of compression algorithms for three different use cases on public models and datasets: Image Classification, Object Detection and Semantic Segmentation.

## Key Features

- Support of various compression algorithms, applied during a model fine-tuning process to achieve best compression parameters and accuracy:
    - [Quantization](./docs/compression_algorithms/Quantization.md)
    - [Binarization](./docs/compression_algorithms/Binarization.md)
    - [Sparsity](./docs/compression_algorithms/Sparsity.md)
    - [Filter pruning](./docs/compression_algorithms/Pruning.md)
- Automatic, configurable model graph transformation to obtain the compressed model. The source model is wrapped by the custom class and additional compression-specific layers are inserted in the graph.
- Common interface for compression methods
- GPU-accelerated layers for faster compressed model fine-tuning
- Distributed training support
- Configuration file examples for each supported compression algorithm.
- Git patches for prominent third-party repositories ([mmdetection](https://github.com/open-mmlab/mmdetection), [huggingface-transformers](https://github.com/huggingface/transformers)) demonstrating the process of integrating NNCF into custom training pipelines
- Exporting compressed models to ONNX\* checkpoints ready for usage with [OpenVINO&trade; toolkit](https://github.com/opencv/dldt).

## Usage
The NNCF is organized as a regular Python package that can be imported in your target training pipeline script.
The basic workflow is loading a JSON configuration script containing NNCF-specific parameters determining the compression to be applied to your model, and then passing your model along with the configuration script to the `nncf.create_compressed_model` function.
This function returns a wrapped model ready for compression fine-tuning, and handle to the object allowing you to control the compression during the training process:

```python
import nncf
from nncf import create_compressed_model, Config as NNCFConfig

# Instantiate your uncompressed model
from torchvision.models.resnet import resnet50
model = resnet50()

# Apply compression according to a loaded NNCF config
nncf_config = NNCFConfig.from_json("resnet50_int8.json")
comp_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Now use compressed_model as a usual torch.nn.Module

# ... the rest of the usual PyTorch-powered training pipeline

# Export to ONNX or .pth when done fine-tuning
comp_ctrl.export_model("compressed_model.onnx")
torch.save(compressed_model.state_dict(), "compressed_model.pth")
```

For a more detailed description of NNCF usage in your training code, see [Usage.md](./docs/Usage.md). For in-depth examples of NNCF integration, browse the [sample scripts](#Model Compression Samples) code, or the [example patches](#Third-party repository integration) to third-party repositories.

For more details about the framework architecture, refer to the [NNCFArchitecture.md](./docs/NNCFArchitecture.md).


### Model Compression Samples

For a quicker start with NNCF-powered compression, you can also try the sample scripts, each of which provides a basic training pipeline for classification, semantic segmentation and object detection neural network training correspondingly.

To run the samples please refer to the corresponding tutorials:
- [Image Classification sample](examples/classification/README.md)
- [Object Detection sample](examples/object_detection/README.md)
- [Semantic Segmentation sample](examples/semantic_segmentation/README.md)

### Third-party repository integration
NNCF may be straightforwardly integrated into training/evaluation pipelines of third-party repositories.
See [third_party_integration](./third_party_integration) for examples of code modifications (Git patches and base commit IDs are provided) that are necessary to integrate NNCF into select repositories.


### System requirements
- Ubuntu\* 16.04 or later (64-bit)
- Python\* 3.5 or later
- NVidia CUDA\* Toolkit 10.1 or later
- PyTorch\* 1.3.1 or higher.

### Installation
We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).

#### As a package built from checked-out repository:
1) Install the following system dependencies:

`sudo apt-get install python3-dev`

2) Install the package and its dependencies by running the following in the repository root directory:

- For CPU & GPU-powered execution:
`pip install -r requirements.txt`
or
`python setup.py`
- For CPU-only installation
`python setup.py --cpu-only`

#### As a Docker image
Use one of the Dockerfiles in the [docker](./docker) directory to build an image with an environment already set up and ready for running NNCF [sample scripts](#Model Compression Samples).


## NNCF compression results

Achieved using sample scripts and NNCF configuration files provided with this repository. See README.md files for [sample scripts](#Model Compression Samples) for links to exact configuration files and final PyTorch checkpoints.


|Model|Compression algorithm|Dataset|PyTorch FP32 baseline|PyTorch compressed accuracy|
| :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|-|76.13|
|ResNet-50|INT8|ImageNet|76.13|76.05|
|ResNet-50|Mixed, 44.8% INT8 / 55.2% INT4|ImageNet|76.13|76.3|
|ResNet-50|INT8 + Sparsity 61% (RB)|ImageNet|76.13|75.28|
|ResNet-50|Filter pruning, 30%, magnitude criterion|ImageNet|76.13|75.7|
|ResNet-50|Filter pruning, 30%, geometric median criterion|ImageNet|76.13|75.7|
|Inception V3|None|ImageNet|-|77.32|
|Inception V3|INT8|ImageNet|77.32|76.92|
|Inception V3|INT8 + Sparsity 61% (RB)|ImageNet|77.32|76.98|
|MobileNet V2|None|ImageNet|-|71.81|
|MobileNet V2|INT8|ImageNet|71.81|71.34|
|MobileNet V2|Mixed, 46.6% INT8 / 53.4% INT4|ImageNet|71.81|70.89|
|MobileNet V2|INT8 + Sparsity 52% (RB)|ImageNet|71.81|70.99|
|SqueezeNet V1.1|None|ImageNet|-|58.18|
|SqueezeNet V1.1|INT8|ImageNet|58.18|58.02|
|SqueezeNet V1.1|Mixed, 54.7% INT8 / 45.3% INT4|ImageNet|58.18|58.84|
|ResNet-18|None|ImageNet|-|69.76|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|69.76|61.61|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|69.76|61.59|
|ResNet-18|Filter pruning, 30%, magnitude criterion|ImageNet|69.76|68.69|
|ResNet-18|Filter pruning, 30%, geometric median criterion|ImageNet|69.76|68.97|
|ResNet-34|None|ImageNet|-|73.31|
|ResNet-34|Filter pruning, 30%, magnitude criterion|ImageNet|73.31|72.54|
|ResNet-34|Filter pruning, 30%, geometric median criterion|ImageNet|73.31|72.60|
|SSD300-BN|None|VOC12+07|-|78.28|
|SSD300-BN|INT8|VOC12+07|78.28|78.07|
|SSD300-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07|78.28|78.01|
|SSD512-BN|None|VOC12+07|-|80.26|
|SSD512-BN|INT8|VOC12+07|80.26|80.02|
|SSD512-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07|80.26|79.98|
|UNet|None|CamVid|-|71.95|
|UNet|INT8|CamVid|71.95|71.66|
|UNet|INT8 + Sparsity 60% (Magnitude)|CamVid|71.95|71.72|
|ICNet|None|CamVid|-|67.89|
|ICNet|INT8|CamVid|67.89|67.87|
|ICNet|INT8 + Sparsity 60% (Magnitude)|CamVid|67.89|67.24|
|UNet|None|Mapillary|-|56.23|
|UNet|INT8|Mapillary|56.23|56.12|
|UNet|INT8 + Sparsity 60% (Magnitude)|Mapillary|56.23|56.0|

