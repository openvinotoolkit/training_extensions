# OpenVINO™ Training Extensions

OpenVINO™ Training Extensions provide a convenient environment to train
Deep Learning models and convert them using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Quick Start Guide

### Prerequisites
* Ubuntu 18.04 / 20.04
* Python 3.6+
* [OpenVINO™](https://software.intel.com/en-us/openvino-toolkit) - for exporting and running models
* [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive) - for training on GPU

### Setup OpenVINO™ Training Extensions

1. Clone repository in the working directory by running the following:
    ```
    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
    git checkout -b brave_new_world origin/brave_new_world
    git submodule update --init --recursive
    ```

2. Install prerequisites by running the following:
    ```
    sudo apt-get install python3-pip virtualenv
    ```

3. Create and activate virtual environment:
    ```
    virtualenv venv
    source venv/bin/activate
    ```

4. Install `ote_cli` package:
    ```
    pip3 install -e ote_cli/
    ```
    
5. Instantiate templates and create virtual environments:
   ```
   python3 tools/instantiate.py --destination model_templates --verbose --init-venv
   ```
6. Activate algo-backend related virtual environment:
   ```
   source model_templates/OTEDetection_v2.9.1/venv/bin/activate
   ```
7. Use Jupiter notebooks or OTE CLI tools to start working with models:
   * To run notebook:
     ```
     cd ote_cli/notebooks/; jupyter notebook
     ```
   * OTE CLI TBD

## Misc

Models that were previously developed can be found [here](misc).

## Contributing

Please read the [contribution guidelines](CONTRIBUTING.md) before starting work on a pull request.

## Known Limitations

Currently, training, exporting, evaluation scripts for TensorFlow\*-based models and the most of PyTorch\*-based models from [Misc](#misc) section are exploratory and are not validated.

---
\* Other names and brands may be claimed as the property of others.
