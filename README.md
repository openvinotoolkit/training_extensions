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
    sudo apt-get install python3-pip python3-venv virtualenv
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
   * OTE CLI tools
      * ote_train
          ```
          ote_train --help
          usage: ote_train [-h] --train-ann-files TRAIN_ANN_FILES --train-data-roots TRAIN_DATA_ROOTS --val-ann-files VAL_ANN_FILES --val-data-roots VAL_DATA_ROOTS [--load-weights LOAD_WEIGHTS] --save-weights
                            SAVE_WEIGHTS
                            {params} ...

          positional arguments:
            {params}              sub-command help
              params              Hyper parameters defined in template file.

          optional arguments:
            -h, --help            show this help message and exit
            --train-ann-files TRAIN_ANN_FILES
                                  Comma-separated paths to training annotation files.
            --train-data-roots TRAIN_DATA_ROOTS
                                  Comma-separated paths to training data folders.
            --val-ann-files VAL_ANN_FILES
                                  Comma-separated paths to validation annotation files.
            --val-data-roots VAL_DATA_ROOTS
                                  Comma-separated paths to validation data folders.
            --load-weights LOAD_WEIGHTS
                                  Load only weights from previously saved checkpoint
            --save-weights SAVE_WEIGHTS
                                  Location to store wiehgts.
          ```
          ```
          ote_train params --help
          usage: ote_train params [-h] [--learning_parameters.batch_size BATCH_SIZE] [--learning_parameters.learning_rate LEARNING_RATE]
                            [--learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS] [--learning_parameters.num_iters NUM_ITERS]
                            [--postprocessing.confidence_threshold CONFIDENCE_THRESHOLD] [--postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD]
          optional arguments:
            -h, --help            show this help message and exit
            --learning_parameters.batch_size BATCH_SIZE
                                  header: Batch size
                                  type: INTEGER
                                  default_value: 64
                                  max_value: 512
                                  min_value: 1
            --learning_parameters.learning_rate LEARNING_RATE
                                  header: Learning rate
                                  type: FLOAT
                                  default_value: 0.05
                                  max_value: 0.1
                                  min_value: 1e-07
            --learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS
                                  header: Number of iterations for learning rate warmup
                                  type: INTEGER
                                  default_value: 100
                                  max_value: 10000
                                  min_value: 0
            --learning_parameters.num_iters NUM_ITERS
                                  header: Number of training iterations
                                  type: INTEGER
                                  default_value: 10000
                                  max_value: 100000
                                  min_value: 10
            --postprocessing.confidence_threshold CONFIDENCE_THRESHOLD
                                  header: Confidence threshold
                                  type: FLOAT
                                  default_value: 0.35
                                  max_value: 1
                                  min_value: 0
            --postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD
                                  header: Result based confidence threshold
                                  type: BOOLEAN
                                  default_value: True
          ```
      * ote_eval
          ```
          ote_eval --help
          usage: ote_eval [-h] --test-ann-files TEST_ANN_FILES --test-data-roots TEST_DATA_ROOTS --load-weights LOAD_WEIGHTS {params} ...

          positional arguments:
            {params}              sub-command help
              params              Hyper parameters defined in template file.

          optional arguments:
            -h, --help            show this help message and exit
            --test-ann-files TEST_ANN_FILES
                                  Comma-separated paths to test annotation files.
            --test-data-roots TEST_DATA_ROOTS
                                  Comma-separated paths to test data folders.
            --load-weights LOAD_WEIGHTS
                                  Load only weights from previously saved checkpoint
          ```
          ```
          ote_eval params --help
          usage: ote_eval params [-h] [--postprocessing.confidence_threshold CONFIDENCE_THRESHOLD] [--postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD]

          optional arguments:
            -h, --help            show this help message and exit
            --postprocessing.confidence_threshold CONFIDENCE_THRESHOLD
                                  header: Confidence threshold
                                  type: FLOAT
                                  default_value: 0.35
                                  max_value: 1
                                  min_value: 0
            --postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD
                                  header: Result based confidence threshold
                                  type: BOOLEAN
                                  default_value: True
          ```
      * ote_export
          ```
          ote_export --help
          usage: ote_export [-h] --load-weights LOAD_WEIGHTS --save-model-to SAVE_MODEL_TO [--ann-files ANN_FILES] [--labels LABELS [LABELS ...]]

          optional arguments:
            -h, --help            show this help message and exit
            --load-weights LOAD_WEIGHTS
                                  Load only weights from previously saved checkpoint
            --save-model-to SAVE_MODEL_TO
                                  Location where exported model will be stored.
            --ann-files ANN_FILES
            --labels LABELS [LABELS ...]
          ```
## Misc

Models that were previously developed can be found [here](misc).

## Contributing

Please read the [contribution guidelines](CONTRIBUTING.md) before starting work on a pull request.

## Known Limitations

Currently, training, exporting, evaluation scripts for TensorFlow\*-based models and the most of PyTorch\*-based models from [Misc](#misc) section are exploratory and are not validated.

---
\* Other names and brands may be claimed as the property of others.
