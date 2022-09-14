# Exportable code

Exportable code is a .zip archive that contains simple demo to get and visualize result of model inference.

## Structure of generated zip

- model
  - `model.xml`
  - `model.bin`
  - `config.json`
- python
  - model_wrappers (Optional)
    - `__init__.py`
    - model_wrappers required to run demo
  - `README.md`
  - `LICENSE`
  - `demo.py`
  - `requirements.txt`

> **NOTE**: Zip archive contains model_wrappers when [ModelAPI](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/openvino/model_zoo/model_api) has no appropriate standard model wrapper for the model.

## Prerequisites

- [Python 3.8](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

## Install requirements to run demo

1. Install [prerequisites](#prerequisites). You may also need to [install pip](https://pip.pypa.io/en/stable/installation/). For example, on Ubuntu execute the following command to get pip installed:

   ```bash
   sudo apt install python3-pip
   ```

1. Create clean virtual environment:

   One of the possible ways for creating a virtual environment is to use `virtualenv`:

   ```bash
   python -m pip install virtualenv
   python -m virtualenv <directory_for_environment>
   ```

   Before starting to work inside virtual environment, it should be activated:

   On Linux and macOS:

   ```bash
   source <directory_for_environment>/bin/activate
   ```

   On Windows:

   ```bash
   .\<directory_for_environment>\Scripts\activate
   ```

   Please make sure that the environment contains [wheel](https://pypi.org/project/wheel/) by calling the following command:

   ```bash
   python -m pip install wheel
   ```

   > **NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`.

1. Install requirements in the environment:

   ```bash
   python -m pip install -r requirements.txt
   ```

1. Add `model_wrappers` package to PYTHONPATH:

   On Linux and macOS:

   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/model_wrappers
   ```

   On Windows:

   ```bash
   set PYTHONPATH=%PYTHONPATH%;/path/to/model_wrappers
   ```

## Usecase

Running the `demo.py` application with the `-h` option yields the following usage message:

```bash
usage: demo.py [-h] -i INPUT -m MODELS [MODELS ...] [-it {sync,async}] [-l]
Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera
                        id.
  -m MODELS [MODELS ...], --models MODELS [MODELS ...]
                        Required. Path to directory with trained model and
                        configuration file. If you provide several models you
                        will start the task chain pipeline with the provided
                        models in the order in which they were specified
  -it {sync,async}, --inference_type {sync,async}
                        Optional. Type of inference for single model
  -l, --loop            Optional. Enable reading the input in a loop.
  --no_show
                        Optional. If this flag is specified, the demo
                        won't show the inference results on UI.
```

As a model, you can use path to model directory from generated zip. So you can use the following command to do inference with a pre-trained model:

```bash
python3 demo.py \
  -i <path_to_video>/inputVideo.mp4 \
  -m <path_to_model_directory>
```

You can press `Q` to stop inference during demo running.

> **NOTE**: If you provide a single image as an input, the demo processes and renders it quickly, then exits. To continuously
> visualize inference results on the screen, apply the `loop` option, which enforces processing a single image in a loop.
>
> **NOTE**: Default configuration contains info about pre- and post processing for inference and is guaranteed to be correct.
> Also you can change `config.json` that specifies needed parameters, but any changes should be made with caution.

## Troubleshooting

1. If you have access to the Internet through the proxy server only, please use pip with proxy call as demonstrated by command below:

   ```bash
   python -m pip install --proxy http://<usr_name>:<password>@<proxyserver_name>:<port#> <pkg_name>
   ```

1. If you use Anaconda environment, you should consider that OpenVINO has limited [Conda support](https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_conda.html) for Python 3.6 and 3.7 versions only. But the demo package requires python 3.8. So please use other tools to create the environment (like `venv` or `virtualenv`) and use `pip` as a package manager.

1. If you have problems when you try to use `pip install` command, please update pip version by following command:

   ```bash
   python -m pip install --upgrade pip
   ```
