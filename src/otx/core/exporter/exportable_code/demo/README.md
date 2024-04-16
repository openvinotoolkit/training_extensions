# Exportable code

Exportable code is a .zip archive that contains simple demo to get and visualize result of model inference.

## Structure of generated zip

- `README.md`
- `LICENSE`
- model
  - `model.xml`
  - `model.bin`
  - `config.json`
- python
  - demo_package
    - `__init__.py`
    - executors
      - `__init__.py`
      - `asynchronous.py`
      - `synchronous.py`
    - inference
      - `__init__.py`
      - `inference.py`
    - streamer
      - `__init__.py`
      - `streamer.py`
    - visualizers
      - `__init__.py`
      - `visualizer.py`
      - `vis_utils.py`
  - `demo.py`
  - `requirements.txt`
  - `setup.py`

## Prerequisites

- [Python 3.10](https://www.python.org/downloads/)
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
   cd python
   python setup.py install
   ```

## Usecase

1. Running the `demo.py` application with the `-h` option yields the following usage message:

   ```bash
   usage: demo.py [-h] -i INPUT -m MODEL [MODEL ...] [-it {sync,async}] [-l] [--no_show] [-d {CPU,GPU}] [--output OUTPUT]

   Options:
   -h, --help            Show this help message and exit.
   -i INPUT, --input INPUT
                           Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
   -m MODEL [MODEL ...], --model MODELS [MODELS ...]
                           Optional. Path to directory with trained model and configuration file. Default value points to deployed model folder '../model'.
   -it {sync,async}, --inference_type {sync,async}
                           Optional. Type of inference for single model.
   -l, --loop            Optional. Enable reading the input in a loop.
   --no_show             Optional. Disables showing inference results on UI.
   -d {CPU,GPU}, --device {CPU,GPU}
                           Optional. Device to infer the model.
   --output OUTPUT       Optional. Output path to save input data with predictions.
   ```

2. As a `model` parameter the default value `../model` will be used. Or you can specify the other path to the model directory from generated zip. You can pass as `input` a single image, a folder of images, a video file, or a web camera id. So you can use the following command to do inference with a pre-trained model:

   ```bash
   python3 demo.py -i <path_to_video>/inputVideo.mp4
   ```

   You can press `Q` to stop inference during demo running.

   > **NOTE**: If you provide a single image as input, the demo processes and renders it quickly, then exits. To continuously
   > visualize inference results on the screen, apply the `--loop` option, which enforces processing a single image in a loop.
   > In this case, you can stop the demo by pressing `Q` button or killing the process in the terminal (`Ctrl+C` for Linux).
   >
   > **NOTE**: Default configuration contains info about pre- and post processing for inference and is guaranteed to be correct.
   > Also you can change `config.json` that specifies the confidence threshold and color for each class visualization, but any
   > changes should be made with caution.

3. To save inferenced results with predictions on it, you can specify the folder path, using `--output`.
   It works for images, videos, image folders and web cameras. To prevent issues, do not specify it together with a `--loop` parameter.

   ```bash
   python3 demo.py \
      --input <path_to_image>/inputImage.jpg \
      --models ../model \
      --output resulted_images
   ```

4. To run a demo on a web camera, you need to know its ID.
   You can check a list of camera devices by running this command line on Linux system:

   ```bash
   sudo apt-get install v4l-utils
   v4l2-ctl --list-devices
   ```

   The output will look like this:

   ```bash
   Integrated Camera (usb-0000:00:1a.0-1.6):
      /dev/video0
   ```

   After that, you can use this `/dev/video0` as a camera ID for `--input`.

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
