# Exportable code - demo package

Demo package contains simple demo to get and visualize result of model inference.

## Structure of generated package:

* model
  - `model.xml`
  - `model.bin`
* python
  - `README.md`
  - `LICENSE`
  - `demo.py`
  - `requirements.txt`
  - `demo_package-0.0-py3-none-any.whl`


## Prerequisites
* [Python 3.8](https://www.python.org/downloads/)
* [Git](https://git-scm.com/)

## Setup Demo Package

1. Install [prerequisites](#prerequisites). You may also need to [install pip](https://pip.pypa.io/en/stable/installation/). For example, on Ubuntu execute the following command to get pip installed:
   ```
   sudo apt install python3-pip
   ```

2. Create clean virtual environment:

   One of the possible ways for creating a virtual environment is to use `virtualenv`:
   ```
   python -m pip install virtualenv
   python -m virtualenv <directory_for_environment>
   ```

   Before starting to work inside virtual environment, it should be activated:

   On Linux and macOS:
   ```
   source <directory_for_environment>/bin/activate
   ```

   On Windows:
   ```
   .\<directory_for_environment>\Scripts\activate
   ```

   Please make sure that the environment contains [wheel](https://pypi.org/project/wheel/) by calling the following command:

   ```
   python -m pip install wheel
   ```
   > **NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`.

3. Install the package in the environment:
   ```
   python -m pip install demo_package-0.0-py3-none-any.whl
   ```


When the package is installed, you can import it as follows:
```
python -c "from demo_package import create_model"
```

## Usecases

1. Running the `demo.py` application with the `-h` option yields the following usage message:
   ```
   usage: demo.py [-h] -i INPUT -m MODEL [-c CONFIG]

   Options:
     -h, --help            Show this help message and exit.
     -i INPUT, --input INPUT
                           Required. An input to process. The input must be a
                           single image, a folder of images, video file or camera
                           id.
     -m MODEL, --model MODEL
                           Required. Path to an .xml file with a trained model.
     -c CONFIG, --config CONFIG
                           Optional. Path to an .json file with parameters for
                           model.

   ```

   As a model, you can use `model.xml` from generated zip. So can use the following command to do inference with a pre-trained model:
   ```
   python3 demo.py \
     -i <path_to_video>/inputVideo.mp4 \
     -m <path_to_model>/model.xml
   ```
   You can press `Q` to stop inference during demo running.
   > **NOTE**: Default configuration contains info about pre- and postprocessing to model inference and is guaranteed to be correct.
   > Also you can define own json config that specifies needed parameters, but any change should be made with caution.
   > To create this config please see `config.json` in demo_package wheel.

2. You can create your own demo application, using `demo_package`. The main function of package is `create_model`:
   ```python
   def create_model(model_path: Path, config_file: Path = None) -> Model:
    """
    Create model using ModelAPI factory

    :param model_path: Path to .xml model
    :param config_file: Path to .json config. If it is not defined, use config from demo_package
    """
   ```
   Function returns model wrapper from ModelAPI. To get more information please see [ModelAPI](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/openvino/model_zoo/model_api).

   Some example how to use `demo_package`:
   ```python
   import cv2
   from demo_package import create_model

   # read input
   frame = cv2.imread(path_to_image)
   # create model
   model = create_model(path_to_model)
   # inference
   objects = model(frame)
   # show results using some visualizer
   output = visualizer.draw(frame, objects)
   cv2.imshow(output)
   ```

## Troubleshooting

1. If you have access to the Internet through the proxy server only, please use pip with proxy call as demonstrated by command below:
   ```
   python -m pip install --proxy http://<usr_name>:<password>@<proxyserver_name>:<port#> <pkg_name>
   ```

2. If you use Anaconda environment, you should consider that OpenVINO has limited [Conda support](https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_conda.html) for Python 3.6 and 3.7 versions only. But the demo package requires python 3.8. So please use other tools to create the environment (like `venv` or `virtualenv`) and use `pip` as a package manager.