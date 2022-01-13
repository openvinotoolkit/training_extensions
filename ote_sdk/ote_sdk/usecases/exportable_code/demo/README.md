# Exportable code - demo package

Demo package contains simple demo to get and visualize result of model inference.

## Structure of generated package:

* model
  - `model.xml`
  - `model.bin`
  - `config.json`
* python
  - `README.md`
  - `demo.py`
  - `model.py` (Optional)
  - `requirements.txt`

> **NOTE**: zip archive will contain `model.py` when [ModelAPI](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/openvino/model_zoo/model_api) has no appropriate standard model wrapper for the model

## Prerequisites
* Python 3.8+

## Setup Demo Package

1. Install Python (version 3.8 or higher).

2. Install needed requirements in the clean environment (please make sure that the environment contains [setuptools](https://pypi.org/project/setuptools/), [wheel](https://pypi.org/project/wheel/)):
```
python -m pip install -r requirements.txt
```

> **NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`. You may also need to [install pip](https://pip.pypa.io/en/stable/installation/).
> For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-pip`.

## Usecases

1. Running the `demo.py` application with the `-h` option yields the following usage message:
   ```
   usage: demo.py [-h] -i INPUT -m MODEL -c CONFIG

   Options:
     -h, --help            Show this help message and exit.
     -i INPUT, --input INPUT
                           Required. An input to process. The input must be a
                           single image, a folder of images, video file or
                           camera id.
     -m MODEL, --model MODEL
                           Required. Path to an .xml file with a trained model.
     -c CONFIG, --config CONFIG
                           Required. Path to an .json file with parameters for
                           model.


   ```

   As a model, you can use `model.xml` from generated zip. So can use the following command to do inference with a pre-trained model:
   ```
   python3 demo.py \
     -i <path_to_video>/inputVideo.mp4 \
     -m <path_to_model>/model.xml \
     -c <path_to_model>/config.json
   ```
   You can press `Q` to stop inference during demo running.
   > **NOTE**: Default configuration contains info about pre- and postprocessing to model inference and is guaranteed to be correct.
   > Also you can define own json config that specifies needed parameters, but any change should be made with caution.
   > To create this config please see `config.json` in model files from generated zip.

2. You can create your own demo application, using `demo_package`. The main function of package is `create_model`:
   ```python
   def create_model(model_file: Path, config_file: Path, path_to_wrapper: Optional[Path] = None) -> Model:
    """
    Create model using ModelAPI factory

    :param model_path: Path to .xml model
    :param config_file: Path to .json config.
    :param path_to_wrapper: Path to model wrapper
    """
   ```
   Function returns model wrapper from ModelAPI. To get more information please see [ModelAPI](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/openvino/model_zoo/model_api). If you want to use your own model wrapper you should provide path to wrapper as argument of `create_model` function.

   Some example how to use `demo_package`:
   ```python
   import cv2
   from ote_sdk.usecases.exportable_code.demo.demo_package import create_model

   # read input
   frame = cv2.imread(path_to_image)
   # create model
   model = create_model(path_to_model, path_to_config)
   # inference
   objects = model(frame)
   # show results using some visualizer
   output = visualizer.draw(frame, objects)
   cv2.imshow(output)
   ```