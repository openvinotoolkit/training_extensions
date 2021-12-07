# Exportable code - demo package

Demo package contains simple demo to get and visualize result of model inference.
Name of package is a name of model which was deployed.

## Structure of generated package:

* [model](../model)
  * [model.xml](../model/model.xml)
  * [model.bin](../model/model.bin)
* [python](.)
  * [README.md](./README.md)
  * [demo.py](./demo.py)
  * [requirements.txt](./requirements.txt)
  * [demo_package-0.0-py3-none-any.whl](./demo_package-0.0-py3-none-any.whl)


## Prerequisites
* Python 3.8+

## Setup Demo Package

1. Install Python (version 3.8 or higher).

2. Install the package in the clean environment:
```
python -m pip install demo_package-0.0-py3-none-any.whl
```


When the package is installed, you can import it as follows:
```
python -c "from demo_package import create_model"
```

> **NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`. You may also need to [install pip](https://pip.pypa.io/en/stable/installation/).
> For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-pip`.

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

   Also you can define own json config that specify some model parameters. To create this config please see `config.json` in demo_package wheel.

2. You can create your own demo application, using `demo_package`. The main function of package is `create_model`:
   ```python
   def create_model(model_path, config_file=None):
    """
    Create model using ModelAPI factory

    :param model_path: Path to .xml model
    :param config_file: Path to .json config. If not define, use config from demo_package
    """
   ```
   Function returns model wrapper from ModelAPI. To get more information please see [ModelAPI](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/openvino/model_zoo/model_api).

   Some example how to use `demo_package`:
   ```python
   import cv2
   from demo_package import create_model
   path_to_model = ""
   path_to_image = ""
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