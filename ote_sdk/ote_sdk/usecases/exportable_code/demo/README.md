# Exportable code - demo package

Demo package contains simple demo to get and visualize result of model inference.
Name of package is a name of model which was deployed.

## Structure of generated package:

openvino.zip
  * [model](./model)
    * [model.xml](./model/model.xml)
    * [model.bin](./model/model.bin)
  * [python](./python)
    * [README.md](./python/README.md)
    * [demo.py](./python/demo.py)
    * [requirements.txt](./python/requirements.txt)
    * <model_name>-0.0-py3-none-any.whl


## Prerequisites
* Python 3.6+

## Setup Demo Package

1. Install Python (version 3.6 or higher).

2. Install the package in the clean environment:
```
python -m pip install <model_name>-0.0-py3-none-any.whl
```


When the package is installed, you can import it as follows:
```
python -c "from <model_name> import create_model"
```

> **NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`. You may also need to [install pip](https://pip.pypa.io/en/stable/installation/).
> For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-pip`.

## Usecases

1. You can run demo, using entry-point <model_name>. For example if you model name `atss`, running the application with the -h option yields the following usage message:
   ```
   usage: atss [-h] -i INPUT -m MODEL

   Options:
     -h, --help            Show this help message and exit.
     -i INPUT, --input INPUT
                           Required. An input to process. The input must be a
                           single image.
     -m MODEL, --model MODEL
                           Required. Path to an .xml file with a trained model.
   ```

   As a model, you can use `model.xml` from generated zip.

2. You can create your own demo application, using <model_name> package.