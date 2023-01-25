How to deploy the model
=======================

This guide shows, how to deploy a model trained in previous stage. 
It provides the exported model together with self-contained python package, a demo application to port and infer it outside of this repository.

To be specific, this tutorial uses as an example the object detection ATSS model trained and expoted in previuos step and located in ``outputs/openvino``. 

1. Activate virtual environment created in previous step.

.. code-block::

    source <directory_for_environment>/bin/activate

2. ``otx deploy`` returns an ``openvino.zip`` archive with the following files:

- model

  - ``model.xml`` and ``model.bin`` - model exported to the OpenVINO™ format
  - ``config.json`` - file, containing post-processing info and meta information about labels in dataset

- python

  - model_wrappers (Optional)
  - ``README.md``
  - ``LICENSE``
  - ``demo.py``- simple demo to visualize results of model inference
  - ``requirements.txt`` - minimal packages required to run demo


3. We can deploy exported to IR model, using the command below:

.. code-block::

    (detection) ...$ otx deploy otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml 
               --load-weights outputs/openvino/openvino.xml
               --save-model-to outputs/deploy

    2023-01-20 09:30:40,938 | INFO : Loading OpenVINO OTXDetectionTask
    2023-01-20 09:30:41,736 | INFO : OpenVINO task initialization completed
    2023-01-20 09:30:41,737 | INFO : Deploying the model
    2023-01-20 09:30:41,753 | INFO : Deploying completed


After that, we can use the resulting ``openvino.zip`` archive to visualize the inference, please refer :doc:`demo`.