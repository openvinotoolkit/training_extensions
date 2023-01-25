How to deploy the model
=======================

This guide shows, how to deploy a model trained in the previous stage. 
As a result of this step, we'll get the exported model together with the self-contained python package, a demo application to port and infer it outside of this repository.

To be specific, this tutorial uses as an example the object detection ATSS model trained and exported in the previuos step and located in ``outputs/openvino``. 

1. Activate the virtual environment created in the previous step.

.. code-block::

    source <directory_for_environment>/bin/activate

2. ``otx deploy`` returns an ``openvino.zip`` archive with the following files:

- model

  - ``model.xml`` and ``model.bin`` - model exported to the OpenVINO™ format
  - ``config.json`` - file containing the post-processing info and meta information about labels in the dataset

- python

  - model_wrappers (Optional)
  - ``README.md``
  - ``LICENSE``
  - ``demo.py``- simple demo to visualize results of model inference
  - ``requirements.txt`` - minimal packages required to run the demo


3. We can deploy the model exported to IR, using the command below:

.. code-block::

    (detection) ...$ otx deploy otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml 
               --load-weights outputs/openvino/openvino.xml
               --save-model-to outputs/deploy

    2023-01-20 09:30:40,938 | INFO : Loading OpenVINO OTXDetectionTask
    2023-01-20 09:30:41,736 | INFO : OpenVINO task initialization completed
    2023-01-20 09:30:41,737 | INFO : Deploying the model
    2023-01-20 09:30:41,753 | INFO : Deploying completed

We also can deploy the quantized model, that was optimized with NNCF in the previous step, passing the path to this model to ``--load-weights`` parameter.

After that, we can use the resulting ``openvino.zip`` archive to visualize the inference :doc:`demo`.