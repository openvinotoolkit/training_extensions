Algorithm
===================

Introduction
------------
This section contains algorithmic implementations. OpenVINO™ Training Extensions provides number of
different algorithms such as classification, detection,
segmentation and anomaly with various learning types such as supervised,
semi and self-supervised learning.

.. toctree::
   :maxdepth: 1

   action/index
   anomaly/index
   classification/index
   detection/index
   segmentation/index


Organizational Structure
------------------------
Algorithms have the following organizational structure:

.. code-block:: bash

      <algorithm>
      ├── adapters
      │   └── <library>
      │       ├── config
      │       ├── data
      │       └── ...
      ├── configs
      │   └── <model_name>
      │       ├── template.yaml
      │       ├── configuration.py
      │       ├── configuration.yaml
      │       ├── compression_config.json
      │       └── hpo_config.yaml
      ├── tasks
      │   ├── train.py
      │   ├── inference.py
      │   ├── nncf.py
      │   └── openvino.py
      └── tools
         ├── README.md
         └── sample.py

where each algorithm has ``adapters``, ``configs``, ``tasks`` and ``tools``.

Adapters
^^^^^^^^
``adapters`` contain modules to wrap the original library used to perform the
task. For instance, detection task uses
`mmdetection <https://github.com/open-mmlab/mmdetection>`_ library, meaning that
``adapters`` comprises adapters to wrap ``mmdetection`` to use with OpenVINO™ Training Extensions.

Configs
^^^^^^^
``configs`` contain configuration related files including training, inference,
`NNCF <https://github.com/openvinotoolkit/nncf>`_ and
`HPO <https://github.com/openvinotoolkit/hyper_parameter_optimization>`_.

Tasks
^^^^^
.. _tasks:

Tasks contain implementations that correspond to each phase in the workflow from
training to OpenVINO inference. Each algorithm expects ``train``, ``inference``,
``nncf`` and ``openvino`` python modules that implement the
`task interfaces <https://github.com/openvinotoolkit/training_extensions/tree/develop/otx/api/usecases/tasks/interfaces>`_.

Tools
^^^^^
Tools contain python implementations that performs :ref:`tasks <tasks>` in
end-to-end workflow. For example, current anomaly implementation has ``sample.py``
file that reads an input dataset, trains a model and exports the model to
OpenVINO IR via either `POT <https://docs.openvino.ai/2020.4/pot_README.html>`_
or `NNCF <https://docs.openvino.ai/latest/docs_nncf_introduction.html>`_.