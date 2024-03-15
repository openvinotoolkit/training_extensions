Simple HPO Tutorial
============================

This tutorial provides a step-by-step guide on how to use Hyper-Parameter Optimization (HPO) for classification tasks.
In this tutorial, we will optimize the learning rate and batch size using HPO.

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../get_started/installation>`
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual
environment:

.. code-block::

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate


*************************
Build workspace
*************************

First, let's build a workspace. You can do this by running the following command:

.. code-block::

    (otx) ...$ otx build --train-data-roots data/flower_photos --model MobileNet-V3-large-1x

    [*] Load Model Template ID: Custom_Image_Classification_MobileNet-V3-large-1x
    [*] Load Model Name: MobileNet-V3-large-1x
    [*] Saving data configuration file to: ./otx-workspace-CLASSIFICATION/data.yaml

    (otx) ...$ cd ./otx-workspace-CLASSIFICATION

.. note::

    This is copied from :doc:`../../tutorials/base/how_to_train/classification`.
    You can find more detail explanation from it.

*************************
Set hpo_config.yaml
*************************

Before running HPO, you can configure HPO using the ``hpo_config.yaml`` file.
This file contains all the information that the HPO module needs, including the hyperparameters that you want to optimize.
The file is located in the workspace you have made and comes with default values.

Here's the default hpo_config.yaml:

.. code-block::

    metric: accuracy
    search_algorithm: asha
    hp_space:
      learning_parameters.learning_rate:
        param_type: qloguniform
        range:
          - 0.0007
          - 0.07
          - 0.0001
      learning_parameters.batch_size:
        param_type: qloguniform
        range:
          - 32
          - 128
          - 2

Although this default configuration can be used for HPO, the search space for the learning rate is too wide.
Therefore, we will modify the configuration file slightly to make the search space more reasonable. You can easily modify the configuration file to optimize different hyperparameters.

Here's the updated ``hpo_config.yaml``:

.. code-block::

  ...
    ...
    ...
      learning_parameters.learning_rate:
        param_type: quniform
        range:
          - 0.001
          - 0.01
          - 0.001
    ...
    ...
    ...

By modifying the ``hpo_config.yaml`` file, you can easily change the search space or hyperparameters that will be optimized during the HPO process.

*************************************
Run OpenVINO™ Training Extensions
*************************************

Now it's time to run OpenVINO™ Training Extensions. You can enable HPO by adding the argument **--enable-hpo**. By default, HPO will use four times the time allocated to training. However, if you are short on time, you can reduce the time for HPO as training by adding the argument   **--hpo-time-ratio** and setting it to 2. This means that HPO will use twice the time allocated to training.

Here's an tutorial command:

.. code-block::

    $ otx train \
        ... \
        --enable-hpo \
        --hpo-time-ratio 2

With this command, HPO is automatically set to use twice the time allocated for training. You can easily adjust the HPO time allocation by modifying the value of the **--hpo-time-ratio** argument.
