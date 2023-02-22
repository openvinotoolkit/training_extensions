Hyperparameters Optimization
============================

Hyper-parameter optimization (HPO) can be a time-consuming process, even with state-of-the-art off-the-shelf libraries. OpenVINO™ Training Extensions makes HPO faster and easier by providing an easy-to-use interface and automatic configuration.

With OpenVINO™ Training Extensions, you can run hyper-parameter optimization by simply adding a time constraint parameter. The auto-config feature automatically sets internal control parameters, guaranteeing that HPO will finish within the given time constraint.

OpenVINO™ Training Extensions provides both sequential and parallel methods, making it scalable for different training environments. If you have multiple GPUs, you can accelerate HPO by utilizing all available GPU resources.

Key features of OpenVINO™ Training Extensions include:

- **Easy usability** : By using time as the control parameter, OpenVINO™ Training Extensions offers a straightforward and intuitive interface for users.

- **Auto-config** : The automatic configuration feature sets internal control parameters automatically, ensuring that HPO finishes within the given time constraint.

- **Scalability** : OpenVINO™ Training Extensions offers both sequential and parallel methods, making it scalable for different training environments. If you have multiple GPUs, you can take advantage of all available GPU resources to accelerate HPO.

**************
Simpe HPO Exapmle
**************

This exapmle provides a step-by-step guide on how to use Hyper-Parameter Optimization (HPO) for classification tasks. In this example, we will optimize the learning rate and batch size using HPO.

=========================
1. Build workspace
=========================

First, let's build a workspace. You can do this by running the following command:

.. code-block::

    (otx) ...$ otx build --train-data-roots data/flower_photos --model MobileNet-V3-large-1x

    [*] Load Model Template ID: Custom_Image_Classification_MobileNet-V3-large-1x
    [*] Load Model Name: MobileNet-V3-large-1x
    [*] Saving data configuration file to: ./otx-workspace-CLASSIFICATION-MobileNet-V3-large-1x/data.yaml

    (otx) ...$ cd ./otx-workspace-CLASSIFICATION-MobileNet-V3-large-1x

.. note::

    This is copied from :doc:`../../tutorials/base/how_to_train/classification`.
    You can find more detail explanation from it.

=========================
2. Set hpo_config.yaml
=========================

Before running HPO, you can configure HPO using the hpo_config.yaml file. This file contains all the information that the HPO module needs, including the hyperparameters that you want to optimize. The file is located in the workspace you have made and comes with default values.

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

Although this default configuration can be used for HPO, the search space for the learning rate is too wide. Therefore, we will modify the configuration file slightly to make the search space more reasonable. You can easily modify the configuration file to optimize different hyperparameters.

Here's the updated hpo_config.yaml:

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

By modifying the hpo_config.yaml file, you can easily change the search space or hyperparameters that will be optimized during the HPO process.

=========================
3. Run OpenVINO™ Training Extensions
=========================

Now it's time to run OpenVINO™ Training Extensions. You can enable HPO by adding the argument **--enable-hpo**. By default, HPO will use four times the time allocated to training. However, if you are short on time, you can reduce the time for HPO as training by adding the argument   **--hpo-time-ratio** and setting it to 2. This means that HPO will use twice the time allocated to training.

Here's an example command:

.. code-block::

    $ otx train \
        ... \
        --enable-hpo \
        --hpo-time-ratio 2

With this command, HPO is automatically set to use twice the time allocated for training. You can easily adjust the HPO time allocation by modifying the value of the **--hpo-time-ratio** argument.

**************
How to configure hyper-parameter optimization
**************

You can configure HPO by modifying the **hpo_config.yaml** file. This file contains everything related to HPO, including the hyperparameters to optimize, the HPO algorithm, and more. The **hpo_config.yaml** file already exists with default values in the same directory where *template.yaml* resides. Here is the default hpo_config.yaml file for classification:

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

As you can see, there are a few attributes required to run HPO.
Fortunately, there are not many attributes, so it's not difficult to write your own **hpo_config.yaml** file. The more detailed description is as follows:

- **hp_space** (*List[Dict[str, Any]]*, `required`) - Hyper parameter search space to find. It should be list of dictionary. Each dictionary has a hyperparameter name as the key and param_type and range as the values. You can optimize any learning parameters of each task.

  - **Keys of each hyper parameter**

    - **param_type** (*str*, `required`) : Hyper parameter search space type. It must be one of the following:

      - uniform : Samples a float value uniformly between the lower and upper bounds.
      - quniform : Samples a quantized float value uniformly between the lower and upper bounds.
      - loguniform : Samples a float value after scaling search space by logarithm scale.
      - qloguniform : Samples a quantized float value after scaling the search space by logarithm scale.
      - choice : Samples a categorical value.

    - **range** (*List[Any]*, `required`)

      - uniform : List[Union[float, int]]

        - min (*Union[float, int]*, `required`) : The lower bound of search space.
        - max (*Union[float, int]*, `required`) : The upper bound of search space.

      - quniform : List[Union[float, int]]

        - min (*Union[float, int]*, `required`) : The lower bound of search space.
        - max (*Union[float, int]*, `required`) : The upper bound of search space.
        - step (*Union[float, int]*, `required`) : The unit value of search space.

      - loguniform : List[Union[float, int])

        - min (*Union[float, int]*, `required`) : The lower bound of search space.
        - max (*Union[float, int]*, `required`) : The upper bound of search space.
        - log_base (*Union[float, int]*, *default=10*) : The logarithm base.

      - qloguniform : List[Union[float, int]]

        - min (*Union[float, int]*, `required`) : The lower bound of search space
        - max (*Union[float, int]*, `required`) : The upper bound of search space
        - step (*Union[float, int]*, `required`) : The unit value of search space
        - log_base (*Union[float, int]*, *default=10*) : The logarithm base.

      - choice : List[Any]

        - vaule : values to be chosen from candidates.

- **metric** (*str*, *default='mAP*') - Name of the metric that will be used to evaluate the performance of each trial. The hyperparameter optimization algorithm will aim to maximize or minimize this metric depending on the value of the mode hyperparameter. The default value is 'mAP'.

- **mode** (*str*, *default='max*') - Optimization mode for the metric. It determines whether the metric should be maximized or minimized. The possible values are 'max' and 'min', respectively. The default value is 'max'.

- **maximum_resource** (*int*, *default=None*) - Maximum number of training epochs for each trial. When the number of training epochs reaches this value, the training of the trial will stop. The default value is None.

- **minimum_resource** (*int*, *default=None*) - Minimum number of training epochs for each trial. Each trial will run for at least this many epochs, even if the performance of the model is not improving. The default value is None.
