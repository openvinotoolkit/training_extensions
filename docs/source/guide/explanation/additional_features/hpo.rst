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
