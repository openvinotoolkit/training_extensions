Hyperparameters Optimization
============================

Hyper-parameter optimization (HPO) can be a time-consuming process, even with state-of-the-art off-the-shelf libraries. OpenVINO™ Training Extensions makes HPO faster and easier by providing an easy-to-use interface and automatic configuration.

With OpenVINO™ Training Extensions, you can run hyper-parameter optimization by simply adding a time constraint parameter. The auto-config feature automatically sets internal control parameters, guaranteeing that HPO will finish within the given time constraint.

OpenVINO™ Training Extensions provides both sequential and parallel methods, making it scalable for different training environments. If you have multiple GPUs, you can accelerate HPO by utilizing all available GPU resources.

Key features of OpenVINO™ Training Extensions include:

- **Easy usability** : By using time as the control parameter, OpenVINO™ Training Extensions offers a straightforward and intuitive interface for users.

- **Auto-config** : The automatic configuration feature sets internal control parameters automatically, ensuring that HPO finishes within the given time constraint.

- **Scalability** : OpenVINO™ Training Extensions offers both sequential and parallel methods, making it scalable for different training environments. If you have multiple GPUs, you can take advantage of all available GPU resources to accelerate HPO.

You can run HPO by just adding **--enable-hpo** argument as below:

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine

            engine = Engine(data_root="<path_to_data_root>")
            engine.train(run_hpo=True)

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... --run_hpo True


=========
Algorithm
=========

If you have abundant GPU resources, it's better to run HPO in parallel.
In that case, `ASHA <https://arxiv.org/pdf/1810.05934.pdf>`_ is a good choice.
Currently, OpenVINO™ Training Extensions uses the ASHA algorithm.

The **Asynchronous Successive Halving Algorithm (ASHA)** is a hyperparameter optimization algorithm that is based on Successive Halving Algorithm (SHA) but is designed to be more efficient in a parallel computing environment. It is used to efficiently search for the best hyperparameters for machine learning models.

ASHA involves running multiple trials in parallel and evaluating them based on their validation metrics. It starts by running many trials for a short time, with only the best-performing trials advancing to the next round. In each subsequent round, the number of trials is reduced, and the amount of time spent on each trial is increased. This process is repeated until only one trial remains.

ASHA is designed to be more efficient than SHA in parallel computing environments because it allows for asynchronous training of the trials. This means that each trial can be trained independently of the others, and they do not have to wait for all the other trials to be complete before advancing to the next round. This reduces the amount of time it takes to complete the optimization process.

ASHA also includes a technique called Hyperband, which is used to determine how much time to allocate to each trial in each round. Hyperband allocates more time to the best-performing trials, with the amount of time allocated decreasing as the performance of the trials decreases. This technique helps to reduce the overall amount of training time required to find the best hyperparameters.

*********************************************
How to configure hyper-parameter optimization
*********************************************

You can configure HPO using argument named *hpo_config*.
It's *HpoConfig* dataclass which inlcudes hyperparameters to optimize, expected time ratio and more.
If HPO is executed without confgiguration, default value of *HpoConfig* is used and learning rate and batch size are set as hyper parameterse to optimize.
Here is how to configure using *hpo_config*.

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine
            from otx.core.config.hpo import HpoConfig

            hpo_config = HpoConfig(
                search_space={
                    "optimizer.lr" : {
                        "type" : "uniform",
                        "min" : 0.001,
                        "max" : 0.1,
                    }
                },
                expected_time_ratio=6,
            )

            engine = Engine(data_root="<path_to_data_root>")
            engine.train(run_hpo=True, hpo_config=hpo_config)

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... --run_hpo True  --hpo_config.expected_time_ratio 6


As above, you can set HPO configuration by both API and CLI. Including ones here, you can configure various parameters of HPO.
You can configure almost parameters using CLI except search space of hyper parameters.
But search space requires dictionray format, so it can be set only on API.
Here is explanation of all HPO configuration.


- **search_space** (*list[dict[str, Any]]*, `required`) - Hyper parameter search space to find. It should be list of dictionary. Each dictionary has a hyperparameter name as the key and param_type and range as the values. You can optimize any learning parameters of each task.

  - **Keys of each hyper parameter**

    - **type** (*str*, `required`) : Hyper parameter search space type. It must be one of the following:

      - uniform : Samples a float value uniformly between the lower and upper bounds.
      - quniform : Samples a quantized float value uniformly between the lower and upper bounds.
      - loguniform : Samples a float value after scaling search space by logarithm scale.
      - qloguniform : Samples a quantized float value after scaling the search space by logarithm scale.
      - choice : Samples a categorical value.

    - **range** (*list*, `required`)

      - uniform : list[float | int]

        - min (*float | int*, `required`) : The lower bound of search space.
        - max (*float | int*, `required`) : The upper bound of search space.

      - quniform : list[float | int]

        - min (*float | int*, `required`) : The lower bound of search space.
        - max (*float | int*, `required`) : The upper bound of search space.
        - step (*float | int*, `required`) : The unit value of search space.

      - loguniform : list[float | int]

        - min (*float | int*, `required`) : The lower bound of search space.
        - max (*float | int*, `required`) : The upper bound of search space.
        - log_base (*float | int*, *default=10*) : The logarithm base.

      - qloguniform : List[Union[float, int]]

        - min (*float | int*, `required`) : The lower bound of search space
        - max (*float | int*, `required`) : The upper bound of search space
        - step (*float | int*, `required`) : The unit value of search space
        - log_base (*float | int*, *default=10*) : The logarithm base.

      - choice : *list | tuple*

        - vaule : values to choose as candidates.


- **save_path** (*str*, *default='None'*) Path to save a HPO result.

- **mode** (*str*, *default='max'*) - Optimization mode for the metric. It determines whether the metric should be maximized or minimized. The possible values are 'max' and 'min', respectively.

- **num_workers** (*int*, *default=1*) How many trials will be executed in parallel.

- **expected_time_ratio** (*int*, *default=4*) How many times to use for HPO compared to training time.

- **maximum_resource** (*int*, *default=None*) - Maximum number of training epochs for each trial. When the training epochs reaches this value, the trial stop to train.

- **minimum_resource** (*int*, *default=None*) - Minimum number of training epochs for each trial. Each trial will run at least this epochs, even if the performance of the model is not improving.

- **prior_hyper_parameters** (*dict | list[dict]*, *default=None*) Hyper parameters to try first.

- **acceptable_additional_time_ratio** (*float | int*, *default=1.0*) How much ratio of additional time is acceptable.

- **reduction_factor** (*int*, *default=3*) How many trials to promote to next rung. Only top 1 / reduction_factor of rung trials can be promoted.

- **asynchronous_bracket** (*bool*, *default=True*) Whether to operate SHA asynchronously.

- **asynchronous_sha** (*bool*, *default=True*) Whether SHAs(brackets) are running parallelly or not.

**reduction_factor**, **asynchronous_bracket** and **asynchronous_sha** are HyperBand hyper parameters. If you want to know them more, please refer `ASHA <https://arxiv.org/pdf/1810.05934.pdf>`_.
