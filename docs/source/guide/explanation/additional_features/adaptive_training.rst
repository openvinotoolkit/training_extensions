Adaptive Training
==================

Adaptive-training focuses to adjust the number of iterations or interval for the validation to achieve the fast training. 
In the small data regime, we don't need to validate the model at every epoch since there are a few iterations at a single epoch. 
To handle this, we have implemented module named ``AdaptiveTrainScheduling``. This callback controls the interval of the validation to do faster training.

.. note::
    ``AdaptiveTrainScheduling`` changes the interval of the validation, evaluation and updating learning rate by checking the number of dataset.


.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.algo.callbacks.adaptive_train_scheduling import AdaptiveTrainScheduling
            
            engine.train(callbacks=[AdaptiveTrainScheduling()])

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... --callbacks otx.algo.callbacks.adaptive_train_scheduling.AdaptiveTrainScheduling
