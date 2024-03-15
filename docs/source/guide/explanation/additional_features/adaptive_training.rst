Adaptive Training
==================

Adaptive-training focuses to adjust the number of iterations or interval for the validation to achieve the fast training. 
In the small data regime, we don't need to validate the model at every epoch since there are a few iterations at a single epoch. 
To handle this, we have implemented two modules named ``AdaptiveTrainingHook`` and ``AdaptiveRepeatDataHook``. Both of them controls the interval of the validation to do faster training.

.. note::
    1. ``AdaptiveTrainingHook`` changes the interval of the validation, evaluation and updating learning rate by checking the number of dataset.  
    2. ``AdaptiveRepeatDataHook`` changes the repeats of the dataset by pathcing the sampler.
