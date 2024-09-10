Class-Incremental Sampler
===========================

This sampler is a sampler that creates an effective batch.
For default setting, the square root of (number of old data/number of new data) is used as the ratio of old data.

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.algo.samplers.class_incremental_sampler import ClassIncrementalSampler

            dataset = OTXDataset(...)
            class_incr_sampler = ClassIncrementalSampler(
                dataset=dataset,
                batch_size=32,
                old_classes=["car", "truck"],
                new_classes=["bus"],
            )

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... \
                                 --data.train_subset.sampler.class_path otx.algo.samplers.class_incremental_sampler.ClassIncrementalSampler \
                                 --data.train_subset.sampler.init_args.old_classes '[car,truck]' \
                                 --data.train_subset.sampler.init_args.new_classes '[bus]'


Balanced Sampler
===========================

This sampler is a sampler that creates an effective batch.
It helps ensure balanced sampling by class based on the distribution of class labels during supervised learning.


.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.algo.samplers.balanced_sampler import BalancedSampler

            dataset = OTXDataset(...)
            class_incr_sampler = BalancedSampler(
                dataset=dataset,
            )

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... \
                                 --data.train_subset.sampler.class_path otx.algo.samplers.balanced_sampler.BalancedSampler
