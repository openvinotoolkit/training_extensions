Fast Data Loading
=================

OpenVINO™ Training Extensions provides several ways to boost model training speed,
one of which is fast data loading.


=======
Caching
=======


*****************
In-Memory Caching
*****************
OpenVINO™ Training Extensions provides in-memory caching for decoded images in main memory.
If the batch size is large, such as for classification tasks, or if dataset contains
high-resolution images, image decoding can account for a non-negligible overhead
in data pre-processing.
One can enable in-memory caching for maximizing GPU utilization and reducing model
training time in those cases.


.. tab-set::

   .. tab-item:: API

      .. code-block:: python

         from otx.core.config.data import DataModuleConfig
         from otx.core.data.module import OTXDataModule

         data_config = DataModuleConfig(..., mem_cache_size="8GB")
         datamodule = OTXDataModule(..., config=data_config)

   .. tab-item:: CLI

      .. code-block:: shell

         (otx) ...$ otx train ... --data.config.mem_cache_size 8GB
