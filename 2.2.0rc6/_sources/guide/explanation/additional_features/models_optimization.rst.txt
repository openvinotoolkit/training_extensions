Models Optimization
===================

OpenVINO™ Training Extensions provides optimization algorithm: `Post-Training Quantization tool (PTQ) <https://github.com/openvinotoolkit/nncf#post-training-quantization>`_.

*******************************
Post-Training Quantization Tool
*******************************

PTQ is designed to optimize the inference of models by applying post-training methods that do not require model retraining or fine-tuning. If you want to know more details about how PTQ works and to be more familiar with model optimization methods, please refer to `documentation <https://docs.openvino.ai/2023.2/ptq_introduction.html>`_.

To run Post-training quantization it is required to convert the model to OpenVINO™ intermediate representation (IR) first. To perform fast and accurate quantization we use ``DefaultQuantization Algorithm`` for each task. Please, refer to the `Tune quantization Parameters <https://docs.openvino.ai/2023.2/basic_quantization_flow.html#tune-quantization-parameters>`_ for further information about configuring the optimization.

Please, refer to our :doc:`dedicated tutorials <../../tutorials/base/how_to_train/index>` on how to optimize your model using PTQ.


.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine
            ...
            engine.optimize(checkpoint="<IR-checkpoint-path>")

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx optimize ... --checkpoint <IR-checkpoint-path>
