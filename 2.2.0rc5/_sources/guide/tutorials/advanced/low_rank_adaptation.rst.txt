LoRA: Low Rank Adaptation for Classification Tasks
===================================================

.. note::

    LoRA is only supported for VisionTransformer models.
    See the model in otx.algo.classification.vit.

Overview
--------

OpenVINO™ Training Extensions now supports Low Rank Adaptation (LoRA) for classification tasks using Transformer models. 
LoRA is a parameter-efficient approach to adapt pre-trained models by introducing low-rank matrices that capture important adaptations without the need to retrain the entire model.

Benefits of LoRA
----------------

- **Efficiency**: LoRA allows for efficient adaptation of large pre-trained models with minimal additional parameters.
- **Performance**: By focusing on key parameters, LoRA can achieve competitive performance with less computational overhead.
- **Flexibility**: LoRA can be applied to various parts of the transformer model, providing flexibility in model tuning.

How to Use LoRA in OpenVINO™ Training Extensions
------------------------------------------------

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.algo.classification.vit import VisionTransformerForMulticlassCls

            model = VisionTransformerForMulticlassCls(..., lora=True)

    .. tab-item:: CLI

        .. code-block:: bash

            (otx) ...$ otx train ... --model.lora True
