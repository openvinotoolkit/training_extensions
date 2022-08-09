_base_ = [
  '../../../submodule/models/segmentation/ocr_litehrnet_x_mod3.yaml',
]

load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnetxv3_imagenet1k_rsc.pth'
optimizer_config = dict(
    type='Fp16OptimizerHook', 
    loss_scale=512.,
    grad_clip=dict(
        max_norm=40,
        norm_type=2
    )
  )