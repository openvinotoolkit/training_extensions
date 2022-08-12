_base_ = [
  '../../../submodule/models/segmentation/ocr_litehrnet_x_mod3.yaml',
]

load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnetxv3_imagenet1k_rsc.pth'
fp16 = dict(loss_scale=512.)
