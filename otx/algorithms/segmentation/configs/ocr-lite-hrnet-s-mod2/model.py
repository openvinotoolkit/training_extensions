_base_ = [
    "../../../../../external/model-preparation-algorithm/submodule/models/segmentation/ocr_litehrnet_s_mod2.yaml",
]

load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions\
/models/custom_semantic_segmentation/litehrnetsv2_imagenet1k_rsc.pth"
fp16 = dict(loss_scale=512.0)
