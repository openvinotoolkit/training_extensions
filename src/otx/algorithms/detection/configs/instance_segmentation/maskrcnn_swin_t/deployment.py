"""MMDployment config of MaskRCNN-SwinT-FP16 model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
    # NOTE: Its necessary to use opset 11 as squeeze does not work in
    # roi_align_default ONNX in opset 16. Beaware of this for future update.
    opset_version=11,
)

scale_ir_input = False
