"""MMDeploy config of MaskRCNN models for Instance-Seg Task."""

_base_ = ["../base/base_instance_segmentation.py"]

scale_ir_input = True

onnx_config = dict(
    output_names=["boxes", "labels", "masks"],
)
