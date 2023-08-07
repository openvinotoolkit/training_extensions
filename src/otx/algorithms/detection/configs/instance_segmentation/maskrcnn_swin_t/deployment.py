"""MMDployment config of MaskRCNN-SwinT-FP16 model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
)

scale_ir_input = False
