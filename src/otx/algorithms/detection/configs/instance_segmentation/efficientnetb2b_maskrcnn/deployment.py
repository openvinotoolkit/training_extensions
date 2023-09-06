"""MMDeploy config of EfficientNetB2B model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

scale_ir_input = True

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
)
