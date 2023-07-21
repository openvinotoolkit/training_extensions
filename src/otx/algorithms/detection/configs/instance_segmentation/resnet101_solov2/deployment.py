"""MMDployment config of SOLOv2 model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

ir_config = dict(
    opset_version=14,
    output_names=["boxes", "labels", "masks"],
)

scale_ir_input = False
