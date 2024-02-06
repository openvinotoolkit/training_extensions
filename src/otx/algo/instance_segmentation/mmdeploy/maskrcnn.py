"""MMDeploy config of MaskRCNN models for Instance-Seg Task.

reference: https://github.com/open-mmlab/mmdeploy/
"""

_base_ = ["./base_instance_segmentation.py"]

scale_ir_input = True

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
)
