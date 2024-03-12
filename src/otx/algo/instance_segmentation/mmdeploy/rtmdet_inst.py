"""MMDeploy config of MaskRCNN models for Instance-Seg Task.

reference: https://github.com/open-mmlab/mmdeploy/
"""

_base_ = ["./base_instance_segmentation.py"]

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
)

codebase_config = dict(
    post_processing=dict(
        max_output_boxes_per_class=100,
        pre_top_k=300,
    )
)
