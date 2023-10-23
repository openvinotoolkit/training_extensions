"""MMDployment config of RTMDet-Inst model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]


ir_config = dict(
    output_names=["boxes", "labels", "masks"],
)

codebase_config = dict(
    post_processing=dict(
        max_output_boxes_per_class=100,
        pre_top_k=300,
    )
)

scale_ir_input = False
