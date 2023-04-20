"""MMDeploy config of EfficientNetB2B model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

ir_scale_factor = 2

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
    input_shape=(1024 * ir_scale_factor, 1024 * ir_scale_factor),
)

backend_config = dict(
    # dynamic batch causes forever running openvino process
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 1024 * ir_scale_factor, 1024 * ir_scale_factor]))],
)
