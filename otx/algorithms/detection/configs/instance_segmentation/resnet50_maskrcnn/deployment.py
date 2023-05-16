"""MMDployment config of Resnet model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

scale_ir_input = True

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
    input_shape=(1344, 800),
)

backend_config = dict(
    # dynamic batch causes forever running openvino process
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 800, 1344]))],
)
