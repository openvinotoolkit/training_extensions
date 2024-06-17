"""MMDployment config of Resnet model for Rotated-Detection Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
)

backend_config = dict(
    # dynamic batch causes forever running openvino process
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 1024, 1024]))],
)
