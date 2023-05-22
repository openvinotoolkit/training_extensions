"""MMDeploy config of EfficientNetB2B model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

scale_ir_input = True

ir_config = dict(
    output_names=["boxes", "labels", "masks"],
<<<<<<< HEAD
    input_shape=(1024, 1024),
=======
    # input_shape=(1024, 1024),
>>>>>>> a67ffdb8aa084b7b989f07a0463a1ea4b3a8cf8b
)

backend_config = dict(
    # dynamic batch causes forever running openvino process
<<<<<<< HEAD
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 1024, 1024]))],
=======
    # model_inputs=[dict(opt_shapes=dict(input=[1, 3, 1024, 1024]))],
>>>>>>> a67ffdb8aa084b7b989f07a0463a1ea4b3a8cf8b
)
