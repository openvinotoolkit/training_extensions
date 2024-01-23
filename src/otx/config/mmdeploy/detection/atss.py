"""MMDeploy config of ATSS model for Detection Task."""

_base_ = ["../base/base_detection.py"]

onnx_config = dict(
    output_names=["boxes", "labels"],
)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 736, 992]))],
)
