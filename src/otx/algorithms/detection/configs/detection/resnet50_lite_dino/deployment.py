"""MMDeploy config of Lite DINO model for Detection Task."""

_base_ = ["../../base/deployments/base_detection_dynamic.py"]

ir_config = dict(
    output_names=["boxes", "labels"],
    opset_version=16,
)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 800, 1333]))],
)
