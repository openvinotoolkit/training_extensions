"""MMDeploy config of DeepLabV3Plus_ResNet18 model for Segmentation Task."""

_base_ = ["../base/deployments/base_segmentation_dynamic.py"]

ir_config = dict(
    output_names=["output"],
)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 544, 544]))],
)
