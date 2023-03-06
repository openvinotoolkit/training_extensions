"""MobileNet-V3-large-1 for multi-class MMDeploy config."""

_base_ = ["../base/deployments/base_classification_dynamic.py"]

ir_config = dict(
    output_names=["logits"],
)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 224, 224]))],
)
