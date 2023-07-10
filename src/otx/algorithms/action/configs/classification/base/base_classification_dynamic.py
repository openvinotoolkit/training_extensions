"""Dynamic Action classification mmdeply cfg."""

_base_ = ["./base_classification_static.py"]

ir_config = dict(
    dynamic_axes=dict(
        data=dict({0: "batch"}),
        logits=dict({0: "batch"}),
    ),
)
