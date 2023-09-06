"""Classification models dynamic deploy config."""

_base_ = ["./base_classification_static.py"]

ir_config = dict(
    dynamic_axes={
        "data": {
            0: "batch",
            1: "channel",
            2: "height",
            3: "width",
        },
        "logits": {
            0: "batch",
        },
    }
)
