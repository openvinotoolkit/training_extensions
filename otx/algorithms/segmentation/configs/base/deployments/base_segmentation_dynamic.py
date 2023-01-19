"""Segmentation models dynamic deploy config."""

_base_ = ["./base_segmentation_static.py"]

ir_config = dict(
    dynamic_axes={
        "input": {
            0: "batch",
            2: "height",
            3: "width",
        },
        "output": {
            0: "batch",
            2: "height",
            3: "width",
        },
    },
)
