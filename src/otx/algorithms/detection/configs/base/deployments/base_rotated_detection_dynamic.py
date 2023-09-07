"""Rotated Detection models dynamic deploy config."""

_base_ = ["./base_rotated_detection_static.py"]

ir_config = dict(
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "boxes": {
            0: "batch",
            1: "num_dets",
        },
        "labels": {
            0: "batch",
            1: "num_dets",
        },
    },
)
