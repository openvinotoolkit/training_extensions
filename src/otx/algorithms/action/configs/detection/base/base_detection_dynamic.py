"""Base Action detection mmdeply cfg."""

_base_ = ["./base_detection_static.py"]

ir_config = dict(
    dynamic_axes=dict(
        input=dict({0: "batch", 1: "channel", 2: "clip_len", 3: "height", 4: "width"}),
        dets=dict({0: "batch", 1: "num_dets"}),
        labels=dict({0: "batch", 1: "num_dets"}),
    ),
)
