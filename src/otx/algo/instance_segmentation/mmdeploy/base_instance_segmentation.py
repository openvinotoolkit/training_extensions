"""Instance segmentation models based deploy config.

reference: https://github.com/open-mmlab/mmdeploy/
"""

_base_ = ["../../detection/mmdeploy/base_detection.py"]

ir_config = dict(
    output_names=[
        "boxes",
        "labels",
        "masks",
    ],
    dynamic_axes={
        "image": {
            0: "batch",
            2: "height",
            3: "width",
        },
        "boxes": {
            0: "batch",
            1: "num_dets",
        },
        "labels": {
            0: "batch",
            1: "num_dets",
        },
        "masks": {
            0: "batch",
            1: "num_dets",
            2: "height",
            3: "width",
        },
    },
)

codebase_config = dict(
    post_processing=dict(
        export_postprocess_mask=False,
    ),
)
