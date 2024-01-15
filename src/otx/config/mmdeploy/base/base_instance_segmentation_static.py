"""Instance segmentation models static deploy config."""

_base_ = ["./base_detection_static.py"]

ir_config = dict(
    output_names=[
        "boxes",
        "labels",
        "masks",
    ]
)

codebase_config = dict(
    post_processing=dict(
        export_postprocess_mask=False,
    )
)
