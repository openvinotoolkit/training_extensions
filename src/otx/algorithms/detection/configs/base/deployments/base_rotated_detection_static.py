"""Rotated Detection models static deploy config."""

ir_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="end2end.onnx",
    input_names=["image"],
    output_names=["boxes", "labels"],
    input_shape=None,
    # TODO
    # optimizing onnx graph mess up NNCF graph at some point
    # where we need to look into
    optimize=False,
)

codebase_config = dict(
    type="mmrotate",
    task="RotatedDetection",
    post_processing=dict(
        score_threshold=0.05, iou_threshold=0.1, pre_top_k=3000, keep_top_k=2000, max_output_boxes_per_class=2000
    ),
)

backend_config = dict(
    type="openvino",
    mo_options=None,
)

input_data = dict(
    shape=(128, 128, 3),
    file_path=None,
)
