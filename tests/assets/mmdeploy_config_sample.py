"""Detection models base deploy config.

reference: https://github.com/open-mmlab/mmdeploy/
"""

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
    },
)

codebase_config = dict(
    type="mmdet",
    task="ObjectDetection",
    model_type="end2end",
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,  # for YOLOv3
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ),
)

backend_config = dict(
    type="openvino",
    mo_options=None,
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 800, 992]))],
)