"""MMDployment config of SOLOv2 model for Instance-Seg Task."""

ir_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=14,
    save_file="end2end.onnx",
    input_names=["image"],
    output_names=["masks", "labels", "scores"],
    input_shape=None,
    optimize=False,
    dynamic_axes=dict(
        input=dict({0: "batch", 1: "channel", 2: "height", 3: "width"}),
        masks=dict({0: "batch", 1: "num_dets", 2: "height", 3: "width"}),
        labels=dict({0: "batch", 1: "num_dets"}),
        scores=dict({0: "batch", 1: "num_dets"}),
    ),
)
codebase_config = dict(
    type="mmdet",
    task="ObjectDetection",
    model_type="end2end",
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
        export_postprocess_mask=False,
    ),
)
backend_config = dict(type="openvino", mo_options=None)
input_data = dict(shape=(128, 128, 3), file_path=None)
scale_ir_input = True
