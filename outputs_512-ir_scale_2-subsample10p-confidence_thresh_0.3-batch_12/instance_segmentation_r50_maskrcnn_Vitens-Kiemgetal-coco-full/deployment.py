ir_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['image'],
    output_names=['boxes', 'labels', 'masks'],
    input_shape=(512, 512),
    optimize=False,
    dynamic_axes=dict(
        input=dict({
            0: 'batch',
            1: 'channel',
            2: 'height',
            3: 'width'
        }),
        boxes=dict({
            0: 'batch',
            1: 'num_dets'
        }),
        labels=dict({
            0: 'batch',
            1: 'num_dets'
        }),
        masks=dict({
            0: 'batch',
            1: 'num_dets',
            2: 'height',
            3: 'width'
        })))
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
        export_postprocess_mask=False))
backend_config = dict(
    type='openvino',
    mo_options=None,
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 800, 1344]))])
input_data = dict(shape=(128, 128, 3), file_path=None)
scale_ir_input = True
