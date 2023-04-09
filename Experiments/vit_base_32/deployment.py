ir_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['data'],
    output_names=['logits'],
    input_shape=None,
    optimize=False,
    dynamic_axes=dict(
        data=dict({
            0: 'batch',
            1: 'channel',
            2: 'height',
            3: 'width'
        }),
        logits=dict({0: 'batch'})))
codebase_config = dict(type='mmcls', task='Classification')
backend_config = dict(
    type='openvino',
    mo_options=None,
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 224, 224]))])
input_data = dict(shape=(128, 128, 3), file_path=None)
