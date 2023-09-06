"""Base Action classification mmdeply cfg."""

ir_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="end2end.onnx",
    input_names=["data"],
    output_names=["logits"],
    input_shape=None,
    optimize=False,
)
codebase_config = dict(type="mmaction", task="VideoRecognition")
backend_config = dict(
    type="openvino",
    mo_options=dict(args=dict({"--source_layout": "?bctwh"})),
    model_inputs=[dict(opt_shapes=dict(input=[1, 1, 3, 32, 224, 224]))],
)
