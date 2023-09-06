"""Segmentation models static deploy config."""

ir_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="end2end.onnx",
    input_names=["input"],
    output_names=["output"],
    input_shape=None,
    # TODO
    # optimizing onnx graph mess up NNCF graph at some point
    # where we need to look into
    optimize=False,
)

codebase_config = dict(
    type="mmseg",
    task="Segmentation",
)

backend_config = dict(
    type="openvino",
    mo_options=None,
)

input_data = dict(
    shape=(128, 128, 3),
    file_path=None,
)
