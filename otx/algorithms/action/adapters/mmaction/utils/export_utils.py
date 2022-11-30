# Copyright (c) OpenMMLab. All rights reserved.
from subprocess import DEVNULL, CalledProcessError, run  # nosec

import numpy as np
import torch

try:
    import onnx
    import onnxruntime as rt

    rt.set_default_logger_severity(3)
except ImportError as e:
    raise ImportError(f"Please install onnx and onnxruntime first. {e}")

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError("please update mmcv to version>=1.0.4")


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(
    model,
    input_shape=[1, 1, 3, 8, 224, 224],
    opset_version=11,
    show=False,
    output_file="tmp.onnx",
    verify=False,
    softmax=False,
    is_localizer=False,
):
    """Convert pytorch model to onnx model.
    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, "forward_dummy"):
        from functools import partial

        model.forward = partial(model.forward_dummy, softmax=softmax)
    elif hasattr(model, "_forward") and is_localizer:
        model.forward = model._forward
    else:
        raise NotImplementedError("Please implement the forward method for exporting.")

    model.cpu().eval()

    input_tensor = torch.randn(input_shape)

    register_extra_symbolics(opset_version)
    torch.onnx.export(
        model,
        input_tensor,
        output_file,
        input_names=["data"],
        output_names=["logits"],
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version,
    )

    print(f"Successfully exported ONNX model: {output_file}")
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(input_tensor)[0].detach().numpy()

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(None, {net_feed_input[0]: input_tensor.detach().numpy()})[0]
        # only compare part of results
        random_class = np.random.randint(pytorch_result.shape[1])
        assert np.allclose(
            pytorch_result[:, random_class], onnx_result[:, random_class]
        ), "The outputs are different between Pytorch and ONNX"
        print("The numerical values are same between Pytorch and ONNX")


def _get_mo_cmd():
    for mo_cmd in ("mo", "mo.py"):
        try:
            run([mo_cmd, "-h"], stdout=DEVNULL, stderr=DEVNULL, check=True)
            return mo_cmd
        except CalledProcessError:
            pass
    raise RuntimeError("OpenVINO Model Optimizer is not found or configured improperly")


def onnx2openvino(
    cfg,
    onnx_model_path,
    output_dir_path,
    input_shape=None,
    input_format="bgr",
    precision="FP32",
    pruning_transformation=False,
):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(onnx_model_path)
    output_names = set(out.name for out in onnx_model.graph.output)
    # Clear names of the nodes that produce network's output blobs.
    for node in onnx_model.graph.node:
        if output_names.intersection(node.output):
            node.ClearField("name")
    onnx.save(onnx_model, onnx_model_path)
    output_names = ",".join(output_names)

    mo_cmd = _get_mo_cmd()

    normalize = None
    for pipeline in cfg.data.test.pipeline:
        if pipeline["type"] == "Normalize":
            normalize = pipeline
            break
    assert normalize, "Could not find normalize parameters in datapipeline"

    mean_values = normalize["mean"]
    scale_values = normalize["std"]
    command_line = [
        mo_cmd,
        f"--input_model={onnx_model_path}",
        f"--mean_values={mean_values}",
        f"--scale_values={scale_values}",
        f"--output_dir={output_dir_path}",
        f"--output={output_names}",
        f"--data_type={precision}",
        f"--source_layout=??c???",
    ]

    assert input_format.lower() in ["bgr", "rgb"]

    if input_shape is not None:
        command_line.append(f"--input_shape={input_shape}")
    if normalize["to_bgr"]:
        command_line.append("--reverse_input_channels")
    if pruning_transformation:
        command_line.extend(["--transform", "Pruning"])

    print(" ".join(command_line))

    run(command_line, check=True)


def export_model(model, config, onnx_model_path=None, output_dir_path=None):
    pytorch2onnx(model, output_file=onnx_model_path)
    onnx2openvino(config, onnx_model_path, output_dir_path)
