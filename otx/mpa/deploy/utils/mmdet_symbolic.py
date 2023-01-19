# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import wraps

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import (
    get_registered_op,
    is_registered_op,
    register_op,
)

from .operations_domain import add_domain


def py_symbolic(op_name=None, namespace="mmdet_custom", adapter=None):
    """
    The py_symbolic decorator allows associating a function with a custom symbolic function
    that defines its representation in a computational graph.

    A symbolic function cannot receive a collection of tensors as arguments.
    If your custom function takes a collection of tensors as arguments,
    then you need to implement an argument converter (adapter) from the collection
    and pass it to the decorator.

    Args:
        op_name (str): Operation name, must match the registered operation name.
        namespace (str): Namespace for this operation.
        adapter (function): Function for converting arguments.

    Adapter conventions:
        1. The adapter must have the same signature as the wrapped function.
        2. The values, returned by the adapter, must match the called symbolic function.
        3. Return value order:
            tensor values (collections are not supported)
            constant parameters (can be passed using a dictionary)

    Usage example:
        1. Implement a custom operation. For example 'custom_op'.
        2. Implement a symbolic function to represent the custom_op in
            a computation graph. For example 'custom_op_symbolic'.
        3. Register the operation before export:
            register_op('custom_op_name', custom_op_symbolic, namespace, opset)
        4. Decorate the custom operation:
            @py_symbolic(op_name='custom_op_name')
            def custom_op(...):
        5. If you need to convert custom function arguments to symbolic function arguments,
            you can implement a converter and pass it to the decorator:
            @py_symbolic(op_name='custom_op_name', adapter=converter)
    """

    def decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            name = op_name if op_name is not None else func.__name__
            opset = sym_help._export_onnx_opset_version

            if is_registered_op(name, namespace, opset):

                class XFunction(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, *xargs):
                        return func(*args, **kwargs)

                    @staticmethod
                    def symbolic(g, *xargs):
                        symb = get_registered_op(name, namespace, opset)
                        if adapter is not None:
                            return symb(g, *xargs, **adapter_kwargs)
                        return symb(g, *xargs)

                if adapter is not None:
                    adapter_args, adapter_kwargs = adapter(*args, **kwargs)
                    return XFunction.apply(*adapter_args)
                return XFunction.apply(*args)
            else:
                return func(*args, **kwargs)

        return wrapped_function

    return decorator


def view_as_symbolic(g, self, other):
    from torch.onnx.symbolic_opset9 import reshape_as

    return reshape_as(g, self, other)


@parse_args("v", "v", "i", "i", "i", "none")
def topk_symbolic(g, self, k, dim, largest, sorted, out=None):

    from torch.onnx.symbolic_opset9 import unsqueeze

    def reverse(x):
        from torch.onnx.symbolic_opset9 import reshape, size, transpose

        y = transpose(g, x, 0, dim)
        shape = g.op("Shape", y)
        y = reshape(g, y, [0, 1, -1])
        n = size(g, y, g.op("Constant", value_t=torch.LongTensor([0])))
        y = g.op("ReverseSequence", y, n, batch_axis_i=1, time_axis_i=0)
        y = reshape(g, y, shape)
        y = transpose(g, y, 0, dim)
        return y

    k = sym_help._maybe_get_const(k, "i")
    if not sym_help._is_value(k):
        k = g.op("Constant", value_t=torch.tensor(k, dtype=torch.int64))
    k = unsqueeze(g, k, 0)

    do_reverse = False
    if sym_help._export_onnx_opset_version <= 10 and not largest:
        do_reverse = True
        largest = True

    top_values, top_indices = sym_help._topk_helper(g, self, k, dim, largest, sorted, out)

    if sym_help._export_onnx_opset_version <= 10 and do_reverse:
        top_values = reverse(top_values)
        top_indices = reverse(top_indices)
    return top_values, top_indices


def multiclass_nms_core_symbolic(g, multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):

    from torch.onnx.symbolic_opset9 import reshape, squeeze
    from torch.onnx.symbolic_opset10 import _slice

    def cast(x, dtype):
        return g.op("Cast", x, to_i=sym_help.cast_pytorch_to_onnx[dtype])

    def get_size(x, dim):
        shape = g.op("Shape", x)
        dim = _slice(g, shape, axes=[0], starts=[dim], ends=[dim + 1])
        return cast(dim, "Long")

    nms_op_type = nms_cfg.get("type", "nms")
    assert nms_op_type == "nms"
    assert "iou_thr" in nms_cfg
    iou_threshold = nms_cfg["iou_thr"]
    assert 0 <= iou_threshold <= 1

    # Transpose and reshape input tensors to fit ONNX NonMaxSuppression.
    multi_bboxes = reshape(g, multi_bboxes, [0, -1, 4])
    multi_bboxes = g.op("Transpose", multi_bboxes, perm_i=[1, 0, 2])

    batches_num = get_size(multi_bboxes, 0)
    spatial_num = get_size(multi_bboxes, 1)

    multi_scores = g.op("Transpose", multi_scores, perm_i=[1, 0])
    scores_shape = g.op(
        "Concat",
        batches_num,
        g.op("Constant", value_t=torch.LongTensor([-1])),
        spatial_num,
        axis_i=0,
    )
    multi_scores = reshape(g, multi_scores, scores_shape)
    classes_num = get_size(multi_scores, 1)

    assert max_num > 0

    indices = g.op(
        "NonMaxSuppression",
        multi_bboxes,
        multi_scores,
        g.op("Constant", value_t=torch.LongTensor([max_num])),
        g.op("Constant", value_t=torch.FloatTensor([iou_threshold])),
        g.op("Constant", value_t=torch.FloatTensor([score_thr])),
    )

    # Flatten bboxes and scores.
    multi_bboxes_flat = reshape(g, multi_bboxes, [-1, 4])
    multi_scores_flat = reshape(
        g,
        multi_scores,
        [
            -1,
        ],
    )

    # Flatten indices.
    batch_indices = _slice(g, indices, axes=[1], starts=[0], ends=[1])
    class_indices = _slice(g, indices, axes=[1], starts=[1], ends=[2])
    box_indices = _slice(g, indices, axes=[1], starts=[2], ends=[3])

    def add(*args, dtype="Long"):
        x = g.op("Add", args[0], args[1])
        if dtype is not None:
            x = cast(x, dtype)
        return x

    def mul(*args, dtype="Long"):
        x = g.op("Mul", args[0], args[1])
        if dtype is not None:
            x = cast(x, dtype)
        return x

    flat_box_indices = add(mul(batch_indices, spatial_num), box_indices)
    flat_score_indices = add(
        mul(add(mul(batch_indices, classes_num), class_indices), spatial_num),
        box_indices,
    )

    # Select bboxes.
    out_bboxes = reshape(g, g.op("Gather", multi_bboxes_flat, flat_box_indices, axis_i=0), [-1, 4])
    out_scores = reshape(g, g.op("Gather", multi_scores_flat, flat_score_indices, axis_i=0), [-1, 1])
    # Having either batch size or number of classes here equal to one is the limitation of implementation.
    class_indices = reshape(g, cast(add(class_indices, batch_indices), "Float"), [-1, 1])

    # Combine bboxes, scores and labels into a single tensor.
    # This a workaround for a PyTorch bug (feature?),
    # limiting ONNX operations to output only single tensor.
    out_combined_bboxes = g.op("Concat", out_bboxes, out_scores, class_indices, axis_i=1)

    # Get the top scored bboxes only.
    elements_num = sym_help._size_helper(g, out_scores, dim=g.op("Constant", value_t=torch.LongTensor([0])))
    max_num = g.op("Constant", value_t=torch.LongTensor([max_num]))
    if sym_help._export_onnx_opset_version < 12:
        kn = g.op("Concat", max_num, elements_num, axis_i=0)
        kn = g.op("ReduceMin", kn, keepdims_i=0)
    else:
        kn = g.op("Min", max_num, elements_num)
    _, top_indices = sym_help._topk_helper(g, out_scores, kn, dim=0)
    # top_indices = squeeze(g, top_indices, dim=1)
    top_indices = reshape(
        g,
        top_indices,
        [
            -1,
        ],
    )
    out_combined_bboxes = g.op("Gather", out_combined_bboxes, top_indices, axis_i=0)

    return out_combined_bboxes


def roi_feature_extractor_symbolics(g, rois, *feats, output_size=1, featmap_strides=1, sample_num=1):
    from torch.onnx.symbolic_helper import _slice_helper

    rois = _slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
    roi_feats = g.op(
        add_domain("ExperimentalDetectronROIFeatureExtractor"),
        rois,
        *feats,
        output_size_i=output_size,
        pyramid_scales_i=featmap_strides,
        sampling_ratio_i=sample_num,
        image_id_i=0,
        distribute_rois_between_levels_i=1,
        preserve_rois_order_i=0,
        aligned_i=1,
        outputs=1,
    )
    return roi_feats


def patch_dcn_symbolic():
    from mmcv.ops.deform_conv import DeformConv2dFunction

    def symbolic(
        g,
        input,
        offset,
        weight,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
        bias=False,
        im2col_step=32,
    ):
        assert not bias
        assert groups == 1
        kh, kw = weight.type().sizes()[2:]
        return g.op(
            add_domain("DeformableConv2D"),
            input,
            offset,
            weight,
            strides_i=stride,
            pads_i=[p for pair in zip(padding, padding) for p in pair],
            dilations_i=dilation,
            groups_i=groups,
            deformable_groups_i=deform_groups,
            kernel_shape_i=[kh, kw],
        )

    DeformConv2dFunction.symbolic = staticmethod(symbolic)


def patch_nms_aten_to():
    """
    This patch fixes the following bug:
        RuntimeError: 0 INTERNAL ASSERT FAILED at /pytorch/torch/csrc/jit/ir/alias_analysis.cpp:318,
        please report a bug to PyTorch. We don't have an op for aten::to but it isn't a special case.
        Argument types: Tensor, None, int, Device, bool, bool, bool, None,
    In PyTorch version >= 1.7.1 it should be fixed.
    """
    from packaging import version

    if version.parse(torch.__version__) < version.parse("1.7.1"):
        from mmdet.ops.nms import NMSop

        original_forward = NMSop.forward

        def forward(ctx, bboxes, scores, iou_threshold, score_threshold, max_num, offset):
            with torch.jit._disable_tracing():
                inds = original_forward(ctx, bboxes, scores, iou_threshold, score_threshold, max_num, offset)
            return inds

        NMSop.forward = staticmethod(forward)


def patch_conv_ws():
    import torch.nn.functional as F
    from mmcv.cnn.bricks.conv_ws import ConvWS2d

    def normalize_weights(weight, eps=1e-5):
        c_in = weight.size(0)
        weight_flat = weight.view(c_in, -1)
        mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
        std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
        weight = (weight - mean) / (std + eps)
        return weight

    def forward(self, x):
        weight = normalize_weights(self.weight, self.eps)
        if torch.onnx.is_in_onnx_export():
            weight = weight.data
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    ConvWS2d.forward = forward


def nms_symbolic_with_score_thr(g, bboxes, scores, iou_threshold, score_threshold, max_num, offset):
    """
    This function adds 'score_threshold' and 'max_output_boxes_per_class' to ONNX::NonMaxSuppression.
    It should be removed after adding support for 'score_threshold' and 'max_num' in MMCV::NMSop.
    """
    from mmcv.onnx import is_custom_op_loaded

    has_custom_op = is_custom_op_loaded()
    if has_custom_op:
        return g.op(
            "mmcv::NonMaxSuppression",
            bboxes,
            scores,
            iou_threshold_f=float(iou_threshold),
            offset_i=int(offset),
        )
    else:
        from torch.onnx.symbolic_opset9 import select, squeeze, unsqueeze

        boxes = unsqueeze(g, bboxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        if not sym_help._is_value(max_num):
            max_num = g.op("Constant", value_t=torch.tensor(max_num, dtype=torch.long))
        max_output_per_class = max_num
        iou_threshold = g.op("Constant", value_t=torch.tensor([iou_threshold], dtype=torch.float))
        score_threshold = g.op("Constant", value_t=torch.tensor([score_threshold], dtype=torch.float))
        nms_out = g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_per_class,
            iou_threshold,
            score_threshold,
        )
        return squeeze(
            g,
            select(
                g,
                nms_out,
                1,
                g.op("Constant", value_t=torch.tensor([2], dtype=torch.long)),
            ),
            1,
        )


def register_extra_symbolics_for_openvino(opset=11):
    assert opset >= 10
    #  register_op("view_as", view_as_symbolic, "", opset)
    #  register_op("topk", topk_symbolic, "", opset)
    # register_op('multiclass_nms_core', multiclass_nms_core_symbolic, 'mmdet_custom', opset)
    register_op("roi_feature_extractor", roi_feature_extractor_symbolics, "mmdet_custom", opset)

    #  patch_nms_aten_to()
    #  patch_conv_ws()
    patch_dcn_symbolic()


def unregister_extra_symbolics_for_openvino(opset=11):
    from torch.onnx.symbolic_registry import _registry

    #  _registry.get(('', opset), {}).pop('view_as', None)
    #  _registry.get(('', opset), {}).pop('topk', None)
    _registry.get(("mmdet_custom", opset), {}).pop("roi_feature_extractor", None)
