from torchvision.ops.roi_align import RoIAlign
from torch.autograd import Function
import torch


class RoIAlignMMCV(Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(g, *args, **kwargs):
        return RoIAlignMMCV.origin_output

    @staticmethod
    def symbolic(
        g, 
        input, 
        rois, 
        output_size, 
        spatial_scale, 
        sampling_ratio,
        pool_mode, 
        aligned,
    ):
        from torch.onnx import TensorProtoDataType
        from torch.onnx.symbolic_opset9 import sub

        def _select(g, self, dim, index):
            return g.op('Gather', self, index, axis_i=dim)

        # batch_indices = rois[:, 0].long()
        batch_indices = _select(
            g, rois, 1,
            g.op('Constant', value_t=torch.tensor([0], dtype=torch.long)))
        batch_indices = g.op('Squeeze', batch_indices, axes_i=[1])
        batch_indices = g.op(
            'Cast', batch_indices, to_i=TensorProtoDataType.INT64)
        # rois = rois[:, 1:]
        rois = _select(
            g, rois, 1,
            g.op(
                'Constant',
                value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))

        if aligned:
            # rois -= 0.5/spatial_scale
            aligned_offset = g.op(
                'Constant',
                value_t=torch.tensor([0.5 / spatial_scale],
                                     dtype=torch.float32))
            rois = sub(g, rois, aligned_offset)
        # roi align
        return g.op(
            'RoiAlign',
            input,
            rois,
            batch_indices,
            output_height_i=output_size[0],
            output_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=max(0, sampling_ratio),
            mode_s=pool_mode)



class OTXRoIAlign(RoIAlign):

    def export(self, input, rois):
        state = torch._C._get_tracing_state()  # noqa: SLF001
        origin_output = self(input, rois)
        RoIAlignMMCV.origin_output = origin_output
        torch._C._set_tracing_state(state)  # noqa: SLF001

        output_size = self.output_size
        spatial_scale = self.spatial_scale
        sampling_ratio = self.sampling_ratio
        pool_mode = "avg"
        aligned = self.aligned

        return RoIAlignMMCV.apply(
            input, 
            rois, 
            output_size, 
            spatial_scale, 
            sampling_ratio,
            pool_mode,
            aligned,
        )