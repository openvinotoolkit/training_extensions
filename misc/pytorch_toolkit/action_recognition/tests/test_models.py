from argparse import Namespace

from torch import nn

from action_recognition.model import create_model
from action_recognition.models.backbone.mobilenetv2 import InvertedResidual
from action_recognition.models.video_transformer import VideoTransformer


def _make_args(**kwargs):
    kwargs.setdefault('fp16', False)
    kwargs.setdefault('cuda', False)
    kwargs.setdefault('hidden_size', 512)
    kwargs.setdefault('sample_duration', 16)
    kwargs.setdefault('n_classes', 10)
    kwargs.setdefault('sample_size', 224)
    kwargs.setdefault('pretrain_path', False)
    kwargs.setdefault('layer_norm', False)
    kwargs.setdefault('resume_path', None)
    return Namespace(**kwargs)


class TestCreateModel:
    def test_create_resnet34_vtn(self):
        args = _make_args()
        model, _ = create_model(args, 'resnet34_vtn')
        num_convs = len([l for l in model.resnet.modules() if isinstance(l, nn.Conv2d)])

        assert isinstance(model, VideoTransformer)
        assert 36 == num_convs

    def test_create_mobilenetv2_vtn(self):
        args = _make_args()
        model, _ = create_model(args, 'mobilenetv2_vtn')
        num_convs = len([l for l in model.resnet.modules() if isinstance(l, nn.Conv2d)])
        num_mobnet_blocks = len([l for l in model.resnet.modules() if isinstance(l, InvertedResidual)])

        assert isinstance(model, VideoTransformer)
        assert 52 == num_convs
        assert 17 == num_mobnet_blocks

    def test_select_encoder_from_args(self):
        args = _make_args(encoder='mobilenetv2')
        model, _ = create_model(args, 'vtn')
        num_convs = len([l for l in model.resnet.modules() if isinstance(l, nn.Conv2d)])
        num_mobnet_blocks = len([l for l in model.resnet.modules() if isinstance(l, InvertedResidual)])

        assert isinstance(model, VideoTransformer)
        assert 52 == num_convs
        assert 17 == num_mobnet_blocks
