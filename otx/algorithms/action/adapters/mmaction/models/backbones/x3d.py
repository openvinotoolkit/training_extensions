"""Adapt mmaction X3D model into OTX."""
from mmaction.models import BACKBONES
from mmaction.models.backbones import X3D as mmX3D
from mmaction.models.backbones.x3d import BlockX3D
from mmaction.utils import get_root_logger
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm
from torch import nn


@BACKBONES.register_module(force=True)
class X3D(mmX3D):
    """X3D for OTX."""

    # This is temporary solution because otx.mmedet is different with latest mmdet
    # pylint: disable=invalid-name
    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from scratch."""

        if self.pretrained is not None:
            pretrained = self.pretrined
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info(f"load model from: {pretrained}")

            load_checkpoint(self, pretrained, strict=False, logger=logger)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BlockX3D):
                        constant_init(m.conv3.bn, 0)
        else:
            raise TypeError("pretrained must be a str or None")
