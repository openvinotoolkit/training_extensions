from .fcn_mask_head import FCNMaskHead
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class HTCMaskHead(FCNMaskHead):

    def __init__(self, *args, **kwargs):
        super(HTCMaskHead, self).__init__(*args, **kwargs)
        self.conv_res = ConvModule(
            self.conv_out_channels,
            self.conv_out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

    def init_weights(self):
        super(HTCMaskHead, self).init_weights()
        self.conv_res.init_weights()

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        if res_feat is not None:
            res_feat = self.conv_res(res_feat)
            x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        res_feat = x
        outs = []
        if return_logits:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]
