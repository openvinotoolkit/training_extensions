"""Segmentor config for semi-supervised learning."""

model = dict(
    type="MeanTeacherSegmentor",
    orig_type="OTXEncoderDecoder",
    unsup_weight=0.1,
    train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)),
    test_cfg=dict(mode="whole", output_scale=5.0),
)
__norm_cfg = dict(type="BN", requires_grad=True)
