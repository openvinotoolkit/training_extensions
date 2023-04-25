_base_ = "./encoder_decoder.ote.py"

__norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="OTXEncoderDecoder",
    pretrained=None,
    decode_head=dict(
        type="CustomFCNHead",
        in_channels=40,
        in_index=0,
        channels=40,
        input_transform=None,
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2,
        norm_cfg=__norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            )
        ],
    ),
    test_cfg=dict(
        mode="whole",
        output_scale=3.0,
    ),
)
