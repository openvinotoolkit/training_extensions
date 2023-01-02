_base_ = "./segmentor.py"

__norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    decode_head=dict(
        type="FCNHead",
        in_channels=40,
        in_index=0,
        channels=40,
        input_transform=None,
        kernel_size=1,
        num_convs=0,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=21,
        norm_cfg=__norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss"),
    ),
    # model training and testing settings
    train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)),
    test_cfg=dict(mode="whole"),
)

find_unused_parameters = True
