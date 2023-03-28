# model settings
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='ClassIncrEncoderDecoder',
    backbone=dict(
        type='MSCAN',
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)),
    test_cfg=dict(mode='slide', crop_size=(512,512), stride=(341, 341)))

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

load_from='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth'