_base_ = ["./semisl.py"]

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=0.0001,
    paramwise_cfg={"bias_decay_mult ": 0.0, "norm_decay_mult ": 0.0},
)
optimizer_config = dict(_delete_=True)
# learning policy
lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=300,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=1e-6,
    by_epoch=False,
)

custom_hooks = [
    dict(
        type="DualModelEMAHook",
        momentum=0.99,
        start_epoch=5,
        src_model_name="model_s",
        dst_model_name="model_t",
    ),
]