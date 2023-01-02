_base_ = ["./pipelines/selfsl.py"]

__resize_target_size = 224
__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

__train_pipeline_v0 = {{_base_.train_pipeline_v0}}
__train_pipeline_v1 = {{_base_.train_pipeline_v1}}

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    pipeline_options=dict(RandomResizedCrop=dict(size=__resize_target_size), Normalize=dict(**__img_norm_cfg)),
    train=dict(
        type="SelfSLDataset",
        datasource=dict(cfg=dict(type=""), reg=None),
        pipeline=dict(
            view0=__train_pipeline_v0,
            view1=__train_pipeline_v1,
        ),
    ),
)

train_pipeline_v0 = dict(_delete_=True)
train_pipeline_v1 = dict(_delete_=True)
