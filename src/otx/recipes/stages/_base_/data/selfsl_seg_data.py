"""Base Self-SL dataset."""

__resize_target_size = 224
__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

__train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="TwoCropTransform",
        view0=[
            dict(type="RandomResizedCrop", size=__resize_target_size),
            dict(type="RandomFlip", prob=0.5, direction="horizontal"),
            dict(
                type="ProbCompose",
                transforms=[dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                probs=[0.8],
            ),
            dict(type="RandomGrayscale", p=0.2),
            dict(type="GaussianBlur", kernel_size=23),
            dict(type="Normalize", **__img_norm_cfg),
        ],
        view1=[
            dict(type="RandomResizedCrop", size=__resize_target_size),
            dict(type="RandomFlip", prob=0.5, direction="horizontal"),
            dict(
                type="ProbCompose",
                transforms=[dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                probs=[0.8],
            ),
            dict(type="RandomGrayscale", p=0.2),
            dict(type="ProbCompose", transforms=[dict(type="GaussianBlur", kernel_size=23)], probs=[0.1]),
            dict(type="ProbCompose", transforms=[dict(type="Solarization", threshold=128)], probs=[0.2]),
            dict(type="Normalize", **__img_norm_cfg),
        ],
    ),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

data = dict(samples_per_gpu=16, workers_per_gpu=2, train=dict(type="OTXSegDataset", pipeline=__train_pipeline))
