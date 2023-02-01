__img_norm_cfg = dict(mean=None, std=None)
__resize_target_size = -1

train_pipeline_v0 = [
    dict(type="RandomResizedCrop", size=__resize_target_size),
    dict(type="RandomHorizontalFlip"),
    dict(
        type="RandomAppliedTrans",
        transforms=[dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8,
    ),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="RandomAppliedTrans", transforms=[dict(type="GaussianBlur", sigma_min=0.1, sigma_max=2.0)], p=1.0),
    dict(type="RandomAppliedTrans", transforms=[dict(type="Solarization")], p=0.0),
    dict(type="ToNumpy"),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

train_pipeline_v1 = [
    dict(type="RandomResizedCrop", size=__resize_target_size),
    dict(type="RandomHorizontalFlip"),
    dict(
        type="RandomAppliedTrans",
        transforms=[dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8,
    ),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="RandomAppliedTrans", transforms=[dict(type="GaussianBlur", sigma_min=0.1, sigma_max=2.0)], p=0.1),
    dict(type="RandomAppliedTrans", transforms=[dict(type="Solarization")], p=0.2),
    dict(type="ToNumpy"),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
