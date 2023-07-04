_base_ = ["./data_seg.py"]

__dataset_type = "CustomDataset"
__data_root = "tests/assets/common_semantic_segmentation_dataset"

data = dict(
    train=dict(
        type="RepeatDataset",
        times=1,
        dataset=dict(
            type=__dataset_type,
            data_root=__data_root,
            img_dir="train/images",
            ann_dir="train/masks",
            classes=["background", "person"],
        ),
    ),
    val=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir="val/images",
        ann_dir="val/masks",
        classes=["background", "person"],
    ),
    test=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir="val/images",
        ann_dir="val/masks",
        classes=["background", "person"],
    ),
)
