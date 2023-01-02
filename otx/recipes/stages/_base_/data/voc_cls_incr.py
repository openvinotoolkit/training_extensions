_base_ = ["./data_seg.py"]

__dataset_type = "SegIncrVOCDataset"
__data_root = "data/pascal_voc"

data = dict(
    train=dict(
        type="RepeatDataset",
        times=1,
        dataset=dict(
            type=__dataset_type,
            data_root=__data_root,
            img_dir="train/img",
            ann_dir="train/anno",
            split="train.txt",
            classes=["background", "person", "car"],
        ),
    ),
    val=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir="val/img",
        ann_dir="val/anno",
        split="val.txt",
        classes=["background", "person", "car"],
    ),
    test=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir="test/img",
        ann_dir="test/anno",
        split="test.txt",
        classes=["background", "person", "car"],
    ),
)
