_base_ = ["./data_seg.py"]

__dataset_type = "KvasirDataset"
__data_root = "data/Kvasir-SEG"

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            type="PseudoSemanticSegDataset",
            orig_type=__dataset_type,
            data_root=__data_root,
            img_dir="images",
            ann_dir="masks",
            split="train_label_8_seed_0.txt",
            unlabeled_split="train_label_8_unlabel_seed_0.txt",
        )
    ),
    val=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir="images",
        ann_dir="masks",
        split="val.txt",
    ),
    test=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir="images",
        ann_dir="masks",
        split="val.txt",
    ),
)
