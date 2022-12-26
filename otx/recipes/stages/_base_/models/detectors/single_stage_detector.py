_base_ = "./detector.py"

model = dict(
    type="SingleStageDetector",
    train_cfg=dict(
        assigner=dict(type="MaxIoUAssigner", min_pos_iou=0.0, ignore_iof_thr=-1, gt_max_assign_all=False),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False,
    ),
    test_cfg=dict(nms=dict(type="nms", iou_threshold=0.45), min_bbox_size=0, score_thr=0.02, max_per_img=200),
)
