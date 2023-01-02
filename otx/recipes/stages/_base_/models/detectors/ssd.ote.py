_base_ = "./single_stage_detector.py"

__width_mult = 1.0

model = dict(
    bbox_head=dict(
        type="SSDHead",
        num_classes=80,
        in_channels=(int(__width_mult * 96), int(__width_mult * 320)),
        anchor_generator=dict(
            type="SSDAnchorGeneratorClustered",
            strides=(16, 32),
            reclustering_anchors=True,
            widths=[
                [38.641007923271076, 92.49516032784699, 271.4234764938237, 141.53469410876247],
                [206.04136086566515, 386.6542727907841, 716.9892752215089, 453.75609561761405, 788.4629155558277],
            ],
            heights=[
                [48.9243877087132, 147.73088476194903, 158.23569788707474, 324.14510379107367],
                [587.6216059488938, 381.60024152086544, 323.5988913027747, 702.7486097568518, 741.4865860938451],
            ],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2),
        ),
        depthwise_heads=True,
        depthwise_heads_activations="relu",
        loss_balancing=False,
    ),
    train_cfg=dict(
        assigner=dict(
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
        ),
        use_giou=False,
        use_focal=False,
    ),
)
load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-992x736.pth"
