"""EfficientNet-B0 config for semi-supervised multi-label classification."""

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/classification/multilabel/semisl.yaml", "../../base/models/efficientnet.py"]

model = dict(
    task="classification",
    type="SemiSLMultilabelClassifier",
    backbone=dict(
        version="b0",
    ),
    head=dict(
        type="SemiLinearMultilabelClsHead",
        use_dynamic_loss_weighting=True,
        unlabeled_coef=0.1,
        in_channels=-1,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        normalized=True,
        scale=7.0,
        loss=dict(type="AsymmetricAngularLossWithIgnore", gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        aux_loss=dict(
            type="BarlowTwinsLoss",
            off_diag_penality=1.0 / 128.0,
            loss_weight=1.0,
        ),
    ),
)

fp16 = dict(loss_scale=512.0)
