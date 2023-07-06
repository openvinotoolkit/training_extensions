_base_ = ["./classifiers/classifier.py"]

model = dict(
    type="SemiSLClassifier",
    pretrained=None,
    backbone=dict(),
    head=dict(
        type="SemiSLClsHead",
    ),
)
