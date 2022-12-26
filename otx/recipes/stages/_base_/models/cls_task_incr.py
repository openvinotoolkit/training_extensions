_base_ = "./classifiers/ti_classifier.py"

model = dict(
    head=dict(
        in_channels=-1,
        distillation_loss=dict(type="LwfLoss", T=2.0, loss_weight=1.0),
        # tasks=dict(
        #     # Age=["Other", "Senior", "Kids", "Unknown"]
        # ),
        tasks=None,
        topk=1,
    )
)
