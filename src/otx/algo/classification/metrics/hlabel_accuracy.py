from torchmetrics import Metric
from torchmetrics.classification import Accuracy, MultilabelAccuracy

class HLabelAccuracy(Metric):
    def __init__(
        self, 
        num_classes_multiclass,
        num_classes_multilabel,
        threshold_multilabel=0.5, 
        dist_sync_on_step=False
    ):
        super(HLabelAccuracy, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes_multiclass = num_classes_multiclass
        self.num_classes_multilabel = num_classes_multilabel
        self.threshold_multilabel = threshold_multilabel

        # Multiclass classification accuracy metric
        self.multiclass_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes_multiclass)

        # Multilabel classification accuracy metrics
        self.multilabel_accuracy = MultilabelAccuracy(
            num_labels=num_classes_multilabel,
            threshold=threshold_multilabel,
            average="micro"
        )

    def update(self, preds, target):
        # Split preds into multiclass and multilabel parts
        preds_multiclass = preds[:, :self.num_classes_multiclass]
        target_multiclass = target[:, :self.num_classes_multiclass] 
        
        preds_multilabel = preds[:, self.num_classes_multiclass:]
        target_multilabel = target[:, self.num_classes_multiclass:]
        
        # Exclude -1 label  
        multiclass_mask = target_multiclass > 0
        
        # Multiclass update
        self.multiclass_accuracy.update(preds_multiclass[multiclass_mask], target_multiclass[multiclass_mask])

        # Multilabel update
        self.multilabel_accuracy.update(preds_multilabel, target_multilabel)
        
    def compute(self):
        # Calculate multiclass and multilabel metrics
        multiclass_acc = self.multiclass_accuracy.compute()
        multilabel_acc = self.multilabel_accuracy.compute()

        # Combine the metrics as per your preference
        combined_accuracy = (multiclass_acc + multilabel_acc) / 2

        return combined_accuracy