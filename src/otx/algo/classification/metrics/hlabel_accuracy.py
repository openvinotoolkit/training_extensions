import torch
from torchmetrics import Metric
from torchmetrics.classification import Accuracy, MultilabelAccuracy, MultilabelAveragePrecision


class HLabelAccuracy(Metric):
    def __init__(
        self, 
        num_multiclass_heads: int,
        num_multilabel_classes: int,
        threshold_multilabel: float = 0.5,
        head_idx_to_logits_range: dict[str, tuple[int, int]] | None = None,
    ):
        """Custom accuracy metric for h-label classification.
        
        Args:
            num_multiclass_heads (int): Number of multi-class heads.
            num_multilabel_classes (int): Number of multi-label classes.
            threshold_multilabel (float): Predictions with scores under the thresholds 
                                          are considered as negative. Defaults to 0.5.
            head_idx_to_logits_range (dict[str, tuple[int, int]]): The range of logits which represents
                                                                   the number of classes for each heads.
        """
        super(HLabelAccuracy, self).__init__()

        self.num_multiclass_heads = num_multiclass_heads
        self.num_multilabel_classes = num_multilabel_classes
        self.threshold_multilabel = threshold_multilabel
        self.head_idx_to_logits_range = head_idx_to_logits_range

        # Multiclass classification accuracy will be defined later
        self.multiclass_head_accuracy: list[Accuracy] | None = None
        
        # Multilabel classification accuracy metrics
        if num_multilabel_classes > 0:
            # self.multilabel_accuracy = MultilabelAveragePrecision(
            #     num_labels=self.num_multilabel_classes, average='macro'
            # )
            self.multilabel_accuracy = MultilabelAccuracy(
                num_labels=self.num_multilabel_classes,
                threshold=0.5,
                average='macro'
            )

        if num_multiclass_heads == 0:
            msg = "The number of multiclass heads should be larger than 0"
            raise ValueError(msg)
        
    def _set_multiclass_head_accuracy(self, device: str) -> None:
        """Set the multiclass metric for each multiclass heads."""
        self.multiclass_head_accuracy = [
            Accuracy(
                task="multiclass", 
                num_classes=int(head_range[1]-head_range[0])
            ).to(device) for head_range in self.head_idx_to_logits_range.values()
        ]
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets."""
        if not self.multiclass_head_accuracy:
            self._set_multiclass_head_accuracy(preds.device)
        
        # Split preds into multiclass and multilabel parts
        for head_idx in range(self.num_multiclass_heads):
            preds_multiclass = preds[:, head_idx]
            target_multiclass = target[:, head_idx]
            multiclass_mask = target_multiclass > 0
            self.multiclass_head_accuracy[head_idx].update(
                preds_multiclass[multiclass_mask], target_multiclass[multiclass_mask]
            )
            
        if self.num_multilabel_classes > 0: 
            # Split preds into multiclass and multilabel parts
            preds_multilabel = preds[:, self.num_multiclass_heads:]
            target_multilabel = target[:, self.num_multiclass_heads:]
            # Multilabel update
            self.multilabel_accuracy.update(preds_multilabel, target_multilabel)
        
    def compute(self) -> torch.Tensor:
        """Compute the final statistics."""
        # Calculate multiclass and multilabel metrics
        multiclass_accs = []
        for acc in self.multiclass_head_accuracy:
            multiclass_accs.append(acc.compute())
        multiclass_accs = torch.mean(torch.stack(multiclass_accs))
        
        if self.num_multilabel_classes > 0: 
            multilabel_acc = self.multilabel_accuracy.compute()
            
            return (multiclass_accs + multilabel_acc) / 2
        
        return multiclass_accs